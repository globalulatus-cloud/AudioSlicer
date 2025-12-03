import streamlit as st
import numpy as np
import os
from pydub import AudioSegment
from datetime import datetime, timedelta
import json
import zipfile
import io
from tqdm import tqdm

# ------------------------------------------
# Utility Functions
# ------------------------------------------

def format_time(ms):
    """Format milliseconds into HH:MM:SS.sss"""
    sec = ms / 1000
    d = datetime(1,1,1) + timedelta(seconds=sec)
    return f"{d.hour:02d}:{d.minute:02d}:{d.second:02d}.{int(ms%1000):03d}"


# ------------------------------------------
# Fast, Chunk-Based Energy Silence Detection
# ------------------------------------------

def detect_silence_segments(sound, min_silence_ms, threshold, step_ms):
    """
    sound           → AudioSegment
    min_silence_ms  → minimum silent chunk length
    threshold       → silence threshold (dBFS)
    step_ms         → step window
    returns list of (start_ms, end_ms)
    """

    silence_starts = []
    silence_current = None

    for i in range(0, len(sound), step_ms):
        chunk = sound[i:i + step_ms]
        if chunk.dBFS < threshold:
            # silence started
            if silence_current is None:
                silence_current = i
        else:
            # silence ended
            if silence_current is not None:
                duration = i - silence_current
                if duration >= min_silence_ms:
                    silence_starts.append((silence_current, i))
                silence_current = None

    # Handle last segment
    if silence_current is not None:
        if len(sound) - silence_current >= min_silence_ms:
            silence_starts.append((silence_current, len(sound)))

    return silence_starts


# ------------------------------------------
# Streamlit UI
# ------------------------------------------

st.title("Optimized Audio Silence Splitter (Large File Support)")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

min_silence = st.number_input("Min Silence Length (seconds)", value=0.5)
threshold = st.number_input("Silence Threshold (dBFS)", value=-40.0)
step = st.number_input("Step Duration (ms)", value=10)

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    # Save temp
    temp_dir = "/tmp/slicer"
    os.makedirs(temp_dir, exist_ok=True)
    input_path = os.path.join(temp_dir, "input.wav")

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("Loading audio (very large files supported)...")

    sound = AudioSegment.from_wav(input_path)

    min_silence_ms = int(min_silence * 1000)
    step_ms = int(step)

    st.write("Detecting silences...")

    silence_segments = detect_silence_segments(
        sound,
        min_silence_ms=min_silence_ms,
        threshold=threshold,
        step_ms=step_ms
    )

    st.write(f"Detected {len(silence_segments)} silence points")

    # Compute cut points
    cut_points = [0] + [end for (_, end) in silence_segments] + [len(sound)]
    cut_points = sorted(set(cut_points))  # remove duplicates

    segments = []
    for i in range(len(cut_points) - 1):
        start = cut_points[i]
        end = cut_points[i+1]
        segments.append((i, start, end))

    st.write(f"Total segments created: {len(segments)}")

    # ---------------------------------------
    # Export segments into temp folder
    # ---------------------------------------
    output_dir = os.path.join(temp_dir, "splits")
    os.makedirs(output_dir, exist_ok=True)

    json_data = {}

    for idx, start, end in segments:
        seg = sound[start:end]
        out_path = os.path.join(output_dir, f"{uploaded_file.name[:-4]}_{idx:03d}.wav")
        seg.export(out_path, format="wav")

        json_data[idx] = {
            "start": format_time(start),
            "end": format_time(end)
        }

    # Write metadata JSON
    json_path = os.path.join(output_dir, uploaded_file.name[:-4] + ".json")
    with open(json_path, "w") as jf:
        json.dump(json_data, jf, indent=2)

    st.success("Audio successfully split!")

    # ---------------------------------------
    # Create ZIP for download
    # ---------------------------------------
    st.subheader("Download All Segments (ZIP)")

    memory_zip = io.BytesIO()
    with zipfile.ZipFile(memory_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(output_dir):
            fpath = os.path.join(output_dir, fname)
            zf.write(fpath, arcname=fname)

    memory_zip.seek(0)

    st.download_button(
        "Download ZIP",
        data=memory_zip,
        file_name="audio_segments.zip",
        mime="application/zip"
    )

    st.subheader("Download JSON Metadata")

    with open(json_path, "rb") as jf:
        st.download_button(
            "Download JSON",
            data=jf,
            file_name=os.path.basename(json_path)
        )
