import streamlit as st
import numpy as np
import os
import json
from scipy.io import wavfile
from tqdm import tqdm
from datetime import datetime, timedelta

# ---------------------------------------------------
# Utility Functions
# ---------------------------------------------------

def GetTime(video_seconds):
    if video_seconds < 0:
        return "00:00:00.000"
    sec = timedelta(seconds=float(video_seconds))
    d = datetime(1,1,1) + sec
    return f"{str(d.hour).zfill(2)}:{str(d.minute).zfill(2)}:{str(d.second).zfill(2)}.001"

def GetTotalTime(video_seconds):
    sec = timedelta(seconds=float(video_seconds))
    d = datetime(1,1,1) + sec
    return f"{d.hour}:{d.minute}:{d.second}"

def windows(signal, window_size, step_size):
    for i_start in range(0, len(signal), step_size):
        i_end = i_start + window_size
        if i_end >= len(signal):
            break
        yield signal[i_start:i_end]

def energy(samples):
    return np.sum(np.power(samples, 2.0)) / float(len(samples))

def rising_edges(binary_signal):
    previous_value = 0
    index = 0
    for x in binary_signal:
        if x and not previous_value:
            yield index
        previous_value = x
        index += 1

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------

st.title("Audio Silence-Based Splitter")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

min_silence_length = st.number_input("Min Silence Length (seconds)", value=0.6)
silence_threshold = st.number_input("Silence Energy Threshold", value=1e-4, format="%.6f")
step_duration = st.number_input("Step Duration (seconds)", value=0.003)

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    # Save uploaded file safely for Streamlit Cloud
    temp_dir = "/tmp" if not os.path.exists("temp") else "temp"
    os.makedirs(temp_dir, exist_ok=True)
    input_path = os.path.join(temp_dir, "input.wav")

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Output directory
    output_dir = os.path.join(temp_dir, "splits")
    os.makedirs(output_dir, exist_ok=True)

    st.write("Processing...")

    # ---------------------------------------------------
    # Silence Detection + Splitting
    # ---------------------------------------------------
    sample_rate, samples = wavfile.read(input_path, mmap=True)

    max_amplitude = np.iinfo(samples.dtype).max
    max_energy = energy([max_amplitude])

    window_size = int(min_silence_length * sample_rate)
    step_size = int(step_duration * sample_rate)

    signal_windows = windows(samples, window_size, step_size)

    window_energy = (energy(w) / max_energy for w in tqdm(
        signal_windows,
        total=int(len(samples) / float(step_size))
    ))

    window_silence = (e > silence_threshold for e in window_energy)

    cut_times = (r * step_duration for r in rising_edges(window_silence))

    cut_samples = [int(t * sample_rate) for t in cut_times]
    cut_samples.append(-1)

    cut_ranges = [(i, cut_samples[i], cut_samples[i + 1]) for i in range(len(cut_samples) - 1)]

    base_name = uploaded_file.name.replace(".wav", "")
    video_sub = {
        str(i): [
            GetTime(cut_samples[i] / sample_rate),
            GetTime(cut_samples[i + 1] / sample_rate)
        ]
        for i in range(len(cut_samples) - 1)
    }

    st.write("Splitting audio...")

    output_files = []

    for i, start, stop in tqdm(cut_ranges):
        output_file_path = os.path.join(output_dir, f"{base_name}_{i:03d}.wav")
        wavfile.write(output_file_path, sample_rate, samples[start:stop])
        output_files.append(output_file_path)

    # Write JSON
    json_path = os.path.join(output_dir, base_name + ".json")
    with open(json_path, "w") as output_json:
        json.dump(video_sub, output_json)

    st.success("Audio successfully split!")

    # Provide download links
    st.subheader("Download Files")

    for fpath in output_files[:10]:   # Show first 10
        with open(fpath, "rb") as fp:
            st.download_button(
                label=f"Download {os.path.basename(fpath)}",
                data=fp,
                file_name=os.path.basename(fpath)
            )

    with open(json_path, "rb") as fp:
        st.download_button(
            label="Download JSON File",
            data=fp,
            file_name=os.path.basename(json_path)
        )
