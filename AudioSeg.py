import streamlit as st
import os
import io
import zipfile
import json
from datetime import datetime, timedelta
from pydub import AudioSegment, utils
from pydub.utils import mediainfo
import math

# -------------------------
# Helper functions
# -------------------------

def format_time_ms(ms):
    """Return HH:MM:SS.mmm for milliseconds."""
    seconds = ms / 1000.0
    d = datetime(1, 1, 1) + timedelta(seconds=seconds)
    ms_part = int(ms % 1000)
    return f"{d.hour:02d}:{d.minute:02d}:{d.second:02d}.{ms_part:03d}"

def ensure_ffmpeg_available():
    """Return tuple (ffmpeg_path, ffprobe_path). Raise if not found."""
    ffmpeg_path = utils.which("ffmpeg")
    ffprobe_path = utils.which("ffprobe")
    if not ffmpeg_path or not ffprobe_path:
        raise FileNotFoundError("ffmpeg or ffprobe not found. Ensure packages.txt/apt.txt includes ffmpeg.")
    return ffmpeg_path, ffprobe_path

def detect_silence_by_dbfs(sound, min_silence_ms, silence_thresh_db, step_ms):
    """
    Detect silent intervals using dBFS threshold over sliding windows.
    Returns list of (start_ms, end_ms) for silence intervals >= min_silence_ms.
    """
    silence_intervals = []
    silence_start = None
    L = len(sound)
    pos = 0
    while pos < L:
        window = sound[pos: pos + step_ms]
        # pydub chunk.dBFS can be -inf for pure silence, handle that
        dbfs = window.dBFS if window.dBFS != float("-inf") else -1000.0
        if dbfs <= silence_thresh_db:
            if silence_start is None:
                silence_start = pos
        else:
            if silence_start is not None:
                dur = pos - silence_start
                if dur >= min_silence_ms:
                    silence_intervals.append((silence_start, pos))
                silence_start = None
        pos += step_ms

    # trailing silence
    if silence_start is not None:
        dur = L - silence_start
        if dur >= min_silence_ms:
            silence_intervals.append((silence_start, L))

    return silence_intervals

def compute_segments_from_silences(silences, total_ms):
    """
    Given a list of silence intervals (start,end), produce non-silent segments:
    segments = [(seg_idx, start_ms, end_ms), ...]
    """
    if not silences:
        return [(0, 0, total_ms)]

    # merge / sort silences
    silences_sorted = sorted(silences)
    merged = []
    cur_s = silences_sorted[0][0]
    cur_e = silences_sorted[0][1]
    for s, e in silences_sorted[1:]:
        if s <= cur_e + 1:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    # build segments from merged silences
    points = [0]
    for s, e in merged:
        points.append(e)
    points.append(total_ms)

    segments = []
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        if end - start > 0:
            segments.append((i, start, end))
    return segments

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="PyDub Audio Splitter", layout="centered")
st.title("PyDub Audio Splitter (FFmpeg required)")

# check ffmpeg/ffprobe availability early and show friendly guidance if missing
try:
    ffmpeg_path, ffprobe_path = ensure_ffmpeg_available()
    st.text(f"ffmpeg available: {os.path.basename(ffmpeg_path)}  ffprobe: {os.path.basename(ffprobe_path)}")
except Exception as exc:
    st.error("ffmpeg / ffprobe not available in runtime. Add packages.txt with 'ffmpeg' to your repo root and redeploy.")
    st.stop()

uploaded_file = st.file_uploader("Upload WAV file (large files supported)", type=["wav", "mp3", "flac"], help="Supported formats: wav/mp3/flac (ffmpeg required).")
if uploaded_file is None:
    st.info("Upload an audio file to begin.")
    st.stop()

# user params
min_silence_sec = st.number_input("Min silence length (seconds)", value=0.5, min_value=0.05, step=0.05)
silence_thresh_db = st.number_input("Silence threshold (dBFS)", value=-40.0, step=0.5)
step_ms = st.number_input("Step window (ms)", value=10, min_value=1)

# prepare temp paths
temp_dir = "/tmp/slicer"
os.makedirs(temp_dir, exist_ok=True)
uploaded_basename = os.path.splitext(uploaded_file.name)[0]
input_path = os.path.join(temp_dir, uploaded_file.name)

# save uploaded file
with open(input_path, "wb") as fh:
    fh.write(uploaded_file.getbuffer())

st.info("Loading audio via ffmpeg (pydub). This may take a moment for large files...")

# load audio (wrapped in try/except to show friendly message)
try:
    audio = AudioSegment.from_file(input_path)
except Exception as e:
    st.error(f"Error loading audio: {e}")
    st.stop()

total_ms = len(audio)
st.write(f"Audio loaded: duration = {math.floor(total_ms/1000)}.{int(total_ms%1000):03d} seconds | channels={audio.channels} | frame_rate={audio.frame_rate}")

# detect silences
st.info("Detecting silences...")
min_silence_ms = int(min_silence_sec * 1000)

silences = detect_silence_by_dbfs(
    sound=audio,
    min_silence_ms=min_silence_ms,
    silence_thresh_db=silence_thresh_db,
    step_ms=int(step_ms)
)

st.write(f"Found {len(silences)} raw silence intervals (length >= {min_silence_ms} ms).")

# compute final segments
segments = compute_segments_from_silences(silences, total_ms)
st.write(f"Total segments after merging: {len(segments)}")

# show segments summary
if st.checkbox("Show segments table", value=False):
    for idx, s, e in segments:
        st.write(f"Segment {idx:03d}: {format_time_ms(s)} → {format_time_ms(e)}  ({(e-s)/1000:.3f}s)")

# export segments and metadata
output_dir = os.path.join(temp_dir, "splits")
os.makedirs(output_dir, exist_ok=True)

st.info("Exporting segments...")

metadata = {}
for idx, s, e in segments:
    out_name = f"{uploaded_basename}_{idx:03d}.wav"
    out_path = os.path.join(output_dir, out_name)
    seg = audio[s:e]
    try:
        seg.export(out_path, format="wav")
    except Exception as ex:
        st.error(f"Failed to export segment {idx}: {ex}")
        st.stop()
    metadata[str(idx)] = {
        "file": out_name,
        "start": format_time_ms(s),
        "end": format_time_ms(e),
        "duration_s": round((e - s) / 1000.0, 3)
    }

# write JSON metadata
meta_path = os.path.join(output_dir, f"{uploaded_basename}.json")
with open(meta_path, "w") as jf:
    json.dump(metadata, jf, indent=2)

st.success("Segments exported.")

# create in-memory ZIP
zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        zf.write(fpath, arcname=fname)
zip_buffer.seek(0)

st.download_button(
    label="Download ZIP of all segments",
    data=zip_buffer,
    file_name=f"{uploaded_basename}_segments.zip",
    mime="application/zip"
)

# allow single-file downloads as well
st.subheader("Individual segment downloads")
for fname in sorted(os.listdir(output_dir)):
    fpath = os.path.join(output_dir, fname)
    with open(fpath, "rb") as fh:
        st.download_button(
            label=f"Download {fname}",
            data=fh,
            file_name=fname,
            key=f"dl_{fname}"
        )

st.subheader("Metadata (JSON)")
with open(meta_path, "r") as jf:
    st.code(jf.read(), language="json")
