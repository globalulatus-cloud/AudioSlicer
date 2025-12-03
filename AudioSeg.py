# AudioSeg.py -- FFmpeg-only silence-based splitter (no PyDub)
import streamlit as st
import os
import io
import json
import zipfile
import subprocess
import shlex
import re
from datetime import datetime, timedelta
import math
from shutil import which

# -----------------------
# Helpers
# -----------------------
def ensure_ffmpeg():
    ffmpeg = which("ffmpeg")
    ffprobe = which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise FileNotFoundError("ffmpeg/ffprobe not found. Add 'ffmpeg' to packages.txt (repo root) and redeploy.")
    return ffmpeg, ffprobe

def ms_to_hhmmss_ms(ms):
    seconds = ms / 1000.0
    d = datetime(1,1,1) + timedelta(seconds=seconds)
    return f"{d.hour:02d}:{d.minute:02d}:{d.second:02d}.{int(ms%1000):03d}"

def sec_to_hhmmss_ms(sec):
    ms = int(round(sec * 1000))
    return ms_to_hhmmss_ms(ms)

# Parse ffmpeg silencedetect output (stderr)
SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9]+\.[0-9]+)")
SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9]+\.[0-9]+)")
SILENCE_DURATION_RE = re.compile(r"silence_duration:\s*([0-9]+\.[0-9]+)")

def parse_silencedetect_output(stderr_text):
    """
    Returns list of silence intervals in seconds: [(start_s, end_s), ...]
    ffmpeg prints silence_start and silence_end lines; we convert them to intervals.
    """
    starts = []
    ends = []
    for line in stderr_text.splitlines():
        s_m = SILENCE_START_RE.search(line)
        if s_m:
            starts.append(float(s_m.group(1)))
        e_m = SILENCE_END_RE.search(line)
        if e_m:
            ends.append(float(e_m.group(1)))
    # Pair starts/ends carefully
    silences = []
    # There can be cases where file starts with silence_end (no silence_start) or ends with silence_start (no silence_end)
    # We'll pair sequentially: take earliest start -> next end after it; if missing, handle edges.
    i_s, i_e = 0, 0
    while i_s < len(starts) or i_e < len(ends):
        if i_s < len(starts) and (i_e >= len(ends) or starts[i_s] < ends[i_e]):
            # have a start, look for next end
            start = starts[i_s]
            if i_e < len(ends) and ends[i_e] > start:
                end = ends[i_e]
                silences.append((start, end))
                i_s += 1
                i_e += 1
            else:
                # no end found -> silence to EOF
                silences.append((start, None))
                i_s += 1
        else:
            # an end without a start -> treat as silence from 0 to end
            if i_e < len(ends):
                silences.append((0.0, ends[i_e]))
                i_e += 1
            else:
                break
    return silences

def compute_segments_from_silences(silences, total_duration_s, min_silence_len_s):
    """
    Given silences (list of (start_s, end_s) where end_s may be None meaning EOF),
    produce non-silent segments [(idx, seg_start_s, seg_end_s), ...].
    We also enforce that only silences >= min_silence_len_s are considered (ffmpeg may report shorter ones).
    """
    # Normalize: replace None end with total_duration_s
    normalized = []
    for s, e in silences:
        e = total_duration_s if e is None else e
        if e - s >= min_silence_len_s - 1e-6:  # accept if length >= threshold
            normalized.append((s, e))
    # Merge overlapping/adjacent silences
    normalized.sort()
    merged = []
    for s, e in normalized:
        if not merged:
            merged.append([s,e])
        else:
            last_s, last_e = merged[-1]
            if s <= last_e + 1e-3:
                merged[-1][1] = max(last_e, e)
            else:
                merged.append([s,e])
    # Build segments from merged silence intervals
    if not merged:
        return [(0, 0.0, total_duration_s)]
    points = [0.0]
    for s,e in merged:
        points.append(e)  # cut at the end of silence -> start of next speech
    points.append(total_duration_s)
    segments = []
    for i in range(len(points)-1):
        start = points[i]
        end = points[i+1]
        if end - start > 0.001:
            segments.append((i, start, end))
    return segments

def run_ffmpeg_silencedetect(input_path, silence_thresh_db, min_silence_len_s):
    """
    Calls ffmpeg -hide_banner -vn -af silencedetect to detect silences.
    Returns stderr output (string) and detected silences parsed list.
    """
    # ffmpeg filter wants duration in seconds (d=) and noise in dB e.g. -30dB
    d_val = float(min_silence_len_s)
    noise = f"{float(silence_thresh_db)}dB"
    # build command
    # -nostats and -hide_banner reduce noise; silencedetect outputs to stderr
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-y",
        "-i", input_path,
        "-af", f"silencedetect=noise={noise}:d={d_val}",
        "-f", "null",
        "-"  # write to null
    ]
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    _, stderr = proc.communicate()
    if proc.returncode not in (0, 1):  # ffmpeg returns 1 when silencedetect prints - but still valid; accept 0 or 1
        raise RuntimeError(f"ffmpeg silencedetect failed with returncode {proc.returncode}. stderr:\n{stderr}")
    silences = parse_silencedetect_output(stderr)
    return stderr, silences

def ffmpeg_extract_segment(input_path, start_s, end_s, out_path):
    """
    Extract the segment using ffmpeg. Writes a WAV file (PCM 16).
    Uses re-encoding to WAV PCM S16LE for maximum compatibility.
    """
    # Use -ss (seek) before -i for faster processing; but seeking before -i may be less accurate for some formats.
    # Use -accurate_seek with -ss after -i for accuracy, but slower. We'll put -ss before -i and also -to to limit
    # We'll use re-encoding to pcm_s16le for compatibility.
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-y",
        "-ss", f"{start_s}",
        "-to", f"{end_s}",
        "-i", input_path,
        "-acodec", "pcm_s16le",
        "-ar", "48000",   # standard output sample rate, keeps quality; adjust if you want original
        "-ac", "1",       # convert to mono for smaller size; change to "2" if you want stereo
        out_path
    ]
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    _, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg segment extraction failed for {start_s}-{end_s}. stderr:\n{stderr}")

# -----------------------
# Streamlit App
# -----------------------
st.set_page_config(page_title="FFmpeg Audio Splitter", layout="centered")
st.title("FFmpeg-based Audio Splitter (no PyDub)")

# Ensure ffmpeg available
try:
    ffmpeg_bin, ffprobe_bin = ensure_ffmpeg()
    st.text(f"ffmpeg: {os.path.basename(ffmpeg_bin)}  ffprobe: {os.path.basename(ffprobe_bin)}")
except Exception as e:
    st.error(str(e))
    st.stop()

uploaded_file = st.file_uploader("Upload audio (wav/mp3/flac/...)", type=None, help="Any format supported by ffmpeg. Large files OK if server allows.")
if uploaded_file is None:
    st.info("Upload an audio file to begin.")
    st.stop()

# User params
min_silence_sec = st.number_input("Min silence length (seconds) 'd' (ffmpeg)", min_value=0.05, value=0.6, step=0.05)
silence_thresh_db = st.number_input("Silence threshold (dB) (e.g. -40)", min_value=-80.0, max_value=-1.0, value=-40.0, step=0.5)
convert_to_mono = st.checkbox("Convert output segments to mono (recommended)", value=True)
output_sample_rate = st.selectbox("Output sample rate (Hz)", options=[16000, 32000, 44100, 48000], index=3)

# Save uploaded file to temp
tmp_dir = "/tmp/ffsplit"
os.makedirs(tmp_dir, exist_ok=True)
input_path = os.path.join(tmp_dir, uploaded_file.name)
with open(input_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

st.info("Probing file to get duration (ffprobe)...")
# get duration via ffprobe
try:
    cmd_probe = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path
    ]
    proc = subprocess.Popen(cmd_probe, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {err}")
    total_duration_s = float(out.strip())
except Exception as e:
    st.error(f"Could not probe file duration: {e}")
    st.stop()

st.write(f"File duration: {math.floor(total_duration_s)}.{int((total_duration_s-math.floor(total_duration_s))*1000):03d} seconds")

# Run silencedetect
st.info("Running ffmpeg silencedetect (this may take a while for long files)...")
with st.spinner("Detecting silences..."):
    try:
        stderr_output, silences = run_ffmpeg_silencedetect(
            input_path,
            silence_thresh_db=silence_thresh_db,
            min_silence_len_s=min_silence_sec
        )
    except Exception as e:
        st.error(f"silencedetect failed: {e}")
        st.stop()

st.write(f"Raw silence intervals detected: {len(silences)}")
# Convert silences into finalized segments
segments = compute_segments_from_silences(silences, total_duration_s, min_silence_sec)
st.write(f"Segments to be created: {len(segments)}")

# Show segments optionally
if st.checkbox("Show segments", value=False):
    for idx, s, e in segments:
        st.write(f"{idx:03d}: {sec_to_hhmmss_ms(s)} -> {sec_to_hhmmss_ms(e)}  ({(e-s):.3f}s)")

# Export segments
output_dir = os.path.join(tmp_dir, "splits")
os.makedirs(output_dir, exist_ok=True)
st.info("Extracting segments with ffmpeg (re-encoding to WAV PCM 16-bit)...")

for idx, s, e in segments:
    out_name = f"{os.path.splitext(uploaded_file.name)[0]}_{idx:03d}.wav"
    out_path = os.path.join(output_dir, out_name)
    # build extraction command with desired sample rate and mono/stereo
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-y",
        "-ss", f"{s}",
        "-to", f"{e}",
        "-i", input_path,
        "-acodec", "pcm_s16le",
        "-ar", str(output_sample_rate),
    ]
    if convert_to_mono:
        cmd += ["-ac", "1"]
    else:
        cmd += ["-ac", "2"]
    cmd += [out_path]
    # run
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _, err = proc.communicate()
    if proc.returncode != 0:
        st.error(f"Failed to extract segment {idx}: ffmpeg returned non-zero. stderr:\n{err}")
        st.stop()

# Write metadata JSON
meta = {}
for idx, s, e in segments:
    meta[str(idx)] = {
        "file": f"{os.path.splitext(uploaded_file.name)[0]}_{idx:03d}.wav",
        "start": sec_to_hhmmss_ms(s),
        "end": sec_to_hhmmss_ms(e),
        "duration_s": round(e - s, 3)
    }
meta_path = os.path.join(output_dir, f"{os.path.splitext(uploaded_file.name)[0]}.json")
with open(meta_path, "w") as jf:
    json.dump(meta, jf, indent=2)

st.success("Segments extracted and metadata written.")

# ZIP all outputs
zip_buf = io.BytesIO()
with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for fname in sorted(os.listdir(output_dir)):
        z.write(os.path.join(output_dir, fname), arcname=fname)
zip_buf.seek(0)

st.download_button("Download ZIP with segments + metadata", data=zip_buf, file_name=f"{os.path.splitext(uploaded_file.name)[0]}_segments.zip", mime="application/zip")

st.subheader("Individual downloads")
for fname in sorted(os.listdir(output_dir)):
    fp = os.path.join(output_dir, fname)
    with open(fp, "rb") as fh:
        st.download_button(label=f"Download {fname}", data=fh, file_name=fname, key=f"{fname}")

st.subheader("Metadata (JSON)")
with open(meta_path, "r") as jf:
    st.code(jf.read(), language="json")
