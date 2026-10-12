from pathlib import Path
from tempfile import TemporaryDirectory
from hashlib import sha256

import joblib
import pandas as pd
import streamlit as st

from src.seekaudio.config import load_config
from src.seekaudio.analysis import analyze_audio


st.set_page_config(page_title="seekaudio", page_icon="🎵")
st.markdown(
    """<style>
    [data-testid='stHeader'], [data-testid='stToolbar'],
    [data-testid='stStatusWidget'], .stAppDeployButton, #MainMenu {
        display: none !important;
    }
    </style>""",
    unsafe_allow_html=True,
)
st.title("seekaudio")
st.write("Upload a song to estimate its genre and explore its audio characteristics.")
config = load_config()
space_dir = Path(config.paths.space_dir)
if not (space_dir / "vectors.parquet").exists() or not (space_dir / "models.joblib").exists():
    st.info("Run python -m src.seekaudio run from the project folder first.")
    st.stop()


@st.cache_data
def read_table(path: str, modified: int) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_resource
def read_models(path: str, modified: int) -> dict:
    return joblib.load(path)


manifest_path = Path(config.paths.manifest)
vectors_path = space_dir / "vectors.parquet"
models_path = space_dir / "models.joblib"
manifest = read_table(str(manifest_path), manifest_path.stat().st_mtime_ns)
vectors = read_table(str(vectors_path), vectors_path.stat().st_mtime_ns)
models = read_models(str(models_path), models_path.stat().st_mtime_ns)
st.subheader("Upload your song")
uploaded = st.file_uploader(
    "Choose an audio file",
    type=["wav", "flac", "mp3"],
    help="Supported formats: WAV, FLAC, and MP3. We analyze a centered excerpt of your audio.",
)

upload_bytes = uploaded.getvalue() if uploaded is not None else None
signature = sha256(upload_bytes).hexdigest() if upload_bytes is not None else None
if st.session_state.get("query_signature") != signature:
    st.session_state.pop("analysis", None)
    st.session_state["query_signature"] = signature
if uploaded is not None:
    st.caption("Your uploaded audio")
    st.audio(upload_bytes)

if st.button("Analyze audio", type="primary", disabled=uploaded is None):
    try:
        with st.spinner("Analyzing your audio..."):
            with TemporaryDirectory(prefix="seekaudio-query-") as temporary_dir:
                audio_path = Path(temporary_dir) / Path(uploaded.name).name
                audio_path.write_bytes(upload_bytes)
                st.session_state["analysis"] = analyze_audio(audio_path, vectors, models, manifest)
    except (OSError, ValueError, RuntimeError) as error:
        st.session_state.pop("analysis", None)
        st.error(str(error))

if "analysis" in st.session_state:
    analysis = st.session_state["analysis"]
    with st.container(border=True):
        st.metric("Predicted genre", analysis["genre"].title())
        st.caption("Estimated from genre votes among the nearest catalog audio features; not a calibrated classifier.")
    if "mel_spectrogram_png" in analysis:
        st.subheader("Sound over time")
        st.image(analysis["mel_spectrogram_png"])
        st.caption(f"Mel spectrogram of the analyzed {analysis['analysis_duration']:g}-second excerpt. Brighter colors show stronger sound at each time and frequency. Colors are relative to this excerpt's peak, not absolute loudness.")
        st.download_button("Download spectrogram", analysis["mel_spectrogram_png"], file_name="mel_spectrogram.png", mime="image/png")
    st.subheader("Audio characteristics")
    duration_seconds = round(analysis["duration"])
    first = st.columns(3)
    first[0].metric("File duration", f"{duration_seconds // 60}:{duration_seconds % 60:02d}")
    first[1].metric("Original sample rate", f"{analysis['sample_rate'] / 1000:g} kHz")
    channel_label = {1: "Mono", 2: "Stereo"}.get(analysis["channels"], f"{analysis['channels']} channels")
    first[2].metric("Channels", channel_label)
    second = st.columns(3)
    tempo = analysis["tempo_bpm"]
    second[0].metric("Estimated tempo", f"{tempo:.0f} BPM" if tempo else "Unavailable")
    second[1].metric("Spectral brightness", f"{analysis['spectral_centroid_hz']:.0f} Hz", help="Average spectral centroid: the magnitude-weighted frequency center.")
    second[2].metric("RMS energy", f"{analysis['rms_energy']:.3f}", help="Average short-window signal strength. This is not perceptual loudness in LUFS.")
    third = st.columns(3)
    third[0].metric("Frequency spread", f"{analysis['spectral_bandwidth_hz']:.0f} Hz", help="Spectral bandwidth: how widely frequencies are distributed around the centroid.")
    third[1].metric("High-frequency rolloff", f"{analysis['spectral_rolloff_hz']:.0f} Hz", help="Frequency below which 85% of spectral magnitude is contained, averaged over windows.")
    third[2].metric("Zero-crossing rate", f"{analysis['zero_crossing_rate']:.1%}", help="Fraction of neighboring samples that change sign. Noisy or high-frequency sounds often have higher rates.")
    fourth = st.columns(3)
    peak = analysis["peak_dbfs"]
    fourth[0].metric("Sample peak level", f"{peak:.1f} dBFS" if peak is not None else "Silence", help="Highest absolute sample in the prepared mono excerpt, relative to digital full scale. This is not a true-peak measurement.")
    fourth[1].metric("Energy variation", f"{analysis['energy_variation']:.3f}", help="Standard deviation of short-window RMS energy. Larger values indicate greater changes in signal strength.")
    onset = analysis["onset_strength"]
    fourth[2].metric("Onset strength", f"{onset:.4f}" if onset is not None else "Unavailable", help="Average positive change in short-window RMS energy; a simple measure of attack activity.")
    st.caption(f"Acoustic measurements use a centered {analysis['analysis_duration']:g}-second excerpt, padded with silence if shorter. Tempo can have half-time or double-time ambiguity.")
