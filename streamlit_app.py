from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import pandas as pd
import streamlit as st

from src.seekaudio.analysis import analyze_audio
from src.seekaudio.config import load_config


PROJECT_DIR = Path(__file__).resolve().parent


@st.cache_data
def read_table(path: str, modified: int) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_resource
def read_models(path: str, modified: int) -> dict:
    return joblib.load(path)


def load_catalog(config):
    manifest_path = PROJECT_DIR / config.paths.manifest
    space_dir = PROJECT_DIR / config.paths.space_dir
    vectors_path = space_dir / "vectors.parquet"
    models_path = space_dir / "models.joblib"

    for path in [manifest_path, vectors_path, models_path]:
        if not path.exists():
            st.error("The saved analysis catalog is missing. Include the manifest, vectors, and models in the deployment.")
            st.stop()

    manifest = read_table(str(manifest_path), manifest_path.stat().st_mtime_ns)
    vectors = read_table(str(vectors_path), vectors_path.stat().st_mtime_ns)
    models = read_models(str(models_path), models_path.stat().st_mtime_ns)
    return manifest, vectors, models


def configure_page() -> None:
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
    st.write("Upload a song or try the sample to estimate its genre and explore its audio characteristics.")


def update_upload_state(audio_bytes) -> None:
    signature = None
    if audio_bytes is not None:
        signature = sha256(audio_bytes).hexdigest()
    if st.session_state.get("query_signature") != signature:
        st.session_state.pop("analysis", None)
        st.session_state["query_signature"] = signature


def analyze_upload(uploaded, manifest, vectors, models) -> dict:
    with TemporaryDirectory(prefix="seekaudio-query-") as temporary_dir:
        audio_path = Path(temporary_dir) / Path(uploaded.name).name
        audio_path.write_bytes(uploaded.getvalue())
        return analyze_audio(audio_path, vectors, models, manifest)


def show_spectrogram(analysis: dict) -> None:
    image = analysis.get("mel_spectrogram_png")
    if image is None:
        return
    st.subheader("Sound over time")
    st.image(image)
    duration = analysis["analysis_duration"]
    st.caption(
        f"Mel spectrogram of the analyzed {duration:g}-second excerpt. "
        "Brighter colors show stronger sound at each time and frequency. "
        "Colors are relative to this excerpt's peak, not absolute loudness."
    )
    st.download_button("Download spectrogram", image, file_name="mel_spectrogram.png", mime="image/png")


def show_metric_row(measurements: list) -> None:
    columns = st.columns(len(measurements))
    for index, measurement in enumerate(measurements):
        label, value, explanation = measurement
        columns[index].metric(label, value, help=explanation)


def show_audio_characteristics(analysis: dict) -> None:
    st.subheader("Audio characteristics")
    duration_seconds = round(analysis["duration"])
    duration_label = f"{duration_seconds // 60}:{duration_seconds % 60:02d}"
    channel_names = {1: "Mono", 2: "Stereo"}
    channel_label = channel_names.get(analysis["channels"], f"{analysis['channels']} channels")

    tempo_label = "Unavailable"
    if analysis["tempo_bpm"]:
        tempo_label = f"{analysis['tempo_bpm']:.0f} BPM"
    peak_label = "Silence"
    if analysis["peak_dbfs"] is not None:
        peak_label = f"{analysis['peak_dbfs']:.1f} dBFS"
    onset_label = "Unavailable"
    if analysis["onset_strength"] is not None:
        onset_label = f"{analysis['onset_strength']:.4f}"

    show_metric_row([
        ("File duration", duration_label, "Length of the uploaded recording."),
        ("Original sample rate", f"{analysis['sample_rate'] / 1000:g} kHz", "Samples per second in the original file."),
        ("Channels", channel_label, "Original audio channel layout."),
    ])
    show_metric_row([
        ("Estimated tempo", tempo_label, "Approximate rate of recurring energy increases."),
        ("Spectral brightness", f"{analysis['spectral_centroid_hz']:.0f} Hz", "Average magnitude-weighted frequency center."),
        ("RMS energy", f"{analysis['rms_energy']:.3f}", "Average short-window signal strength, not perceptual loudness in LUFS."),
    ])
    show_metric_row([
        ("Frequency spread", f"{analysis['spectral_bandwidth_hz']:.0f} Hz", "Frequency spread around the spectral centroid."),
        ("High-frequency rolloff", f"{analysis['spectral_rolloff_hz']:.0f} Hz", "Frequency containing 85% of spectral magnitude below it."),
        ("Zero-crossing rate", f"{analysis['zero_crossing_rate']:.1%}", "Fraction of neighboring samples that change sign."),
    ])
    show_metric_row([
        ("Sample peak level", peak_label, "Highest sample in the prepared mono excerpt, relative to full scale. Not true peak."),
        ("Energy variation", f"{analysis['energy_variation']:.3f}", "Standard deviation of short-window RMS energy."),
        ("Onset strength", onset_label, "Average positive change in short-window RMS energy."),
    ])
    duration = analysis["analysis_duration"]
    st.caption(
        f"Acoustic measurements use a centered {duration:g}-second excerpt, "
        "padded with silence if shorter. Tempo can have half-time or double-time ambiguity."
    )


def show_analysis_result(analysis: dict) -> None:
    with st.container(border=True):
        st.metric("Predicted genre", analysis["genre"].title())
        st.caption("Estimated from genre votes among the nearest catalog audio features; not a calibrated classifier.")
    show_spectrogram(analysis)
    show_audio_characteristics(analysis)


def main() -> None:
    configure_page()
    config = load_config(PROJECT_DIR / "config" / "default.yaml")
    manifest, vectors, models = load_catalog(config)
    source = st.radio(
        "Choose your audio",
        ["Upload a song", "Try sample song"],
        horizontal=True,
    )
    uploaded = None
    sample_path = PROJECT_DIR / "data" / "demo" / "sweet_child_o_mine.mp3"
    audio_bytes = None
    if source == "Upload a song":
        uploaded = st.file_uploader(
            "Choose an audio file",
            type=["wav", "flac", "mp3"],
            help="Supported formats: WAV, FLAC, and MP3. We analyze a centered excerpt of your audio.",
        )
        if uploaded is not None:
            audio_bytes = uploaded.getvalue()
    else:
        st.write("Sweet Child O' Mine — Guns N' Roses")
        if sample_path.is_file():
            audio_bytes = sample_path.read_bytes()
        else:
            st.info("The sample song is not available yet. You can still upload your own audio.")

    update_upload_state(audio_bytes)
    if audio_bytes is not None:
        st.caption("Audio preview")
        st.audio(audio_bytes)

    if st.button("Analyze audio", type="primary", disabled=audio_bytes is None):
        try:
            with st.spinner("Analyzing your audio..."):
                if source == "Upload a song":
                    analysis = analyze_upload(uploaded, manifest, vectors, models)
                else:
                    analysis = analyze_audio(sample_path, vectors, models, manifest)
                st.session_state["analysis"] = analysis
        except (OSError, ValueError, RuntimeError) as error:
            st.session_state.pop("analysis", None)
            st.error(str(error))

    if "analysis" in st.session_state:
        show_analysis_result(st.session_state["analysis"])


if __name__ == "__main__":
    main()
