"""Render offline evaluation charts and the live analysis Mel spectrogram."""

from pathlib import Path
from io import BytesIO

import matplotlib
matplotlib.use("Agg")  # Headless rendering works on cloud servers without a desktop.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.figure import Figure

from .features.spectrogram import compute_mel_spectrogram


def plot_evaluation(results: pd.DataFrame, comparison: pd.DataFrame, output_dir: str | Path, k: int) -> None:
    """Save by_genre.png and ablations.png with Precision@K axes from zero to one.

    The genre chart includes expected random retrieval as a reference line.
    Figures are closed after saving so repeated runs do not retain GUI resources.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_genre = results.groupby("genre")["precision_at_k"].mean().sort_values()
    figure, axis = plt.subplots(figsize=(9, 5))
    by_genre.plot.barh(ax=axis, color="#2976a3")
    axis.axvline(results.random_precision_at_k.mean(), color="#d66a32", linestyle="--", label="Expected random")
    axis.set(xlim=(0, 1), xlabel=f"Mean Precision@{k}", title="Retrieval quality by genre")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "by_genre.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(9, 5))
    comparison.set_index("experiment")["precision_at_k"].sort_values().plot.barh(ax=axis, color="#2976a3")
    axis.set(xlim=(0, 1), xlabel=f"Mean Precision@{k}", title="Feature ablation comparison")
    figure.tight_layout()
    figure.savefig(output_dir / "ablations.png", dpi=160)
    plt.close(figure)


def render_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
) -> bytes:
    """Return a PNG showing the prepared excerpt's log-Mel power over time.

    Input audio is already mono/resampled/cropped. Frequency coordinates use the
    HTK Mel scale but tick labels are in Hz. Time is relative to this excerpt,
    not the position in the original full recording. Colors span -80 to 0 dB
    relative to the excerpt peak. A byte image is easy to store in UI session state.
    """
    decibels = compute_mel_spectrogram(waveform, sample_rate, n_fft, hop_length, n_mels)
    highest_mel = 2_595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_spacing = highest_mel / (n_mels + 1)
    # Place pixel edges half a hop/band away from each frame/filter center.
    left_time = (n_fft / 2 - hop_length / 2) / sample_rate
    right_time = (n_fft / 2 + (decibels.shape[1] - 0.5) * hop_length) / sample_rate
    # A standalone Figure avoids relying on pyplot's shared current-figure state.
    figure = Figure(figsize=(9, 3.8), facecolor="#0e1117", layout="constrained")
    axis = figure.subplots()
    axis.set_facecolor("#0e1117")
    image = axis.imshow(
        decibels, origin="lower", aspect="auto", cmap="magma",
        vmin=-80, vmax=0, interpolation="nearest",
        extent=[left_time, right_time, mel_spacing / 2, highest_mel - mel_spacing / 2],
    )
    frequency_ticks = np.array([0, 250, 500, 1_000, 2_000, 4_000, 8_000])
    frequency_ticks = frequency_ticks[frequency_ticks <= sample_rate / 2]
    # Nyquist (half the sample rate) is the highest representable frequency.
    axis.set_yticks(2_595 * np.log10(1 + frequency_ticks / 700), frequency_ticks.astype(str))
    axis.set(xlim=(0, len(waveform) / sample_rate), ylim=(0, highest_mel),
             xlabel="Time in analyzed excerpt (seconds)", ylabel="Frequency (Hz, Mel scale)")
    axis.tick_params(colors="#e4e7eb")
    axis.xaxis.label.set_color("#e4e7eb")
    axis.yaxis.label.set_color("#e4e7eb")
    colorbar = figure.colorbar(image, ax=axis, label="dB relative to excerpt peak")
    colorbar.ax.tick_params(colors="#e4e7eb")
    colorbar.ax.yaxis.label.set_color("#e4e7eb")
    output = BytesIO()
    # Render in memory: no disk image path or raw waveform must survive UI reruns.
    figure.savefig(output, format="png", dpi=150)
    return output.getvalue()
