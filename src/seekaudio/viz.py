from pathlib import Path
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.figure import Figure

from .features.spectrogram import compute_mel_spectrogram


def plot_evaluation(results: pd.DataFrame, comparison: pd.DataFrame, output_dir: str | Path, k: int) -> None:
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
    decibels = compute_mel_spectrogram(waveform, sample_rate, n_fft, hop_length, n_mels)
    highest_mel = 2_595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_spacing = highest_mel / (n_mels + 1)
    left_time = (n_fft / 2 - hop_length / 2) / sample_rate
    right_time = (n_fft / 2 + (decibels.shape[1] - 0.5) * hop_length) / sample_rate
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
    figure.savefig(output, format="png", dpi=150)
    return output.getvalue()
