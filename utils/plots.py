"""Figures of spectra and evaluations (matplotlib, headless), shown by the GUI or saved to files."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Colour of each method in the evaluation plots
METHOD_COLORS = {"SpectroVQ": "#d1242f", "TopN Peaks": "#1f3feb"}


def _normalize(intensity):
    intensity = np.asarray(intensity, dtype=float)
    return (
        intensity / intensity.max()
        if intensity.size and intensity.max() > 0
        else intensity
    )


def mirror_plot(
    top_mz, top_it, bottom_mz, bottom_it, top_label, bottom_label, title=""
):
    """Two spectra mirrored about the m/z axis, each scaled to its base peak."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.vlines(
        top_mz, 0, _normalize(top_it), color="#1f6feb", linewidth=0.9, label=top_label
    )
    ax.vlines(
        bottom_mz,
        0,
        -_normalize(bottom_it),
        color="#d1242f",
        linewidth=0.9,
        label=bottom_label,
    )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylim(-1.08, 1.08)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(["100", "50", "0", "50", "100"])
    ax.set_xlabel("m/z")
    ax.set_ylabel("Relative intensity (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    fig.tight_layout()
    plt.close(fig)
    return fig


def similarity_curve(similarities, highlight):
    """Cosine similarity against the number of quantizers (1, 2, ...), circling highlight."""
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    qs = np.arange(1, len(similarities) + 1)
    ax.plot(qs, similarities, marker="o", color="#1f6feb")
    ax.scatter(
        [highlight],
        [similarities[highlight - 1]],
        s=90,
        facecolors="none",
        edgecolors="#d1242f",
        linewidths=2,
        zorder=3,
    )
    ax.set_xlabel("Number of quantizers")
    ax.set_ylabel("Cosine similarity to input")
    ax.set_xticks(qs)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _hist(ax, ratio, sim, color, xmax, ymin, alpha=1.0):
    cmap = LinearSegmentedColormap.from_list("m", ["#ffffff00", color])
    counts, xe, ye = np.histogram2d(
        ratio, sim, bins=[120, 80], range=[[0, xmax], [ymin, 1.0]]
    )
    counts = np.ma.masked_equal(np.log1p(counts), 0)
    ax.pcolormesh(xe, ye, counts.T, cmap=cmap, alpha=alpha, vmin=0, rasterized=True)


def _style(ax, xmax, ymin, ylabel=True):
    ax.set_xlim(0, xmax)
    ax.set_ylim(ymin, 1.0)
    ax.set_xlabel("Compression Ratio (Fold)")
    if ylabel:
        ax.set_ylabel("Spectrum Similarity")
    ax.spines[["top", "right"]].set_visible(False)


def evaluation_plot(rows, layout="Side by side"):
    """2-D histograms of compression ratio vs similarity per method, from utils.evaluate rows.

    layout "Overlay" draws both methods in one plot; anything else side by side.
    """
    data = {
        m: np.array([(r[4], r[5]) for r in rows if r[1] == m]).reshape(-1, 2)
        for m in METHOD_COLORS
    }
    allr = np.concatenate([d[:, 0] for d in data.values()])
    alls = np.concatenate([d[:, 1] for d in data.values()])
    # The 99.5th percentile keeps a few huge spectra from squashing the plot.
    xmax = float(np.percentile(allr, 99.5)) * 1.05 if allr.size else 1.0
    ymin = max(0.0, np.floor(float(alls.min()) * 10) / 10) if alls.size else 0.0
    if layout == "Overlay":
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for m, color in METHOD_COLORS.items():
            _hist(ax, *data[m].T, color, xmax, ymin, alpha=0.6)
        _style(ax, xmax, ymin)
        ax.legend(
            handles=[
                plt.Rectangle((0, 0), 1, 1, color=c) for c in METHOD_COLORS.values()
            ],
            labels=list(METHOD_COLORS),
            title="Method",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
        for i, (ax, (m, color)) in enumerate(zip(axes, METHOD_COLORS.items())):
            _hist(ax, *data[m].T, color, xmax, ymin)
            _style(ax, xmax, ymin, ylabel=i == 0)
            ax.set_title(m)
    fig.tight_layout()
    plt.close(fig)
    return fig
