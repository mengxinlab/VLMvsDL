"""Shared figure styling for all VLMvsDL manuscript figures.

The manuscript figures use a restrained, journal-neutral style with LMRoman10
as the single preferred font, black text, no decorative backgrounds, and
color-blind-safe hues. This module keeps that as the single source of truth so
model colors, line styles, and labels stay consistent across every figure.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.axes import Axes

# ── Journal-compatible accessible palette (Okabe-Ito / Wong + Tol extras) ──
OKABE_ITO = {
    "black":         "#000000",
    "orange":        "#E69F00",
    "sky_blue":      "#56B4E9",
    "bluish_green":  "#009E73",
    "yellow":        "#F0E442",
    "blue":          "#0072B2",
    "vermillion":    "#D55E00",
    "reddish_purple": "#CC79A7",
}
# Two extra color-blind-safe hues (Paul Tol) so all 7 DL baselines and the
# Gemini generations remain mutually distinguishable on line plots.
_TOL_TEAL = "#44AA99"
_TOL_INDIGO = "#332288"
_TOL_WINE = "#882255"
_GREY = "#8C8C8C"
_GOLD = "#C49A00"
_SLATE = "#4D4D4D"
_PANEL_GREY = "#D9D9D9"
_AXIS_GREY = "#333333"
_SOFT_LINE = "#B8B8B8"

# ── Per-model colors (fixed everywhere) ───────────────────────────────────
# Family 1 — supervised DL baselines: cool hues (blues / greens / indigo).
# Family 2 — general-purpose Gemini VLMs: warm hues (orange / vermillion /
#            reddish-purple / brown).
# Family 3 — specialty medical VLM (MedGemma): wine, deliberately off both.
MODEL_COLORS: dict[str, str] = {
    # --- supervised DL (cool) ---
    "STU-Net":          OKABE_ITO["blue"],
    "EfficientNet-B0":  OKABE_ITO["bluish_green"],
    "EffNet-B0":        OKABE_ITO["bluish_green"],
    "ResNet-18":        OKABE_ITO["sky_blue"],
    "ResNet-50":        _TOL_TEAL,
    "DenseNet-121":     _TOL_INDIGO,
    "DenseNet":         _TOL_INDIGO,
    "Swin-UNETR":       "#117733",
    "Swin":             "#117733",
    "ViT-Base":         "#6699CC",
    # --- general-purpose Gemini VLMs (warm) ---
    "Gemini 3 Flash Preview (F3)":  OKABE_ITO["vermillion"],
    "Gemini 3 Flash Preview":       OKABE_ITO["vermillion"],
    "Gemini 3 Flash (F3)":          OKABE_ITO["vermillion"],
    "Gemini F3":                    OKABE_ITO["vermillion"],
    "Gemini 3 Flash Preview zero-shot rich": OKABE_ITO["orange"],
    "Gemini 3 Flash Preview 20-shot rich": OKABE_ITO["reddish_purple"],
    "Gemini 2.5 Flash":             OKABE_ITO["orange"],
    "Gemini 2.5 Pro":               OKABE_ITO["reddish_purple"],
    "Gemini 3.1 Pro Preview":       "#A6761D",
    # --- ablation conditions (Gemini 3 Flash Preview) ---
    "Z1 zero-shot minimal":         _GREY,
    "Z2 zero-shot rich":            OKABE_ITO["orange"],
    "Z3 zero-shot rich+clinical":   _GOLD,
    "F1 20-shot minimal":           "#A6761D",
    "F2 20-shot rich":              OKABE_ITO["reddish_purple"],
    "F3 20-shot rich+clinical":     OKABE_ITO["vermillion"],
    "F3 20-shot+clinical":          OKABE_ITO["vermillion"],
    "F3A 20-shot rich+clinical (run-averaged)": OKABE_ITO["vermillion"],
    "F3A 20-shot+clinical (run-averaged)": OKABE_ITO["vermillion"],
    # --- specialty medical VLM ---
    "MedGemma 1.5-4B":  _TOL_WINE,
    "MedGemma":         _TOL_WINE,
    "20-shot Med3D ResNet18+TTT": _SLATE,
}

# Linestyles let line plots stay separable even in greyscale / for the few
# models that share a hue family.
MODEL_LINESTYLES: dict[str, str] = {
    "STU-Net": "-", "EfficientNet-B0": "-", "EffNet-B0": "-",
    "ResNet-18": "--", "ResNet-50": "-.", "DenseNet-121": ":",
    "Swin-UNETR": (0, (3, 1, 1, 1)), "ViT-Base": (0, (5, 2)),
    "Gemini 3 Flash Preview (F3)": "-",
    "Gemini 3 Flash Preview": "-",
    "Gemini 3 Flash Preview zero-shot rich": "-.",
    "Gemini 3 Flash Preview 20-shot rich": ":",
    "Gemini 2.5 Flash": "--", "Gemini 2.5 Pro": "-.",
    "Gemini 3.1 Pro Preview": ":", "MedGemma 1.5-4B": (0, (3, 1, 1, 1)),
    "Z1 zero-shot minimal": "--", "Z2 zero-shot rich": "-.",
    "Z3 zero-shot rich+clinical": (0, (5, 2)), "F1 20-shot minimal": (0, (3, 1, 1, 1)),
    "F3 20-shot rich+clinical": "-", "F3 20-shot+clinical": "-",
    "F2 20-shot rich": ":",
    "F3A 20-shot rich+clinical (run-averaged)": "-",
    "F3A 20-shot+clinical (run-averaged)": "-",
    "20-shot Med3D ResNet18+TTT": (0, (2, 1)),
}

MODEL_MARKERS: dict[str, str] = {
    "STU-Net": "o",
    "EfficientNet-B0": "s",
    "EffNet-B0": "s",
    "ResNet-18": "^",
    "ResNet-50": "D",
    "DenseNet-121": "v",
    "DenseNet": "v",
    "Swin-UNETR": "P",
    "Swin": "P",
    "ViT-Base": "X",
    "Gemini 3 Flash Preview (F3)": "o",
    "Gemini 3 Flash Preview": "o",
    "Gemini 3 Flash (F3)": "o",
    "Gemini 3 Flash Preview zero-shot rich": "s",
    "Gemini 3 Flash Preview 20-shot rich": "v",
    "Gemini 2.5 Flash": "s",
    "Gemini 2.5 Pro": "^",
    "Gemini 3.1 Pro Preview": "D",
    "Z1 zero-shot minimal": "o",
    "Z2 zero-shot rich": "s",
    "Z3 zero-shot rich+clinical": "^",
    "F1 20-shot minimal": "D",
    "F2 20-shot rich": "v",
    "F3 20-shot rich+clinical": "o",
    "F3 20-shot+clinical": "o",
    "F3A 20-shot rich+clinical (run-averaged)": "o",
    "F3A 20-shot+clinical (run-averaged)": "o",
    "MedGemma 1.5-4B": "X",
    "MedGemma": "X",
    "20-shot Med3D ResNet18+TTT": "s",
}

DISPLAY_NAMES: dict[str, str] = {
    "Gemini 3 Flash Preview (F3)": "Gemini F3",
    "Gemini 3 Flash Preview": "Gemini 3 Flash",
    "Gemini 3 Flash (F3)": "Gemini F3",
    "Gemini 3 Flash Preview zero-shot rich": "Gemini Z2",
    "Gemini 3 Flash Preview 20-shot rich": "Gemini F2",
    "Gemini 3.1 Pro Preview": "Gemini 3.1 Pro",
    "Gemini 2.5 Flash": "Gemini 2.5 Flash",
    "Gemini 2.5 Pro": "Gemini 2.5 Pro",
    "Gemini 3 Flash (Z1)": "Gemini Z1",
    "Gemini 3 Flash (F2)": "Gemini F2",
    "Z1 zero-shot minimal": "Z1",
    "Z2 zero-shot rich": "Z2",
    "Z3 zero-shot rich+clinical": "Z3",
    "F1 20-shot minimal": "F1",
    "F2 20-shot rich": "F2",
    "F3 20-shot rich+clinical": "F3",
    "F3 20-shot+clinical": "F3",
    "F3A 20-shot rich+clinical (run-averaged)": "F3A",
    "F3A 20-shot+clinical (run-averaged)": "F3A",
    "EfficientNet-B0": "EffNet-B0",
    "MedGemma 1.5-4B": "MedGemma",
    "20-shot Med3D ResNet18+TTT": "20-shot Med3D+TTT",
}

FAMILY_COLORS = {
    "DL": OKABE_ITO["blue"],
    "VLM": OKABE_ITO["vermillion"],
    "MedVLM": _TOL_WINE,
}

# LUNA25 malignant prevalence (8.8 %) — PR-curve / decision-curve reference.
LUNA25_PREVALENCE = 81 / 917


def color_for(name: str, default: str = _GREY) -> str:
    """Look up a model color, tolerant of newline-wrapped tick labels."""
    key = str(name).replace("\n", " ").strip()
    if key in MODEL_COLORS:
        return MODEL_COLORS[key]
    for k, v in MODEL_COLORS.items():
        if key.startswith(k) or k in key:
            return v
    return default


def linestyle_for(name: str, default: str = "-") -> str:
    key = str(name).replace("\n", " ").strip()
    if key in MODEL_LINESTYLES:
        return MODEL_LINESTYLES[key]
    for k, v in MODEL_LINESTYLES.items():
        if key.startswith(k) or k in key:
            return v
    return default


def marker_for(name: str, default: str = "o") -> str:
    key = str(name).replace("\n", " ").strip()
    if key in MODEL_MARKERS:
        return MODEL_MARKERS[key]
    for k, v in MODEL_MARKERS.items():
        if key.startswith(k) or k in key:
            return v
    return default


def display_name(name: str) -> str:
    key = str(name).replace("\n", " ").strip()
    if key in DISPLAY_NAMES:
        return DISPLAY_NAMES[key]
    for k, v in DISPLAY_NAMES.items():
        if key.startswith(k) or k in key:
            return v
    return key


def add_panel_label(ax: Axes, label: str) -> None:
    """Panel label: bold lowercase letter, black, top-left."""
    ax.text(
        -0.16,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=_AXIS_GREY,
    )


def apply_style() -> None:
    """Journal-friendly vector defaults. Call before plotting."""
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "serif",
        "font.serif": [
            "LMRoman10",
            "Latin Modern Roman",
            "CMU Serif",
            "Computer Modern Roman",
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "custom",
        "mathtext.rm": "LMRoman10",
        "mathtext.it": "LMRoman10:italic",
        "mathtext.bf": "LMRoman10:bold",
        "mathtext.fallback": "cm",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": _AXIS_GREY,
        "axes.linewidth": 0.8,
        "axes.labelcolor": _AXIS_GREY,
        "axes.titlepad": 4,
        "xtick.color": _AXIS_GREY,
        "ytick.color": _AXIS_GREY,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
        "legend.title_fontsize": 7.5,
        "text.color": _AXIS_GREY,
    })


def rounded_3(x: float) -> str:
    """Round-half-up to 3 decimals, the single convention used everywhere
    (numbers in text/tables/figures must all come from this, never hand-typed
    and never via Python banker's rounding)."""
    return f"{(int(float(x) * 1000 + 0.5) / 1000):.3f}"
