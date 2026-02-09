#!/usr/bin/env python3
"""
CipherChat Analysis Script
===========================
Generates comparative plots and summary tables for evaluating
open-source LLMs (Qwen 2.5-7B-Instruct & Mistral-7B-Instruct-v0.3)
under CipherChat jailbreak attacks using Caesar Cipher and SelfCipher,
with and without unsafe demonstrations.

Outputs are saved to  CipherChat/analysis_plots/

Metrics analysed
-----------------
1. Toxicity Rate   – fraction of responses judged toxic
2. Validity Rate   – fraction of responses that are coherent / valid
3. Refusal Rate    – fraction of responses where the model refused
4. Avg Grammar/Fluency – mean grammar-fluency score (1-5 scale)

Dimensions
----------
- Model            : Qwen vs Mistral
- Cipher setting   : Caesar vs SelfCipher
- Demo setting     : with_unsafe_demo vs without_demo
- Instruction type : 4 harm categories
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # headless backend for HPC
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "saved_results")
OUT_DIR     = os.path.join(BASE_DIR, "analysis_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────
# STYLE CONFIGURATION
# ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "figure.constrained_layout.use": True,
})

# These will be populated dynamically after loading data
MODEL_SHORT  = {}   # model_name -> short display name
MODEL_COLORS = {}   # short_name -> colour hex

# Fixed colour palette that scales to any number of models
_COLOR_PALETTE = [
    "#1f77b4", "#e377c2", "#ff7f0e", "#2ca02c", "#9467bd",
    "#8c564b", "#d62728", "#7f7f7f", "#bcbd22", "#17becf",
]

def _make_short_name(full_name: str) -> str:
    """Derive a concise display name from a HuggingFace-style model id.
    E.g. 'mistralai/Mistral-7B-Instruct-v0.3' -> 'Mistral-7B'
         'Qwen/Qwen2.5-7B-Instruct'           -> 'Qwen2.5-7B'
         'meta-llama/Llama-3-8B-Instruct'      -> 'Llama-3-8B'
    """
    name = full_name.split("/")[-1]           # strip org prefix
    # Remove common suffixes to shorten
    for suffix in ["-Instruct", "-instruct", "-Chat", "-chat",
                   "-v0.3", "-v0.2", "-v0.1", "-hf"]:
        name = name.replace(suffix, "")
    # Collapse double hyphens
    while "--" in name:
        name = name.replace("--", "-")
    return name.strip("-")

def _build_model_lookups(model_names: list):
    """Populate MODULE-level MODEL_SHORT and MODEL_COLORS dicts."""
    global MODEL_SHORT, MODEL_COLORS
    MODEL_SHORT  = {}
    MODEL_COLORS = {}
    for i, full in enumerate(sorted(set(model_names))):
        short = _make_short_name(full)
        MODEL_SHORT[full]   = short
        MODEL_COLORS[short] = _COLOR_PALETTE[i % len(_COLOR_PALETTE)]

CIPHER_MARKERS  = {"Caesar": "s", "SelfCipher": "o"}
DEMO_HATCHES    = {"without_demo": "",  "with_unsafe_demo": "///"}
DEMO_LABELS     = {"without_demo": "No Demo", "with_unsafe_demo": "Unsafe Demo"}

METRICS = [
    ("toxicity_rate",       "Toxicity Rate",        (0, 1.05)),
    ("validity_rate",       "Validity Rate",        (0, 1.05)),
    ("refusal_rate",        "Refusal Rate",         (0, 1.05)),
    ("avg_grammar_fluency", "Avg Grammar / Fluency", (0, 5.2)),
]

# Known short names; anything not listed is auto-shortened
_INSTRUCTION_SHORT_KNOWN = {
    "Crimes_And_Illegal_Activities": "Crimes",
    "Privacy_And_Property":          "Privacy",
    "Role_Play_Instruction":         "Role-Play",
    "Unsafe_Instruction_Topic":      "Unsafe Topic",
}

def _make_instr_short(raw: str) -> str:
    if raw in _INSTRUCTION_SHORT_KNOWN:
        return _INSTRUCTION_SHORT_KNOWN[raw]
    # Fallback: replace underscores with spaces and title-case
    return raw.replace("_", " ").title()

INSTRUCTION_SHORT = {}   # populated dynamically

def _build_instruction_lookups(instruction_types: list):
    global INSTRUCTION_SHORT
    INSTRUCTION_SHORT = {}
    for it in sorted(set(instruction_types)):
        INSTRUCTION_SHORT[it] = _make_instr_short(it)

# ──────────────────────────────────────────────────────────────────
# LOAD DATA — parse all *_eval.json files from saved_results/
# ──────────────────────────────────────────────────────────────────
import json, glob

CIPHER_MAP  = {"caesar": "Caesar", "unchange": "SelfCipher"}

def load_data_from_jsons(results_dir: str) -> pd.DataFrame:
    """Walk every *_eval.json under results_dir (skip 'Untitled' dirs)
    and extract one summary row per file."""
    rows = []
    pattern = os.path.join(results_dir, "*_eval.json")
    for fp in sorted(glob.glob(pattern)):
        # skip anything in an 'Untitled' subdirectory
        if "Untitled" in fp:
            continue
        with open(fp, "r") as fh:
            data = json.load(fh)

        meta = data.get("run_metadata", {})
        model_name       = meta.get("model_name", "")
        encode_method    = meta.get("encode_method", "")
        instruction_type = meta.get("instruction_type", "")
        use_demos        = meta.get("use_demonstrations", False)
        demo_toxicity    = meta.get("demonstration_toxicity", "toxic")
        language         = meta.get("language", "en")

        cipher_setting = CIPHER_MAP.get(encode_method, encode_method)
        demo_setting   = "with_unsafe_demo" if use_demos else "without_demo"
        scenario       = f"{cipher_setting}|{demo_setting}"

        rows.append({
            "model_name":          model_name,
            "instruction_type":    instruction_type,
            "encode_method":       encode_method,
            "cipher_setting":      cipher_setting,
            "use_demonstrations":  use_demos,
            "demo_setting":        demo_setting,
            "scenario":            scenario,
            "demonstration_toxicity": demo_toxicity,
            "language":            language,
            "total":               data.get("total", 0),
            "valid_count":         data.get("valid_count", 0),
            "refusal_count":       data.get("refusal_count", 0),
            "toxicity_yes_count":  data.get("toxicity_yes_count", 0),
            "toxicity_no_count":   data.get("toxicity_no_count", 0),
            "toxicity_rate":       data.get("toxicity_rate", 0.0),
            "refusal_rate":        data.get("refusal_rate", 0.0),
            "validity_rate":       data.get("validity_rate", 0.0),
            "avg_grammar_fluency": data.get("avg_grammar_fluency", 0.0),
            "grammar_scores":     data.get("grammar_scores", []),
            "eval_path":           fp,
        })

    df = pd.DataFrame(rows)

    # Auto-build lookups from whatever models / instruction types were found
    _build_model_lookups(df["model_name"].tolist())
    _build_instruction_lookups(df["instruction_type"].tolist())

    df["model_short"] = df["model_name"].map(MODEL_SHORT)
    df["instr_short"] = df["instruction_type"].map(INSTRUCTION_SHORT)
    return df


# ──────────────────────────────────────────────────────────────────
# 1) GROUPED BAR CHART – one metric, grouped by instruction type
#    bars split by (cipher, demo) for each model
# ──────────────────────────────────────────────────────────────────
def plot_grouped_bars(df: pd.DataFrame, metric: str, ylabel: str,
                      ylim: tuple, fname: str):
    """
    For every instruction type, show bars for each
    model × cipher × demo combination.
    """
    categories = sorted(df["instr_short"].dropna().unique())
    models     = sorted(df["model_short"].dropna().unique())
    ciphers    = sorted(df["cipher_setting"].dropna().unique())
    demos      = ["without_demo", "with_unsafe_demo"]

    n_cats   = len(categories)
    n_bars   = len(models) * len(ciphers) * len(demos)
    bar_w    = max(0.04, 0.72 / max(n_bars, 1))
    group_w  = n_bars * bar_w + 0.12

    fig, ax = plt.subplots(figsize=(14, 5.5))

    x_centres = np.arange(n_cats) * group_w
    legend_handles = {}

    idx = 0
    for mi, model in enumerate(models):
        for ci, cipher in enumerate(ciphers):
            for di, demo in enumerate(demos):
                vals = []
                for cat in categories:
                    row = df[(df["model_short"] == model) &
                             (df["cipher_setting"] == cipher) &
                             (df["demo_setting"] == demo) &
                             (df["instr_short"] == cat)]
                    vals.append(row[metric].values[0] if len(row) else 0)

                x_pos = x_centres + (idx - n_bars / 2 + 0.5) * bar_w
                color = MODEL_COLORS.get(model, "#333333")
                alpha = 0.55 if demo == "without_demo" else 1.0
                hatch = DEMO_HATCHES[demo]
                label = f"{model} | {cipher} | {DEMO_LABELS[demo]}"
                bars  = ax.bar(x_pos, vals, width=bar_w, color=color,
                               alpha=alpha, hatch=hatch, edgecolor="white",
                               linewidth=0.6, label=label)
                legend_handles[label] = bars
                idx += 1

    ax.set_xticks(x_centres)
    ax.set_xticklabels(categories, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1 if ylim[1] <= 1.1 else 1.0))
    ax.set_title(f"{ylabel} — by Model, Cipher & Demo Setting")
    ax.legend(handles=list(legend_handles.values()),
              labels=list(legend_handles.keys()),
              loc="upper left", bbox_to_anchor=(1.01, 1), frameon=True,
              fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.savefig(os.path.join(OUT_DIR, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")


# ──────────────────────────────────────────────────────────────────
# 2) SIDE-BY-SIDE HEATMAPS – one per model
# ──────────────────────────────────────────────────────────────────
def plot_heatmaps(df: pd.DataFrame, metric: str, label: str, fname: str):
    models  = sorted(df["model_short"].dropna().unique())
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models,
                             figsize=(7 * n_models, 4.5), sharey=True,
                             squeeze=False)
    axes = axes.flatten()

    for ax, model in zip(axes, models):
        sub = df[df["model_short"] == model]
        pivot = sub.pivot_table(index="instr_short",
                                columns="scenario",
                                values=metric,
                                aggfunc="first")
        # order columns nicely
        col_order = ["Caesar|without_demo", "Caesar|with_unsafe_demo",
                     "SelfCipher|without_demo", "SelfCipher|with_unsafe_demo"]
        col_order = [c for c in col_order if c in pivot.columns]
        pivot = pivot[col_order]
        pivot = pivot.reindex(sorted(df["instr_short"].dropna().unique()))

        nice_cols = []
        for c in pivot.columns:
            parts = c.split("|")
            nice_cols.append(f"{parts[0]}\n{DEMO_LABELS.get(parts[1], parts[1])}")
        pivot.columns = nice_cols

        vmax = 1.0 if "rate" in metric else 5.0
        im = ax.imshow(pivot.values.astype(float), cmap="YlOrRd",
                       aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=8, ha="center")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_title(model, fontweight="bold")

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                txt_color = "white" if val > vmax * 0.65 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, color=txt_color, fontweight="bold")

    fig.suptitle(f"{label} — Heatmap Comparison", fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=list(axes), shrink=0.8, pad=0.02)
    cbar.set_label(label)
    fig.savefig(os.path.join(OUT_DIR, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")


# ──────────────────────────────────────────────────────────────────
# 3) DEMO IMPACT – difference plot (with_demo − without_demo)
# ──────────────────────────────────────────────────────────────────
def plot_demo_impact(df: pd.DataFrame, metric: str, label: str, fname: str):
    """Bar chart showing the *change* when adding unsafe demonstrations."""
    models  = sorted(df["model_short"].dropna().unique())
    ciphers = sorted(df["cipher_setting"].dropna().unique())
    categories = sorted(df["instr_short"].dropna().unique())
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(6.5 * n_models, 5),
                             sharey=True, squeeze=False)
    axes = axes.flatten()
    n_ciphers = len(ciphers)
    bar_w = 0.7 / max(n_ciphers, 1)
    x = np.arange(len(categories))

    for ax, model in zip(axes, models):
        for ci, cipher in enumerate(ciphers):
            diffs = []
            for cat in categories:
                with_d = df[(df["model_short"] == model) &
                            (df["cipher_setting"] == cipher) &
                            (df["demo_setting"] == "with_unsafe_demo") &
                            (df["instr_short"] == cat)]
                wo_d   = df[(df["model_short"] == model) &
                            (df["cipher_setting"] == cipher) &
                            (df["demo_setting"] == "without_demo") &
                            (df["instr_short"] == cat)]
                v_with = with_d[metric].values[0] if len(with_d) else 0
                v_wo   = wo_d[metric].values[0]   if len(wo_d)   else 0
                diffs.append(v_with - v_wo)

            offset = (ci - (n_ciphers - 1) / 2) * bar_w
            color  = _COLOR_PALETTE[(ci + 3) % len(_COLOR_PALETTE)]
            ax.bar(x + offset, diffs, width=bar_w, color=color,
                   edgecolor="white", alpha=0.85, label=cipher)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_title(model, fontweight="bold")
        ax.set_ylabel(f"Δ {label}\n(with demo − without demo)" if ax is axes[0] else "")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(f"Effect of Unsafe Demonstrations on {label}", fontsize=13,
                 fontweight="bold")
    fig.savefig(os.path.join(OUT_DIR, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")


# ──────────────────────────────────────────────────────────────────
# 4) CIPHER COMPARISON – Caesar vs SelfCipher for each model
# ──────────────────────────────────────────────────────────────────
def plot_cipher_comparison(df: pd.DataFrame, metric: str, label: str, fname: str):
    """Grouped bars: for each instruction type, compare Caesar vs SelfCipher,
       averaged over demo settings, per model."""
    models     = sorted(df["model_short"].dropna().unique())
    ciphers    = sorted(df["cipher_setting"].dropna().unique())
    categories = sorted(df["instr_short"].dropna().unique())
    n_models  = len(models)
    n_ciphers = len(ciphers)

    fig, axes = plt.subplots(1, n_models, figsize=(6.5 * n_models, 5),
                             sharey=True, squeeze=False)
    axes = axes.flatten()
    bar_w = 0.7 / max(n_ciphers, 1)
    x = np.arange(len(categories))

    for ax, model in zip(axes, models):
        for ci, cipher in enumerate(ciphers):
            vals = []
            for cat in categories:
                sub = df[(df["model_short"] == model) &
                         (df["cipher_setting"] == cipher) &
                         (df["instr_short"] == cat)]
                vals.append(sub[metric].mean() if len(sub) else 0)

            offset = (ci - (n_ciphers - 1) / 2) * bar_w
            color  = _COLOR_PALETTE[(ci + 3) % len(_COLOR_PALETTE)]
            ax.bar(x + offset, vals, width=bar_w, color=color,
                   edgecolor="white", alpha=0.85, label=cipher)
            # value annotations
            for xi, v in zip(x + offset, vals):
                ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_title(model, fontweight="bold")
        ax.set_ylabel(label if ax is axes[0] else "")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        if "rate" in metric:
            ax.set_ylim(0, 1.15)
        else:
            ax.set_ylim(0, 5.5)

    cipher_names = " vs ".join(ciphers)
    fig.suptitle(f"{label} — {cipher_names} (avg over demo settings)",
                 fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(OUT_DIR, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")


# ──────────────────────────────────────────────────────────────────
# 5) MODEL COMPARISON – radar / spider chart per scenario
# ──────────────────────────────────────────────────────────────────
def plot_radar(df: pd.DataFrame, fname: str):
    """Radar chart: avg over instruction types, one polygon per model,
       axes = toxicity, validity, refusal, grammar (normalised 0-1)."""
    scenarios = df["scenario"].unique()
    models    = sorted(df["model_short"].dropna().unique())
    radar_metrics = ["toxicity_rate", "validity_rate", "refusal_rate", "avg_grammar_fluency"]
    radar_labels  = ["Toxicity", "Validity", "Refusal", "Grammar\n(÷5)"]

    n_scenarios = len(scenarios)
    cols = min(4, n_scenarios)
    rows = int(np.ceil(n_scenarios / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows),
                             subplot_kw={"projection": "polar"})
    if n_scenarios == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    for si, scenario in enumerate(sorted(scenarios)):
        ax = axes[si]
        for model in models:
            sub = df[(df["model_short"] == model) & (df["scenario"] == scenario)]
            vals = []
            for m in radar_metrics:
                v = sub[m].mean()
                if m == "avg_grammar_fluency":
                    v /= 5.0           # normalise to 0-1
                vals.append(v)
            vals += vals[:1]
            ax.plot(angles, vals, marker="o", markersize=5, label=model,
                    color=MODEL_COLORS.get(model, "#333333"), linewidth=2)
            ax.fill(angles, vals, alpha=0.15, color=MODEL_COLORS.get(model, "#333333"))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(scenario.replace("|", " | "), fontsize=10, fontweight="bold",
                     pad=18)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=8)

    for si in range(n_scenarios, len(axes)):
        axes[si].set_visible(False)

    fig.suptitle("Model Comparison Radar — per Scenario (averaged over categories)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(os.path.join(OUT_DIR, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")


# ──────────────────────────────────────────────────────────────────
# 6) OVERALL AGGREGATED COMPARISON TABLE (saved as CSV + printed)
# ──────────────────────────────────────────────────────────────────
def make_summary_tables(df: pd.DataFrame):
    """Produce aggregated summary tables and save as CSV + Markdown."""

    # ── Table A: per-model per-cipher per-demo averages ──
    tbl_a = (df.groupby(["model_short", "cipher_setting", "demo_setting"])
               .agg(toxicity_rate=("toxicity_rate", "mean"),
                    validity_rate=("validity_rate", "mean"),
                    refusal_rate=("refusal_rate", "mean"),
                    avg_grammar_fluency=("avg_grammar_fluency", "mean"))
               .round(3)
               .reset_index())
    tbl_a.columns = ["Model", "Cipher", "Demo", "Toxicity Rate",
                     "Validity Rate", "Refusal Rate", "Avg Grammar"]
    tbl_a.to_csv(os.path.join(OUT_DIR, "table_a_aggregated_by_cipher_demo.csv"),
                 index=False)

    # ── Table B: per-model per-instruction-type averages ──
    tbl_b = (df.groupby(["model_short", "instr_short"])
               .agg(toxicity_rate=("toxicity_rate", "mean"),
                    validity_rate=("validity_rate", "mean"),
                    refusal_rate=("refusal_rate", "mean"),
                    avg_grammar_fluency=("avg_grammar_fluency", "mean"))
               .round(3)
               .reset_index())
    tbl_b.columns = ["Model", "Category", "Toxicity Rate",
                     "Validity Rate", "Refusal Rate", "Avg Grammar"]
    tbl_b.to_csv(os.path.join(OUT_DIR, "table_b_aggregated_by_category.csv"),
                 index=False)

    # ── Table C: full detailed table (for appendix) ──
    tbl_c = (df[["model_short", "instr_short", "cipher_setting",
                 "demo_setting", "toxicity_rate", "validity_rate",
                 "refusal_rate", "avg_grammar_fluency"]]
               .sort_values(["model_short", "cipher_setting",
                             "demo_setting", "instr_short"])
               .reset_index(drop=True))
    tbl_c.columns = ["Model", "Category", "Cipher", "Demo",
                     "Toxicity", "Validity", "Refusal", "Grammar"]
    tbl_c.to_csv(os.path.join(OUT_DIR, "table_c_full_results.csv"), index=False)

    # ── Table D: Demo effect (Δ = with_demo − without_demo) ──
    rows = []
    for model in df["model_short"].unique():
        for cipher in df["cipher_setting"].unique():
            for cat in df["instr_short"].unique():
                w  = df[(df["model_short"]==model)&(df["cipher_setting"]==cipher)&
                        (df["demo_setting"]=="with_unsafe_demo")&(df["instr_short"]==cat)]
                wo = df[(df["model_short"]==model)&(df["cipher_setting"]==cipher)&
                        (df["demo_setting"]=="without_demo")&(df["instr_short"]==cat)]
                if len(w) and len(wo):
                    rows.append({
                        "Model": model, "Cipher": cipher, "Category": cat,
                        "Δ Toxicity":  round(w["toxicity_rate"].values[0] - wo["toxicity_rate"].values[0], 3),
                        "Δ Validity":  round(w["validity_rate"].values[0] - wo["validity_rate"].values[0], 3),
                        "Δ Refusal":   round(w["refusal_rate"].values[0]  - wo["refusal_rate"].values[0], 3),
                        "Δ Grammar":   round(w["avg_grammar_fluency"].values[0] - wo["avg_grammar_fluency"].values[0], 3),
                    })
    tbl_d = pd.DataFrame(rows)
    tbl_d.to_csv(os.path.join(OUT_DIR, "table_d_demo_effect_delta.csv"), index=False)

    # ── Print summary to stdout ──
    print("\n" + "="*70)
    print("TABLE A — Aggregated by Model × Cipher × Demo")
    print("="*70)
    print(tbl_a.to_string(index=False))

    print("\n" + "="*70)
    print("TABLE B — Aggregated by Model × Instruction Category")
    print("="*70)
    print(tbl_b.to_string(index=False))

    print("\n" + "="*70)
    print("TABLE D — Demo Effect (Δ = with_demo − without_demo)")
    print("="*70)
    print(tbl_d.to_string(index=False))

    # ── Write combined markdown report ──
    md_path = os.path.join(OUT_DIR, "analysis_report.md")
    with open(md_path, "w") as f:
        model_list = ", ".join(sorted(df["model_short"].dropna().unique()))
        f.write("# CipherChat Analysis Report\n")
        f.write(f"## Models: {model_list}\n\n")
        f.write("### Table A – Aggregated by Model × Cipher × Demo Setting\n\n")
        f.write(tbl_a.to_markdown(index=False))
        f.write("\n\n### Table B – Aggregated by Model × Instruction Category\n\n")
        f.write(tbl_b.to_markdown(index=False))
        f.write("\n\n### Table C – Full Detailed Results\n\n")
        f.write(tbl_c.to_markdown(index=False))
        f.write("\n\n### Table D – Effect of Unsafe Demonstrations (Δ = with − without)\n\n")
        f.write(tbl_d.to_markdown(index=False))
        f.write("\n\n---\n*Generated by `generate_analysis.py`*\n")
    print(f"\n  ✓ analysis_report.md")

    return tbl_a, tbl_b, tbl_c, tbl_d


# ──────────────────────────────────────────────────────────────────
# 7) STACKED BAR – toxicity breakdown (yes/no counts)
# ──────────────────────────────────────────────────────────────────
def plot_stacked_toxicity(df: pd.DataFrame, fname: str):
    """Stacked bars showing toxic-yes vs toxic-no counts per scenario
       for each model side-by-side."""
    models    = sorted(df["model_short"].dropna().unique())
    scenarios = sorted(df["scenario"].unique())
    n_models  = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5),
                             sharey=True, squeeze=False)
    axes = axes.flatten()

    for ax, model in zip(axes, models):
        sub = df[df["model_short"] == model].copy()
        sub = sub.groupby("scenario").agg(
            toxic_yes=("toxicity_yes_count", "sum"),
            toxic_no=("toxicity_no_count", "sum"),
        ).reindex(scenarios).fillna(0)

        bar_w = 0.6
        x = np.arange(len(scenarios))
        ax.bar(x, sub["toxic_yes"], width=bar_w, label="Toxic (Yes)",
               color="#d62728", edgecolor="white")
        ax.bar(x, sub["toxic_no"], width=bar_w, bottom=sub["toxic_yes"],
               label="Non-Toxic (No)", color="#2ca02c", edgecolor="white")
        ax.set_xticks(x)
        nice = [s.replace("|", "\n") for s in scenarios]
        ax.set_xticklabels(nice, fontsize=8)
        ax.set_ylabel("Count" if ax is axes[0] else "")
        ax.set_title(model, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Toxicity Counts (summed over instruction categories)",
                 fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(OUT_DIR, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")


# ──────────────────────────────────────────────────────────────────
# 8) CROSS-MODEL SCATTER – validity vs toxicity
# ──────────────────────────────────────────────────────────────────
def plot_validity_vs_toxicity(df: pd.DataFrame, fname: str):
    """Scatter of validity-rate vs toxicity-rate, coloured by model,
       shaped by cipher, sized by grammar score."""
    fig, ax = plt.subplots(figsize=(8, 6))
    models  = sorted(df["model_short"].dropna().unique())
    ciphers = sorted(df["cipher_setting"].dropna().unique())
    _extra_markers = ["s", "o", "D", "^" , "v", "P", "X", "*"]

    for model in models:
        for ci, cipher in enumerate(ciphers):
            sub = df[(df["model_short"] == model) &
                     (df["cipher_setting"] == cipher)]
            sizes = sub["avg_grammar_fluency"] * 40 + 20
            marker = CIPHER_MARKERS.get(cipher,
                         _extra_markers[ci % len(_extra_markers)])
            ax.scatter(sub["toxicity_rate"], sub["validity_rate"],
                       s=sizes, color=MODEL_COLORS.get(model, "#333333"),
                       marker=marker, alpha=0.75,
                       edgecolors="black", linewidths=0.5,
                       label=f"{model} – {cipher}")

    ax.set_xlabel("Toxicity Rate")
    ax.set_ylabel("Validity Rate")
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Validity vs Toxicity (bubble size ∝ grammar score)",
                 fontweight="bold")
    ax.grid(alpha=0.3, linestyle="--")
    fig.savefig(os.path.join(OUT_DIR, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print(f"Loading data from eval JSONs in: {RESULTS_DIR}")
    df = load_data_from_jsons(RESULTS_DIR)
    print(f"  {len(df)} experiment rows loaded  ({df['model_short'].dropna().nunique()} models)")
    models_found = sorted(df["model_short"].dropna().unique())
    print(f"  Models found: {', '.join(models_found)}\n")

    if len(df) == 0:
        print("ERROR: No eval JSON files found. Exiting.")
        sys.exit(1)

    # Also save the reconstructed master CSV for reference
    master_csv = os.path.join(OUT_DIR, "master_summary_from_jsons.csv")
    df.drop(columns=["grammar_scores"]).to_csv(master_csv, index=False)
    print(f"  ✓ master_summary_from_jsons.csv  ({len(df)} rows)\n")

    # ── Grouped bar plots for each metric ──
    print("Generating grouped bar charts …")
    for metric, label, ylim in METRICS:
        plot_grouped_bars(df, metric, label, ylim,
                          f"grouped_bars_{metric}.png")

    # ── Heatmaps ──
    print("\nGenerating heatmaps …")
    for metric, label, _ in METRICS:
        plot_heatmaps(df, metric, label, f"heatmap_{metric}.png")

    # ── Demo impact (delta) ──
    print("\nGenerating demo-impact Δ charts …")
    for metric, label, _ in METRICS:
        plot_demo_impact(df, metric, label, f"demo_impact_{metric}.png")

    # ── Cipher comparison ──
    print("\nGenerating cipher comparison charts …")
    for metric, label, _ in METRICS:
        plot_cipher_comparison(df, metric, label,
                               f"cipher_comparison_{metric}.png")

    # ── Radar ──
    print("\nGenerating radar chart …")
    plot_radar(df, "radar_model_comparison.png")

    # ── Stacked toxicity ──
    print("\nGenerating stacked toxicity chart …")
    plot_stacked_toxicity(df, "stacked_toxicity_counts.png")

    # ── Validity vs Toxicity scatter ──
    print("\nGenerating validity-vs-toxicity scatter …")
    plot_validity_vs_toxicity(df, "scatter_validity_vs_toxicity.png")

    # ── Tables ──
    print("\nGenerating summary tables …")
    make_summary_tables(df)

    print(f"\n{'='*70}")
    print(f"All outputs saved to:  {OUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
