import os, sys, json, math
import numpy as np
import matplotlib.pyplot as plt

import pickle
from tqdm import tqdm

# Add parent directory to path for imports as testing is in subdir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Will use the massive context window of 5.4 to try and summarize all the comments in one go
# 5.4 is quite expensive, so I will be doing this as little as possible 
from llmconfig import commercial_llm_config_5_4
from flexibleAgents.agentChat import flexibleAgentChat



# Create comparison histograms/bar charts for the results data
import os
import math
import numpy as np
import matplotlib.pyplot as plt

# Init result format. Function to avoid shallow copy.
def make_result_format():
    return {
        "correctness": [],
        "scores": [],
        "comments": [],
        "cost": [],
        "inputTokens": [],
        "outputTokens": [],
        "totalTokens": [],
    }

# Gets results from disk (compiles and saves them if not done yet)
def retrieveResultsFromDisk():
    # Check if compiled results already exist to avoid re-processing all the data every time
    if os.path.exists(os.path.join(os.path.dirname(__file__), "compiled_results.pkl")):
        print("Compiled results already exist. Loading from disk...")
        with open(os.path.join(os.path.dirname(__file__), "compiled_results.pkl"), "rb") as f:
            results = pickle.load(f)
        return results

    results = {
        "gpt-4o-mini": {
            "basicChat": make_result_format(),
            "flexibleChat": make_result_format()
        },
        "gpt-5-nano": {
            "basicChat": make_result_format(),
            "flexibleChat": make_result_format()
        }
    }

    testedModels = ["gpt-4o-mini", "gpt-5-nano"]
    testedModelKeys = ["gpt-4o-mini-2024-07-18", "gpt-5-nano-2025-08-07"]
    testedSolvers = ["basicChat", "flexibleChat"]

    print(os.path.dirname(__file__))

    # Load all the pickled results data and print progress
    problemsDir = os.path.join(os.path.dirname(__file__), "problems")
    for dir in tqdm(os.listdir(problemsDir), desc="Loading Results"):
        for model, modelKey in zip(testedModels, testedModelKeys):
            for solverName in testedSolvers:
                resultsDir = os.path.join(problemsDir, dir, "results")

                # Load the LLM scoring results
                with open(os.path.join(resultsDir, f"evaluation_{solverName}_{model}.pkl"), "rb") as f:
                    evaluation = json.loads(pickle.load(f))

                    correctness = evaluation["isAnswerCorrect"]
                    explanationRating = evaluation["explanationRating"]
                    comment = evaluation["comments"]

                # Load the cost results
                with open(os.path.join(resultsDir, f"{solverName}_{model}_cost.pkl"), "rb") as f:
                    costFull = pickle.load(f)

                    costWithCachedTokens = costFull["usage_including_cached_inference"]
                    
                    totalCost = costWithCachedTokens["total_cost"]
                    inputTokens = costWithCachedTokens[modelKey]["prompt_tokens"]
                    outputTokens = costWithCachedTokens[modelKey]["completion_tokens"]
                    totalTokens = costWithCachedTokens[modelKey]["total_tokens"]

                    assert totalTokens == inputTokens + outputTokens, "Total tokens should be the sum of input and output tokens."

                # Append results to the appropriate arrays
                results[model][solverName]["correctness"].append(correctness)
                results[model][solverName]["scores"].append(explanationRating)
                results[model][solverName]["comments"].append(comment)
                results[model][solverName]["cost"].append(totalCost)
                results[model][solverName]["inputTokens"].append(inputTokens)
                results[model][solverName]["outputTokens"].append(outputTokens)
                results[model][solverName]["totalTokens"].append(totalTokens)
    
    # Save the compiled results to disk for future use
    with open(os.path.join(os.path.dirname(__file__), "compiled_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    # print(f"Are results equal: {results['gpt-4o-mini'] == results['gpt-5-nano']}")

    return results



def createComparisons(results, output_dir=os.path.join(os.path.dirname(__file__), "charts"), hist_bins = 100, show=False):
    """
    Creates:
    1. A multi-panel grouped bar chart figure with one subplot per summary statistic,
       each subplot using its own y-scale.
    2. One histogram per raw statistic comparing all model-solver pairs together,
       using explicit bin edges to avoid collapsing into a single bar.

    Parameters
    ----------
    results : dict
        Nested result dictionary.
    output_dir : str
        Directory where plots will be saved.
    show : bool
        Whether to display plots.

    Returns
    -------
    dict
        File paths of generated plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    def to_numeric_array(values):
        arr = []
        for v in values:
            if v is None:
                continue
            if isinstance(v, (int, float, np.integer, np.floating)):
                try:
                    fv = float(v)
                    if not math.isnan(fv):
                        arr.append(fv)
                except Exception:
                    pass
            else:
                try:
                    fv = float(v)
                    if not math.isnan(fv):
                        arr.append(fv)
                except (TypeError, ValueError):
                    pass
        return np.array(arr, dtype=float)

    def correctness_to_binary(values):
        out = []
        for v in values:
            if isinstance(v, bool):
                out.append(int(v))
            elif isinstance(v, (int, float)):
                out.append(1 if v > 0 else 0)
            elif isinstance(v, str):
                val = v.strip().lower()
                if val in {"true", "correct", "yes", "1", "pass"}:
                    out.append(1)
                elif val in {"false", "incorrect", "no", "0", "fail"}:
                    out.append(0)
        return np.array(out, dtype=float)

    def make_bin_edges(data_arrays, stat_key, xlim=None, max_bins=50):
        combined = np.concatenate([a for a in data_arrays if len(a) > 0]) if any(len(a) > 0 for a in data_arrays) else np.array([])
        if len(combined) == 0:
            return np.array([0, 1])

        if stat_key == "correctness":
            return np.array([-0.5, 0.5, 1.5])

        if xlim is not None:
            dmin, dmax = xlim
        else:
            dmin, dmax = combined.min(), combined.max()

        if dmin == dmax:
            pad = max(abs(dmin) * 0.05, 0.5)
            return np.array([dmin - pad, dmax + pad])

        is_integer_like = np.all(np.isclose(combined, np.round(combined)))

        if is_integer_like and (dmax - dmin) <= 100:
            return np.arange(math.floor(dmin) - 0.5, math.ceil(dmax) + 1.5, 1.0)

        return np.linspace(dmin, dmax, max_bins + 1)

    pair_names = []
    summary_stats = {
        "Correctness %": [],
        "Avg Score": [],
        "Avg Cost": [],
        "Avg Input Tokens": [],
        "Avg Output Tokens": [],
        "Avg Total Tokens": [],
        "Correctness %": []
    }

    raw_stats = {
        "scores": {},
        "cost": {},
        "inputTokens": {},
        "outputTokens": {},
        "totalTokens": {},
        "correctness": {}
    }

    hist_xlims = {
        "scores": (0, 10),
        "cost": (0, 0.01),
        "inputTokens": (0, 100000),
        "outputTokens": (0, 50000),
        "totalTokens": (0, 150000),
        # "correctness": (-0.5, 1.5)
    }

    for model_name, solvers in results.items():
        for solver_name, metrics in solvers.items():
            pair_label = f"{model_name}\n{solver_name}"
            pair_names.append(pair_label)

            correctness = correctness_to_binary(metrics.get("correctness", []))
            scores = to_numeric_array(metrics.get("scores", []))
            cost = to_numeric_array(metrics.get("cost", []))
            input_tokens = to_numeric_array(metrics.get("inputTokens", []))
            output_tokens = to_numeric_array(metrics.get("outputTokens", []))
            total_tokens = to_numeric_array(metrics.get("totalTokens", []))

            summary_stats["Correctness %"].append(correctness.mean() * 100 if len(correctness) else 0)
            summary_stats["Avg Score"].append(scores.mean() if len(scores) else 0)
            summary_stats["Avg Cost"].append(cost.mean() if len(cost) else 0)
            summary_stats["Avg Input Tokens"].append(input_tokens.mean() if len(input_tokens) else 0)
            summary_stats["Avg Output Tokens"].append(output_tokens.mean() if len(output_tokens) else 0)
            summary_stats["Avg Total Tokens"].append(total_tokens.mean() if len(total_tokens) else 0)

            raw_stats["scores"][pair_label] = scores
            raw_stats["cost"][pair_label] = cost
            raw_stats["inputTokens"][pair_label] = input_tokens
            raw_stats["outputTokens"][pair_label] = output_tokens
            raw_stats["totalTokens"][pair_label] = total_tokens
            raw_stats["correctness"][pair_label] = correctness

    output_files = {}
    colors = plt.cm.tab10.colors

    # --- Multi-panel grouped bar chart: separate y-scale per statistic ---
    stat_names = list(summary_stats.keys())
    n_pairs = len(pair_names)
    x = np.arange(n_pairs)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, stat_name in enumerate(stat_names):
        ax = axes[idx]
        values = summary_stats[stat_name]
        bars = ax.bar(x, values, color=[colors[i % len(colors)] for i in range(n_pairs)])

        ax.set_title(stat_name)
        ax.set_xticks(x)
        ax.set_xticklabels(pair_names, rotation=20, ha="right")
        ax.set_ylabel("Value")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        ymax = max(values) if values else 1
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    fig.suptitle("Summary Statistics by Model-Solver Pair", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    summary_path = os.path.join(output_dir, "summary_statistics_separate_scales.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    output_files["summary_chart"] = summary_path

    # --- Histograms per statistic comparing all model-solver pairs ---
    hist_titles = {
        "scores": "Scores (1-10, LLM-assigned)",
        "cost": "Inference Cost in $",
        "inputTokens": "Input Tokens",
        "outputTokens": "Output Tokens",
        "totalTokens": "Total Tokens",
        "correctness": "Correctness Percentage",
    }

    for stat_key, title in hist_titles.items():
        fig, ax = plt.subplots(figsize=(12, 6))
        all_arrays = list(raw_stats[stat_key].values())

        custom_xlim = hist_xlims.get(stat_key)
        bin_edges = make_bin_edges(all_arrays, stat_key, xlim=custom_xlim, max_bins=hist_bins)

        has_data = False
        for i, (pair_label, data) in enumerate(raw_stats[stat_key].items()):
            if len(data) == 0:
                continue
            has_data = True

            ax.hist(
                data,
                bins=bin_edges,
                histtype="step",
                linewidth=2,
                label=pair_label,
                color=colors[i % len(colors)]
            )

        if stat_key == "correctness":
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Incorrect", "Correct"])

        if custom_xlim is not None:
            ax.set_xlim(custom_xlim)

        if not has_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

        ax.set_title(f"{title} Distribution Comparison")
        ax.set_xlabel(title)
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(title="Model / Solver", fontsize=9)
        plt.tight_layout()

        hist_path = os.path.join(output_dir, f"{stat_key}_comparison_histogram.png")
        plt.savefig(hist_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

        output_files[f"{stat_key}_histogram"] = hist_path

    return output_files




if __name__ == "__main__":
    results = retrieveResultsFromDisk()
    
    comparison_files = createComparisons(results)

    
