# FIXME: change adaptive attack name
import json

attack_name = "entro_defense_replace_with_mean"  # entropy, l2_spread, adversarial, corr_penalty, tv, l1_cv, group_entropy
mean_values = {}

mean_values[f"lambda_0"] = {
    "uncleansed_acc": 39.91,
    "uncleansed_asr": 49.12,
    "cleansed_acc": 38.52,
    "cleansed_asr": 1.27,
}


# results: [ACC_htba, ACC_Ftrojan, ASR_htba, ASR_Ftrojan]
lmabda_keys = ["0.2", "0.5", "1", "2"]
methods = ["byol", "mocov2", "simclr"]
trigger_types = ["ftrojan", "htba"]
for lambda_key in lmabda_keys:
    # FIXME: change path if needed
    json_file_path = f"results/results_ada_{attack_name}_lambda_{lambda_key}.json"
    # read json file
    with open(json_file_path, "r") as f:
        content = json.load(f)

    uncleansed_acc_values = []
    uncleansed_asr_values = []
    cleansed_acc_values = []
    cleansed_asr_values = []

    # find results for each method, dataset, and trigger type
    for method in methods:
        for trigger_type in trigger_types:
            matches = [
                v
                for _, v in content.items()
                if v.get("method") == method and v.get("trigger_type") == trigger_type
            ]
            match = matches[0] if matches else None
            if match:
                uncleansed_knn_acc = float(match["clean_acc_800"])
                uncleansed_knn_asr = float(match["back_acc_800"])
                uncleansed_linear_acc = float(match["linear_ACC"])
                uncleansed_linear_asr = float(match["linear_ASR"])
                cleansed_knn_acc = float(match["knn_clean_acc"].split("±")[0])
                cleansed_knn_asr = float(match["knn_back_asr"].split("±")[0])
                cleansed_linear_acc = float(match["linear_clean_acc"].split("±")[0])
                cleansed_linear_asr = float(match["linear_back_asr"].split("±")[0])

                uncleansed_acc_values.extend(
                    [uncleansed_knn_acc, uncleansed_linear_acc]
                )
                uncleansed_asr_values.extend(
                    [uncleansed_knn_asr, uncleansed_linear_asr]
                )
                cleansed_acc_values.extend([cleansed_knn_acc, cleansed_linear_acc])
                cleansed_asr_values.extend([cleansed_knn_asr, cleansed_linear_asr])

    mean_values[f"lambda_{lambda_key}"] = {
        "uncleansed_acc": sum(uncleansed_acc_values) / len(uncleansed_acc_values),
        "uncleansed_asr": sum(uncleansed_asr_values) / len(uncleansed_asr_values),
        "cleansed_acc": sum(cleansed_acc_values) / len(cleansed_acc_values),
        "cleansed_asr": sum(cleansed_asr_values) / len(cleansed_asr_values),
    }


# Plotting: five subplots (one per lambda key), four bars per subplot
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

lambda_order = ["lambda_0", "lambda_0.2", "lambda_0.5", "lambda_1", "lambda_2"]
labels = ["-", "cleansed", "-", "cleansed"]
colors = ["tab:blue", "tab:blue", "tab:orange", "tab:orange"]

fig, axes = plt.subplots(1, len(lambda_order), figsize=(22, 4), sharey=True)
for ax, lk in zip(axes, lambda_order):
    vals = [
        mean_values[lk]["uncleansed_acc"],
        mean_values[lk]["cleansed_acc"],
        mean_values[lk]["uncleansed_asr"],
        mean_values[lk]["cleansed_asr"],
    ]
    bars = ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_title(lk)
    # annotate values on bars
    for i, (b, v) in enumerate(zip(bars, vals)):
        y = v if v is not None else 0
        offset = 0.5 if abs(y) < 1 else abs(y) * 0.02
        ax.text(
            b.get_x() + b.get_width() / 2,
            y + offset,
            f"{y:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        # apply mesh/hatch overlay for ASR bars (indices 2 and 3)
        if i in (2, 3):
            b.set_hatch("///")
            b.set_edgecolor("k")
            b.set_linewidth(0.5)

# legend for Acc vs ASR
legend_elements = [
    Patch(facecolor="tab:blue", label="ACC"),
    Patch(facecolor="tab:orange", hatch="///", edgecolor="k", label="ASR"),
]
axes[0].legend(handles=legend_elements, loc="lower left", title="Legend")
fig.suptitle(f"Adaptive {attack_name}")
fig.tight_layout(rect=(0, 0, 1, 0.95))
plt.savefig(f"adaptive_{attack_name.lower()}_summary.png", dpi=200)
# plt.show()
