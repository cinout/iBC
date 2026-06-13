# FIXME: change adaptive attack names
import json

# list four attack names to plot (change as needed)
attack_names = [
    "corr_penalty",
    "l2_spread",
    "tv",
    "entropy",
    "entro_defense_remove5",
    # "entro_defense_remove5_replace_with_mean",
]  # entro, l2_spread, adversarial, corr_penalty, tv, l1_cv, group_entropy

full_names = {
    "corr_penalty": "correlation penalty",
    "l2_spread": "L2-norm variance penalty",
    "tv": "smoothness regularization",
    "entropy": "entropy maximization",
    "entro_defense_remove5": "entropy maximization (remove 5 channels)",
    # "entro_defense_remove5_replace_with_mean": "entropy maximization (remove 5 channels)",
}

# results: [ACC_htba, ACC_Ftrojan, ASR_htba, ASR_Ftrojan]
lmabda_keys = ["0.2", "0.5", "1", "2"]
methods = ["byol", "mocov2", "simclr"]
trigger_types = ["ftrojan", "htba"]

# prepare data per attack
all_mean_values = {}
for attack_name in attack_names:
    mean_values = {}
    # baseline lambda_0 (can be adjusted per-attack if needed)
    mean_values[f"lambda_0"] = {
        "uncleansed_acc": 39.91,
        "uncleansed_asr": 49.12,
        "cleansed_acc": 38.52,
        "cleansed_asr": 1.27,
    }

    for lambda_key in lmabda_keys:
        # FIXME: change path if needed
        json_file_path = (
            f"results/adaptive/results_ada_{attack_name}_lambda_{lambda_key}.json"
        )
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
                    if v.get("method") == method
                    and v.get("trigger_type") == trigger_type
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

    all_mean_values[attack_name] = mean_values


# Plotting: one subplot per attack (stacked vertically). Each subplot shows
# grouped bars for each lambda (uncleansed_acc, cleansed_acc, uncleansed_asr, cleansed_asr)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

lambda_order = ["lambda_0", "lambda_0.2", "lambda_0.5", "lambda_1", "lambda_2"]
lambda_labels = ["0", "0.2", "0.5", "1", "2"]

n_attacks = len(attack_names)
fig, axes = plt.subplots(n_attacks, 1, figsize=(12, 3 * n_attacks), sharey=True)
axes = np.atleast_1d(axes)

bar_width = 0.22
ind = np.arange(len(lambda_order))

for ax, attack in zip(axes, attack_names):
    mean_values = all_mean_values[attack]
    # gather series
    series = {
        "uncleansed_acc": [],
        "cleansed_acc": [],
        "uncleansed_asr": [],
        "cleansed_asr": [],
    }
    for lk in lambda_order:
        mv = mean_values.get(lk, {})
        series["uncleansed_acc"].append(mv.get("uncleansed_acc", 0))
        series["cleansed_acc"].append(mv.get("cleansed_acc", 0))
        series["uncleansed_asr"].append(mv.get("uncleansed_asr", 0))
        series["cleansed_asr"].append(mv.get("cleansed_asr", 0))

    # positions for 4 bars per group
    p0 = ind - 1.5 * bar_width
    p1 = ind - 0.5 * bar_width
    p2 = ind + 0.5 * bar_width
    p3 = ind + 1.5 * bar_width

    b0 = ax.bar(p0, series["uncleansed_acc"], width=bar_width, color="cornflowerblue")
    b1 = ax.bar(
        p1, series["cleansed_acc"], width=bar_width, color="cornflowerblue", alpha=0.7
    )
    b2 = ax.bar(p2, series["uncleansed_asr"], width=bar_width, color="coral")
    b3 = ax.bar(p3, series["cleansed_asr"], width=bar_width, color="coral", alpha=0.7)

    # apply hatch to ASR bars
    for b in list(b2) + list(b3):
        b.set_hatch("///")
        b.set_edgecolor("k")
        b.set_linewidth(0.5)

    # # annotate
    # for bars in (b0, b1, b2, b3):
    #     for b in bars:
    #         y = b.get_height() if b.get_height() is not None else 0
    #         offset = 0.25
    #         # offset = 0.5 if abs(y) < 1 else abs(y) * 0.02
    #         ax.text(
    #             b.get_x() + b.get_width() / 2,
    #             y + offset,
    #             f"{y:.2f}",
    #             ha="center",
    #             va="bottom",
    #             fontsize=11,
    #             # fontweight="bold",
    #         )

    ax.set_xticks(ind)
    ax.set_xticklabels(lambda_labels, fontsize=19)
    ax.tick_params(axis="y", labelsize=19)
    ax.set_title(full_names.get(attack), fontweight="bold", fontsize=19)
    ax.set_ylabel("ACC/ASR %", fontsize=19)
    if attack == attack_names[-1]:
        ax.set_xlabel(r"$\alpha$", fontsize=22)

# legend for Acc vs ASR
legend_elements = [
    Patch(
        facecolor="cornflowerblue",
        label="ACC (Uncleansed)",
    ),
    Patch(
        facecolor="coral",
        hatch="///",
        edgecolor="k",
        label="ASR (Uncleansed)",
    ),
    Patch(
        facecolor="cornflowerblue",
        alpha=0.7,
        label="ACC (Cleansed)",
    ),
    Patch(
        facecolor="coral",
        alpha=0.7,
        hatch="///",
        edgecolor="k",
        label="ASR (Cleansed)",
    ),
]
axes[0].legend(
    handles=legend_elements, ncol=2, loc="upper right", fontsize=16, framealpha=0.6
)
# fig.suptitle("Adaptive attacks summary")
fig.tight_layout(rect=(0, 0, 1, 0.95))
plt.savefig("adaptive_attacks_summary.pdf", bbox_inches="tight", dpi=300)
