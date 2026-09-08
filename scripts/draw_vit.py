import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

methods = ["byol", "mocov2", "simclr"]
trigger_types = ["ftrojan", "htba"]
json_file_path = (
    f"results/results_vit_set2_r105v45.json"  # FIXME: change path if needed
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

            uncleansed_acc_values.extend([uncleansed_knn_acc, uncleansed_linear_acc])
            uncleansed_asr_values.extend([uncleansed_knn_asr, uncleansed_linear_asr])
            cleansed_acc_values.extend([cleansed_knn_acc, cleansed_linear_acc])
            cleansed_asr_values.extend([cleansed_knn_asr, cleansed_linear_asr])

mean_values = {
    "uncleansed_acc": sum(uncleansed_acc_values) / len(uncleansed_acc_values),
    "uncleansed_asr": sum(uncleansed_asr_values) / len(uncleansed_asr_values),
    "cleansed_acc": sum(cleansed_acc_values) / len(cleansed_acc_values),
    "cleansed_asr": sum(cleansed_asr_values) / len(cleansed_asr_values),
}


values = [
    mean_values["uncleansed_acc"],
    mean_values["cleansed_acc"],
    mean_values["uncleansed_asr"],
    mean_values["cleansed_asr"],
]
# labels = [
#     "-",
#     "Cleansed",
#     "-",
#     "Cleansed",
# ]


fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(range(len(values)), values)
ax.set_xticks([])
# ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylabel("ACC/ASR %", fontsize=14)
ax.tick_params(axis="y", labelsize=14)
# ax.set_title("ViT set2 aggregated results")

# apply mesh/hatch overlay for ASR bars (indices 3,4,5)
for i, b in enumerate(bars):
    if i >= 2:
        b.set_hatch("///")
        b.set_edgecolor("k")
        b.set_linewidth(0.5)
    if i == 0:
        b.set_facecolor("cornflowerblue")
    elif i == 1:
        b.set_facecolor("cornflowerblue")
        b.set_alpha(0.7)
    elif i == 2:
        b.set_facecolor("coral")
    elif i == 3:
        b.set_facecolor("coral")
        b.set_alpha(0.7)
    # annotate
    # v = values[i]
    # offset = 0.5 if abs(v) < 1 else abs(v) * 0.02
    # ax.text(
    #     b.get_x() + b.get_width() / 2,
    #     v + offset,
    #     f"{v:.2f}",
    #     ha="center",
    #     va="bottom",
    #     fontsize=9,
    # )

# Legend: ACC (blue) and ASR (orange with hatch)
legend_elements = [
    Patch(
        facecolor="cornflowerblue",
        label="ACC (Uncleansed)",
    ),
    Patch(
        facecolor="cornflowerblue",
        alpha=0.7,
        label="ACC (Cleansed)",
    ),
    Patch(
        facecolor="coral",
        hatch="///",
        edgecolor="k",
        label="ASR (Uncleansed)",
    ),
    Patch(
        facecolor="coral",
        alpha=0.7,
        hatch="///",
        edgecolor="k",
        label="ASR (Cleansed)",
    ),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=14)
fig.tight_layout()
plt.savefig("vit_results.pdf", dpi=300, bbox_inches="tight")
# plt.show()
