# [ACC_htba, ACC_Ftrojan, ASR_htba, ASR_Ftrojan]
results = {
    "knn": {
        "byol": {
            "uncleansed": [58.2, 57.1, 94.8, 81.2],
            "cleansed": {
                "v30r70": [56.8, 56.6, 91.7, 80.1],
                "v45r105": [56.5, 56.5, 90.9, 79.6],
            },
        },
        "mocov2": {
            "uncleansed": [53.1, 56.7, 22, 34.6],
            "cleansed": {
                "v30r70": [52.8, 56.5, 15.9, 30.1],
                "v45r105": [52.5, 56.4, 14.6, 29.0],
            },
        },
        "simclr": {
            "uncleansed": [61.8, 62.8, 88.9, 49.3],
            "cleansed": {
                "v30r70": [47.7, 51.6, 53.2, 35.7],
                "v45r105": [40.5, 42.0, 17.2, 11.3],
            },
        },
    },
    "linear": {
        "byol": {
            "uncleansed": [52.8, 52.9, 96.7, 79.7],
            "cleansed": {
                "v30r70": [51.9, 51.6, 96.4, 78.9],
                "v45r105": [51.6, 51.2, 96.3, 78.5],
            },
        },
        "mocov2": {
            "uncleansed": [41.7, 46.2, 1.5, 20],
            "cleansed": {
                "v30r70": [41.3, 46.0, 1.7, 15.9],
                "v45r105": [41.2, 45.9, 1.5, 14.0],
            },
        },
        "simclr": {
            "uncleansed": [57, 58.4, 91.4, 50.8],
            "cleansed": {
                "v30r70": [33.0, 35.0, 89.1, 20.7],
                "v45r105": [26.5, 24.3, 67.6, 2.1],
            },
        },
    },
}

uncleansed_acc_values = (
    results["knn"]["byol"]["uncleansed"][:2]
    + results["knn"]["mocov2"]["uncleansed"][:2]
    + results["knn"]["simclr"]["uncleansed"][:2]
    + results["linear"]["byol"]["uncleansed"][:2]
    + results["linear"]["mocov2"]["uncleansed"][:2]
    + results["linear"]["simclr"]["uncleansed"][:2]
)
uncleansed_asr_values = (
    results["knn"]["byol"]["uncleansed"][2:]
    + results["knn"]["mocov2"]["uncleansed"][2:]
    + results["knn"]["simclr"]["uncleansed"][2:]
    + results["linear"]["byol"]["uncleansed"][2:]
    + results["linear"]["mocov2"]["uncleansed"][2:]
    + results["linear"]["simclr"]["uncleansed"][2:]
)
cleansed_acc_values_v30r70 = (
    results["knn"]["byol"]["cleansed"]["v30r70"][:2]
    + results["knn"]["mocov2"]["cleansed"]["v30r70"][:2]
    + results["knn"]["simclr"]["cleansed"]["v30r70"][:2]
    + results["linear"]["byol"]["cleansed"]["v30r70"][:2]
    + results["linear"]["mocov2"]["cleansed"]["v30r70"][:2]
    + results["linear"]["simclr"]["cleansed"]["v30r70"][:2]
)
cleansed_asr_values_v30r70 = (
    results["knn"]["byol"]["cleansed"]["v30r70"][2:]
    + results["knn"]["mocov2"]["cleansed"]["v30r70"][2:]
    + results["knn"]["simclr"]["cleansed"]["v30r70"][2:]
    + results["linear"]["byol"]["cleansed"]["v30r70"][2:]
    + results["linear"]["mocov2"]["cleansed"]["v30r70"][2:]
    + results["linear"]["simclr"]["cleansed"]["v30r70"][2:]
)
cleansed_acc_values_v45r105 = (
    results["knn"]["byol"]["cleansed"]["v45r105"][:2]
    + results["knn"]["mocov2"]["cleansed"]["v45r105"][:2]
    + results["knn"]["simclr"]["cleansed"]["v45r105"][:2]
    + results["linear"]["byol"]["cleansed"]["v45r105"][:2]
    + results["linear"]["mocov2"]["cleansed"]["v45r105"][:2]
    + results["linear"]["simclr"]["cleansed"]["v45r105"][:2]
)
cleansed_asr_values_v45r105 = (
    results["knn"]["byol"]["cleansed"]["v45r105"][2:]
    + results["knn"]["mocov2"]["cleansed"]["v45r105"][2:]
    + results["knn"]["simclr"]["cleansed"]["v45r105"][2:]
    + results["linear"]["byol"]["cleansed"]["v45r105"][2:]
    + results["linear"]["mocov2"]["cleansed"]["v45r105"][2:]
    + results["linear"]["simclr"]["cleansed"]["v45r105"][2:]
)


# Plot six aggregated bars: Acc (uncleansed, cleansed v30r70, cleansed v45r105),
# then ASR (uncleansed, cleansed v30r70, cleansed v45r105).
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def safe_mean(lst):
    return sum(lst) / len(lst) if len(lst) > 0 else 0.0


acc_unc = safe_mean(uncleansed_acc_values)
acc_cl_v30 = safe_mean(cleansed_acc_values_v30r70)
acc_cl_v45 = safe_mean(cleansed_acc_values_v45r105)
asr_unc = safe_mean(uncleansed_asr_values)
asr_cl_v30 = safe_mean(cleansed_asr_values_v30r70)
asr_cl_v45 = safe_mean(cleansed_asr_values_v45r105)

values = [acc_unc, acc_cl_v30, acc_cl_v45, asr_unc, asr_cl_v30, asr_cl_v45]
labels = [
    "-",
    "Cleansed v30 r70",
    "Cleansed v45 r105",
    "-",
    "Cleansed v30 r70",
    "Cleansed v45 r105",
]
# colors: darker for uncleansed, lighter tints for cleansed; orange for ASR
colors = ["#1f77b4", "#8fb3e6", "#8fb3e6", "#ff7f0e", "#ffbb78", "#ffbb78"]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(range(len(values)), values, color=colors)
ax.set_xticks(range(len(values)))
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylabel("Value")
ax.set_title("ViT set1 aggregated results")

# apply mesh/hatch overlay for ASR bars (indices 3,4,5)
for i, b in enumerate(bars):
    if i >= 3:
        b.set_hatch("///")
        b.set_edgecolor("k")
        b.set_linewidth(0.5)
    # annotate
    v = values[i]
    offset = 0.5 if abs(v) < 1 else abs(v) * 0.02
    ax.text(
        b.get_x() + b.get_width() / 2,
        v + offset,
        f"{v:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

# Legend: ACC (blue) and ASR (orange with hatch)
legend_elements = [
    Patch(facecolor="#1f77b4", label="ACC"),
    Patch(facecolor="#ff7f0e", hatch="///", edgecolor="k", label="ASR"),
]
ax.legend(handles=legend_elements, loc="upper right")
fig.tight_layout()
plt.savefig("vit_set1_summary.png", dpi=200)
plt.show()
