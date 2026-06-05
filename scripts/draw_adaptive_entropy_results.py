# [ACC_htba, ACC_Ftrojan, ASR_htba, ASR_Ftrojan]
results = {
    "lambda_0": {
        "knn": {
            "byol": {
                "uncleansed": [
                    56.5,
                    57.4,
                    45.4,
                    87.9,
                ],
                "cleansed": {
                    "mean": [54.5, 54.8, 0.0, 0.0],
                    "std": [0.5, 0.2, 0.1, 0.0],
                },
            },
            "mocov2": {
                "uncleansed": [
                    46,
                    46.2,
                    41.3,
                    0.5,
                ],
                "cleansed": {
                    "mean": [45.6, 44.8, 3.9, 0.1],
                    "std": [0.2, 0.2, 1.8, 0.0],
                },
            },
            "simclr": {
                "uncleansed": [49.7, 50.5, 79.6, 70.9],
                "cleansed": {
                    "mean": [47.3, 48.3, 0.0, 0.0],
                    "std": [0.2, 0.2, 0.0, 0.0],
                },
            },
        },
        "linear": {
            "byol": {
                "uncleansed": [34.9, 36.2, 50.6, 67.3],
                "cleansed": {
                    "mean": [38.6, 35.8, 0.0, 0.8],
                    "std": [0.3, 0.3, 0.0, 1.0],
                },
            },
            "mocov2": {
                "uncleansed": [
                    24.5,
                    27.2,
                    22.5,
                    1.4,
                ],
                "cleansed": {
                    "mean": [22.3, 24.8, 1.0, 0.3],
                    "std": [0.3, 0.4, 0.7, 0.1],
                },
            },
            "simclr": {
                "uncleansed": [
                    25,
                    24.8,
                    68.9,
                    53.2,
                ],
                "cleansed": {
                    "mean": [22.7, 22.8, 0.0, 9.1],
                    "std": [0.2, 0.2, 0.0, 3.8],
                },
            },
        },
    },
    "lambda_0.2": {
        "knn": {
            "byol": {
                "uncleansed": [
                    53.5,
                    54.5,
                    23.1,
                    58.5,
                ],
                "cleansed": {
                    "mean": [21.2, 31.0, 0.0, 8.1],
                    "std": [2.0, 1.6, 0.0, 24.4],
                },
            },
            "mocov2": {
                "uncleansed": [42.3, 42.6, 59.5, 0.4],
                "cleansed": {
                    "mean": [34.8, 31.8, 0.0, 0.0],
                    "std": [0.7, 2.3, 0.0, 0.0],
                },
            },
            "simclr": {
                "uncleansed": [49.5, 49.7, 62.6, 85.6],
                "cleansed": {
                    "mean": [41.8, 44.0, 0.0, 0.0],
                    "std": [0.5, 0.5, 0.0, 0.0],
                },
            },
        },
        "linear": {
            "byol": {
                "uncleansed": [
                    41.6,
                    43.4,
                    30,
                    66.6,
                ],
                "cleansed": {"mean": [2.7, 8.5, 0.0, 0.0], "std": [0.7, 3.1, 0.0, 0.0]},
            },
            "mocov2": {
                "uncleansed": [26.1, 26.1, 6.5, 0.5],
                "cleansed": {
                    "mean": [15.0, 11.7, 0.0, 0.0],
                    "std": [1.9, 2.3, 0.0, 0.0],
                },
            },
            "simclr": {
                "uncleansed": [34.9, 36, 29.8, 81.8],
                "cleansed": {
                    "mean": [25.8, 24.8, 0.0, 0.0],
                    "std": [1.0, 1.6, 0.0, 0.0],
                },
            },
        },
    },
    "lambda_0.5": {
        "knn": {
            "byol": {
                "uncleansed": [
                    53.5,
                    53.4,
                    12.3,
                    82.6,
                ],
                "cleansed": {
                    "mean": [19.2, 18.5, 0.0, 17.8],
                    "std": [2.8, 4.9, 0.0, 35.7],
                },
            },
            "mocov2": {
                "uncleansed": [33.8, 37.1, 6.8, 0.5],
                "cleansed": {"mean": [9.1, 8.2, 0.0, 4.7], "std": [0.4, 1.6, 0.0, 8.0]},
            },
            "simclr": {
                "uncleansed": [
                    47.9,
                    48,
                    84.3,
                    62.7,
                ],
                "cleansed": {
                    "mean": [42.1, 38.9, 0.0, 0.0],
                    "std": [1.2, 1.6, 0.0, 0.0],
                },
            },
        },
        "linear": {
            "byol": {
                "uncleansed": [42.9, 42.7, 33.9, 89.4],
                "cleansed": {"mean": [4.5, 7.3, 0.0, 0.0], "std": [1.0, 2.5, 0.0, 0.0]},
            },
            "mocov2": {
                "uncleansed": [20.2, 22.7, 0, 0.5],
                "cleansed": {"mean": [4.8, 3.6, 0.0, 0.0], "std": [0.9, 0.8, 0.0, 0.0]},
            },
            "simclr": {
                "uncleansed": [35.2, 35.2, 2.4, 27.2],
                "cleansed": {
                    "mean": [18.3, 17.3, 0.0, 0.0],
                    "std": [2.2, 2.4, 0.0, 0.0],
                },
            },
        },
    },
    "lambda_1": {
        "knn": {
            "byol": {
                "uncleansed": [
                    52.1,
                    51.1,
                    70.9,
                    95.6,
                ],
                "cleansed": {
                    "mean": [4.1, 2.0, 0.0, 80.0],
                    "std": [1.8, 1.8, 0.0, 40.0],
                },
            },
            "mocov2": {
                "uncleansed": [27.4, 27.4, 13.4, 0.4],
                "cleansed": {
                    "mean": [1.5, 2.7, 0.0, 6.0],
                    "std": [0.3, 1.0, 0.0, 13.2],
                },
            },
            "simclr": {
                "uncleansed": [
                    47,
                    47.4,
                    79.8,
                    84.2,
                ],
                "cleansed": {
                    "mean": [33.5, 25.4, 0.0, 0.0],
                    "std": [1.0, 1.6, 0.0, 0.0],
                },
            },
        },
        "linear": {
            "byol": {
                "uncleansed": [38.6, 38.5, 69.2, 96.1],
                "cleansed": {
                    "mean": [1.5, 3.0, 99.9, 42.6],
                    "std": [0.3, 1.9, 0.1, 47.3],
                },
            },
            "mocov2": {
                "uncleansed": [16.7, 17.2, 1, 0.6],
                "cleansed": {"mean": [2.2, 1.9, 0.0, 0.0], "std": [0.5, 0.6, 0.0, 0.0]},
            },
            "simclr": {
                "uncleansed": [34.4, 34.4, 1.9, 32.9],
                "cleansed": {"mean": [9.5, 7.1, 0.0, 0.0], "std": [2.1, 1.1, 0.0, 0.0]},
            },
        },
    },
    "lambda_2": {
        "knn": {
            "byol": {
                "uncleansed": [49.6, 49.1, 44, 92.8],
                "cleansed": {
                    "mean": [1.6, 1.4, 0.1, 30.0],
                    "std": [0.5, 0.5, 0.4, 45.8],
                },
            },
            "mocov2": {
                "uncleansed": [25.1, 27, 47, 0.5],
                "cleansed": {"mean": [1.2, 1.6, 0.0, 0.0], "std": [0.0, 0.6, 0.0, 0.0]},
            },
            "simclr": {
                "uncleansed": [44.5, 44.3, 85.7, 86.8],
                "cleansed": {"mean": [5.8, 5.9, 0.0, 0.0], "std": [1.0, 1.4, 0.0, 0.0]},
            },
        },
        "linear": {
            "byol": {
                "uncleansed": [33.2, 34.1, 0.5, 92.2],
                "cleansed": {"mean": [1.3, 2.4, 0.1, 0.0], "std": [0.5, 0.9, 0.2, 0.0]},
            },
            "mocov2": {
                "uncleansed": [15.8, 16.8, 17.9, 0.7],
                "cleansed": {"mean": [2.6, 1.5, 0.0, 0.0], "std": [0.2, 0.6, 0.0, 0.0]},
            },
            "simclr": {
                "uncleansed": [31.9, 31.2, 2.7, 92.6],
                "cleansed": {"mean": [4.0, 3.6, 0.0, 0.0], "std": [0.5, 0.8, 0.0, 0.0]},
            },
        },
    },
}

mean_values = {}
for lambda_key in results.keys():

    uncleansed_acc_values = (
        results[lambda_key]["knn"]["byol"]["uncleansed"][:2]
        + results[lambda_key]["knn"]["mocov2"]["uncleansed"][:2]
        + results[lambda_key]["knn"]["simclr"]["uncleansed"][:2]
        + results[lambda_key]["linear"]["byol"]["uncleansed"][:2]
        + results[lambda_key]["linear"]["mocov2"]["uncleansed"][:2]
        + results[lambda_key]["linear"]["simclr"]["uncleansed"][:2]
    )
    uncleansed_asr_values = (
        results[lambda_key]["knn"]["byol"]["uncleansed"][2:]
        + results[lambda_key]["knn"]["mocov2"]["uncleansed"][2:]
        + results[lambda_key]["knn"]["simclr"]["uncleansed"][2:]
        + results[lambda_key]["linear"]["byol"]["uncleansed"][2:]
        + results[lambda_key]["linear"]["mocov2"]["uncleansed"][2:]
        + results[lambda_key]["linear"]["simclr"]["uncleansed"][2:]
    )
    cleansed_acc_values = (
        results[lambda_key]["knn"]["byol"]["cleansed"]["mean"][:2]
        + results[lambda_key]["knn"]["mocov2"]["cleansed"]["mean"][:2]
        + results[lambda_key]["knn"]["simclr"]["cleansed"]["mean"][:2]
        + results[lambda_key]["linear"]["byol"]["cleansed"]["mean"][:2]
        + results[lambda_key]["linear"]["mocov2"]["cleansed"]["mean"][:2]
        + results[lambda_key]["linear"]["simclr"]["cleansed"]["mean"][:2]
    )
    cleansed_asr_values = (
        results[lambda_key]["knn"]["byol"]["cleansed"]["mean"][2:]
        + results[lambda_key]["knn"]["mocov2"]["cleansed"]["mean"][2:]
        + results[lambda_key]["knn"]["simclr"]["cleansed"]["mean"][2:]
        + results[lambda_key]["linear"]["byol"]["cleansed"]["mean"][2:]
        + results[lambda_key]["linear"]["mocov2"]["cleansed"]["mean"][2:]
        + results[lambda_key]["linear"]["simclr"]["cleansed"]["mean"][2:]
    )
    mean_values[lambda_key] = {
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
fig.suptitle("Adaptive results (Entropy)")
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("adaptive_entropy_summary.png", dpi=200)
plt.show()
