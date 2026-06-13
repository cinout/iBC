import matplotlib.pyplot as plt
import numpy as np

p = [8, 7, 6, 5, 4, 3, 2, 1, 0]

byol_c100_ftrojan = {
    "acc": {
        "mean": [54.5, 54.7, 54.9, 54.8, 54.9, 54.8, 54.6, 55.0, 54.8],
        "std": [0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.5],
    },
    "asr": {
        "mean": [0.0, 0.0, 0.0, 0.1, 0.2, 7.4, 15.7, 83.9, 89.7],
        "std": [0.0, 0.0, 0.0, 0.1, 0.3, 16.0, 20.7, 12.8, 4.1],
    },
}
mocov2_img100_ftrojan = {
    "acc": {
        "mean": [41.3, 41.5, 41.5, 41.6, 41.7, 41.9, 41.8, 42.0, 41.9],
        "std": [0.4, 0.4, 0.4, 0.3, 0.3, 0.5, 0.5, 0.4, 0.5],
    },
    "asr": {
        "mean": [0.8, 0.8, 1.1, 1.3, 1.6, 2.3, 7.1, 36.7, 43.9],
        "std": [0.3, 0.4, 0.4, 0.5, 0.6, 0.9, 12.6, 11.5, 9.2],
    },
}
simclr_c10_htba = {
    "acc": {
        "mean": [80.1, 80.2, 80.4, 80.9, 81.3, 81.4, 82.5, 82.5, 82.2],
        "std": [1.7, 1.4, 1.8, 1.6, 1.2, 1.6, 1.2, 1.7, 2.2],
    },
    "asr": {
        "mean": [23.2, 20.7, 24.1, 28.3, 34.8, 37.3, 56.2, 77.3, 86.5],
        "std": [13.5, 14.6, 17.0, 17.1, 19.5, 23.9, 19.3, 13.2, 2.2],
    },
}


experiments = {
    "BYOL + CIFAR-100 + FTrojan": byol_c100_ftrojan,
    "MoCoV2 + ImageNet-100 + FTrojan": mocov2_img100_ftrojan,
    "SimCLR + CIFAR-10 + HTBA": simclr_c10_htba,
}

fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(6, 5),
    sharex=True,
    constrained_layout=True,
)

for ax, (title, data) in zip(axes, experiments.items()):

    acc_mean = np.array(data["acc"]["mean"])
    acc_std = np.array(data["acc"]["std"])

    asr_mean = np.array(data["asr"]["mean"])
    asr_std = np.array(data["asr"]["std"])

    # ACC
    ax.errorbar(
        p,
        acc_mean,
        # yerr=acc_std,
        fmt="o-",
        capsize=4,
        linewidth=2,
        markersize=6,
        label="ACC",
        color="cornflowerblue",
    )

    # ASR
    ax.errorbar(
        p,
        asr_mean,
        # yerr=asr_std,
        fmt="s-",
        capsize=4,
        linewidth=2,
        markersize=6,
        label="ASR",
        color="coral",
    )

    # # Annotate ACC values
    # for x, y in zip(p, acc_mean):
    #     ax.annotate(
    #         f"{y:.1f}",
    #         (x, y),
    #         textcoords="offset points",
    #         xytext=(0, 8),
    #         ha="center",
    #         fontsize=13,
    #         color="tab:blue",
    #     )

    # # Annotate ASR values
    # for x, y in zip(p, asr_mean):
    #     ax.annotate(
    #         f"{y:.1f}",
    #         (x, y),
    #         textcoords="offset points",
    #         xytext=(0, -14),
    #         ha="center",
    #         fontsize=13,
    #         color="tab:red",
    #     )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("ACC/ASR %", fontsize=11)
    # ax.grid(True, linestyle="--", alpha=0.5)
    if title == list(experiments.keys())[-1]:
        ax.legend(fontsize=11, loc="lower right", framealpha=0.6)

    # Keep p=8 on the left, p=0 on the right
    ax.invert_xaxis()
    ax.tick_params(axis="y", labelsize=11)


axes[-1].set_xlabel("p", fontsize=13)
axes[-1].tick_params(axis="x", labelsize=11)

plt.savefig("ablation_impure.pdf", bbox_inches="tight", dpi=300)
