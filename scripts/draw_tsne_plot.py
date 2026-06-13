import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# # 6 classes
legends = {
    0: "C1 clean",
    1: "C2 clean",
    2: "C3 clean",
    3: "C4 clean",
    5: "C5 clean",
    6: "C1 poison",
    7: "C2 poison",
    8: "C3 poison",
    9: "C4 poison",
    11: "C5 poison",
}

colors = {
    0: "cornflowerblue",
    1: "lightgreen",
    2: "tan",
    3: "thistle",
    5: "paleturquoise",
    6: "salmon",
    7: "salmon",
    8: "salmon",
    9: "salmon",
    11: "salmon",
}

markers = {
    0: "*",
    1: "o",
    2: ".",
    3: "<",
    5: "+",
    6: "*",
    7: "o",
    8: ".",
    9: "<",
    11: "+",
}


# vision_features = np.load("visions_for_tsne_imagenet100_ftrojan_simclr_6class.npy")
vision_features = np.load("scripts/visions_for_tsne_imagenet100_htba_simclr_6class.npy")


bs, n_views, C = vision_features.shape


# Flatten for t-SNE: shape [n*views, C]
X_flat = vision_features.reshape(-1, C)

# Optional: first reduce dimension with PCA (improves t-SNE)
from sklearn.decomposition import PCA

X_flat = PCA(n_components=30).fit_transform(X_flat)

# Run t-SNE
tsne = TSNE(n_components=2)
X_2d = tsne.fit_transform(X_flat)  # shape [n*views, 2]

# Prepare labels: same label for all views of a class
labels = np.repeat(np.arange(bs), n_views)  # shape [n*views]


# Plot
plt.figure(figsize=(4, 2.8))
handles = {}
for i in list(legends.keys()):
    idx = labels == i
    sc = plt.scatter(
        X_2d[idx, 0],
        X_2d[idx, 1],
        label=legends[i],
        c=colors[i],
        marker=markers[i],
        s=10,
    )  # all views same color
    handles[i] = sc

# Order handles: row1 = [0,1,2,3,5], row2 = [6,7,8,9,11]
ordered_keys = [0, 1, 2, 3, 5, 6, 7, 8, 9, 11]
ordered_handles = [handles[k] for k in ordered_keys]
ordered_labels = [legends[k] for k in ordered_keys]

plt.legend(
    ordered_handles[:],
    ordered_labels[:],
    ncols=2,
    loc="lower right",
    handletextpad=0.1,  # space between marker and text
    columnspacing=0.2,  # space between columns
    labelspacing=0.1,  # vertical space between rows
    borderpad=0.1,  # padding inside the legend box
    fontsize=9,
    framealpha=0.5,
)
plt.xticks([])  # remove x ticks
plt.yticks([])  # remove y ticks
plt.tight_layout()
plt.savefig("tsne_plot.pdf", bbox_inches="tight", dpi=300, pad_inches=0.02)
