import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

vision_features = np.load("scripts/visions_for_tsne_50_classes.npy")


bs, n_views, C = vision_features.shape

if bs % 2 != 0:
    raise ValueError(
        "Expected features ordered as all clean classes followed by all poisoned classes."
    )

n_classes = bs // 2


# Flatten for t-SNE: shape [n*views, C]
X_flat = vision_features.reshape(-1, C)

# Optional: first reduce dimension with PCA (improves t-SNE)
from sklearn.decomposition import PCA

X_flat = PCA(n_components=30).fit_transform(X_flat)

# Run t-SNE
tsne = TSNE(n_components=2)
X_2d = tsne.fit_transform(X_flat)  # shape [n*views, 2]

# The feature extraction code concatenates clean images and poisoned images before
# generating views, so the first half of the batch is clean and the second half is poisoned.
labels = np.repeat(np.arange(bs), n_views)
clean_labels = labels[: n_classes * n_views]
clean_points = X_2d[: n_classes * n_views]
poison_points = X_2d[n_classes * n_views :]


# Plot
plt.figure(figsize=(7, 5))
base_cmap = plt.get_cmap("turbo", n_classes)
deep_blue = base_cmap(0)
orange = base_cmap(min(38, n_classes - 1))
class_positions = np.arange(n_classes)
class_colors = np.array(
    [
        (
            deep_blue
            if class_id == 0
            else orange if class_id >= 38 else plt.get_cmap("turbo")(class_id / 38)
        )
        for class_id in class_positions
    ]
)
clean_cmap = ListedColormap(class_colors, name="clean_class_colors")

# Give each clean class its own color. A colorbar scales better than a 50-entry legend.
for class_id in range(n_classes):
    idx = clean_labels == class_id
    plt.scatter(
        clean_points[idx, 0],
        clean_points[idx, 1],
        color=class_colors[class_id],
        marker=".",
        s=2,
        alpha=0.3,
    )

# # Use dark red for every poisoned class. The x marker distinguishes status.
# for class_id in range(n_classes):
#     start = class_id * n_views
#     end = start + n_views
#     plt.scatter(
#         poison_points[start:end, 0],
#         poison_points[start:end, 1],
#         color="darkred",
#         marker="x",
#         s=3,
#         alpha=0.8,
#     )

clean_scatter = plt.scatter([], [], color="black", marker=".", s=2, label="Clean")
poison_scatter = plt.scatter([], [], color="black", marker="x", s=3, label="Poisoned")
plt.legend(
    handles=[clean_scatter, poison_scatter],
    loc="best",
    fontsize=8,
    framealpha=0.8,
)

clean_mappable = plt.cm.ScalarMappable(
    norm=BoundaryNorm(np.arange(n_classes + 1) - 0.5, n_classes),
    cmap=clean_cmap,
)
clean_mappable.set_array(np.arange(n_classes))
colorbar = plt.colorbar(
    clean_mappable,
    ax=plt.gca(),
    pad=0.02,
)
colorbar.set_label("Clean images class spectrum")
colorbar.set_ticks(
    [int(tick) for tick in np.linspace(0, n_classes - 1, min(n_classes, 10))]
)
plt.xticks([])  # remove x ticks
plt.yticks([])  # remove y ticks
plt.tight_layout()
plt.savefig("tsne_plot.pdf", bbox_inches="tight", dpi=300, pad_inches=0.02)
