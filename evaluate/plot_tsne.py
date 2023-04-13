import os
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 6))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect="equal")
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=colors)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    # ax.axis('off')
    ax.axis("tight")
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")

    txts = None
    lg = None

    return f, ax, sc, txts, lg


model_name = "nerf_autodecoder"
# model_name = 'SDF_VAE'

if model_name == "nerf_autodecoder":
    folder_dir = "/home/zubairirshad/Downloads/kld_autodecoder_save2"
    latent_name = "latent_shape.pth"
else:
    folder_dir = "/home/zubairirshad/Downloads/DiffisionSDF"
    latent_name = "latent.pth"

folders = os.listdir(folder_dir)

latent_vectors = []
for folder in folders:
    latent = torch.load(
        os.path.join(folder_dir, folder, latent_name), map_location="cpu"
    )
    print("latent shape: ", latent.shape)
    if model_name == "nerf_autodecoder":
        latent_vectors.append(latent.unsqueeze(0))
    else:
        latent_vectors.append(latent)


latent_vectors = torch.cat(latent_vectors, dim=0)
print(latent_vectors.shape)

RS = 20150101

colors = []
for i in range(latent_vectors.shape[0]):
    color = np.array([0.2668, 0.2637, 0.2659])
    colors.append(color)

tsne = TSNE(random_state=RS).fit_transform(latent_vectors.numpy())
f, ax, sc, txts, lg = scatter(tsne, colors)
exp_name = "kld_save_2"
plt.savefig(model_name + exp_name + ".png", dpi=120, bbox_inches="tight")
