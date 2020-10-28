import numpy as np
import os
import argparse
import imageio
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def dis_matrix(flat_rgb, centers):
    return cdist(flat_rgb, centers, metric="euclidean")


def kmeans(img, k, max_iter=10):
    # initialize k centers
    flat_rgb = img.reshape(-1, 3)
    centers = flat_rgb[np.random.choice(len(flat_rgb), k)]
    cluster_index_arr = None
    for _ in range(max_iter):
        # update assignments
        cluster_index_arr = np.argmin(dis_matrix(flat_rgb, centers), axis=1)
        for cluster_index in range(k):
            # update means
            centers[cluster_index] = flat_rgb[cluster_index_arr == cluster_index].mean(
                axis=0
            )

    compressed_rgb = np.zeros(len(flat_rgb) * 3).reshape(len(flat_rgb), 3)
    for cluster_index in range(k):
        compressed_rgb[cluster_index_arr == cluster_index] = centers[cluster_index]

    return compressed_rgb


def kmeans_helper(path, k):
    img = imageio.imread(path).astype(np.float32) / 255
    h, w, _ = img.shape
    compressed_rgb = kmeans(img, k).reshape(h, w, 3)
    output_path = (
        f"{os.path.dirname(path)}{os.path.splitext(os.path.basename(path))[0]}_k{k}.jpg"
    )
    imageio.imwrite(output_path, (compressed_rgb * 255).astype(np.uint8))
    ratio = os.path.getsize(output_path) / os.path.getsize(path)
    return compressed_rgb, ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Instructions:")
    parser.add_argument("-p", help="image path", dest="path", required=True)
    parser.add_argument(
        "-k", help="number of clusters", dest="k", type=int, required=True
    )
    parser.add_argument("-s", help="show image", dest="show", action="store_true")
    args = parser.parse_args()
    rgb, ratio = kmeans_helper(args.path, args.k)
    print(f"{ratio=}")
    if args.show:
        plt.imshow(rgb)
        plt.show()
