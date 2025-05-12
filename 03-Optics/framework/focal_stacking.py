"""Computational Photography Sheet 03-Optics - Exercise 1"""

import os
from pathlib import Path

import scipy

# For convolutions and filtering
# See https://docs.scipy.org/doc/scipy/reference/signal.html
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_image_stack(dir_path: Path) -> np.array:
    """Load all .jpg images inside dir as a numpy array

    Args:
        dir_path (Path): Path to directory containing .jpg files

    Returns:
        np.array: DxHxWx3 array, dtype float32, values in range [0,1]
    """

    image_stack = np.array(
        [
            Image.open(dir_path / name)
            for name in os.listdir(dir_path)
            if name.endswith(".jpg")
        ]
    )

    if image_stack is not None:
        assert image_stack.shape == (23, 683, 1024, 3)
    return image_stack


def stack_to_depth(image_stack: np.array) -> np.array:
    """Compute depth map from contrast matrix

    Args:
        image_stack (np.array): DxHxWx3 array

    Returns:
        np.array: HxW array, dtype int64, values in range {0,...,22}
    """
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / 8
    contrast = scipy.ndimage.convolve(image_stack, laplacian, axes=(1, 2))
    return np.argmax(np.sum(np.abs(contrast), axis=3), axis=0)


def combine_stack(image_stack: np.array, depth: np.array) -> np.array:
    """Combine image stack and depth to a single image

    Args:
        image_stack (np.array): DxHxWx3 array
        depth (np.array): HxW array

    Returns:
        np.array: HxWx3 array, dtype float32, values in range [0,1]
    """
    rows, cols = depth.shape
    x_indices, y_indices = np.meshgrid(np.arange(cols), np.arange(rows))
    return image_stack[depth, y_indices, x_indices] / 255


def denoise_depth(depth):
    """Apply denoising to depth map using e.g. a median filter

    Args:
        depth (np.array): HxW array

    Returns:
        np.array: HxW array, dtype int64, values in range {0,...,22}
    """
    return scipy.signal.medfilt2d(depth, kernel_size=5)


def ex1_focal_stacking():
    print("Assignment Sheet 3, Exercise 1: Focal Stacking")
    # Compute
    image_stack = load_image_stack(Path(__file__).parent)
    depth = stack_to_depth(image_stack)
    combined = combine_stack(image_stack, depth)
    denoised_depth = denoise_depth(depth)
    enhanced = combine_stack(image_stack, denoised_depth)

    # Display
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    plt.title("Focal Stacking")

    if depth is not None:
        axs[0, 0].imshow(depth)
    axs[0, 0].axis("off")
    axs[0, 0].set_title("Depth")

    if denoised_depth is not None:
        axs[0, 1].imshow(denoised_depth)
    axs[0, 1].axis("off")
    axs[0, 1].set_title("Denoised Depth")

    if combined is not None:
        axs[1, 0].imshow(combined)
    axs[1, 0].axis("off")
    axs[1, 0].set_title("Combined")

    if enhanced is not None:
        axs[1, 1].imshow(enhanced)
    axs[1, 1].axis("off")
    axs[1, 1].set_title("Enhanced")

    plt.tight_layout()
    os.makedirs(Path(__file__).parent / "out", exist_ok=True)
    fig.savefig(Path(__file__).parent / "out" / "ex1.png")


if __name__ == "__main__":
    ex1_focal_stacking()
