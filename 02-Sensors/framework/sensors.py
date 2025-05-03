""" Computational Photography Sheet 02-Sensors - Exercise 2"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from PIL import Image

def load_grayscale_image(path: Path) -> npt.NDArray[np.float32]:
    """Load image file as grayscale numpy array

    Args:
        path (Path): File path

    Returns:
        np.ndarray: HxW array, dtype float32, values in range [0,1]
    """
    with Image.open(path) as img:
        img = img.convert('L')
        return np.array(img).astype(np.float32) / 255


def add_gaussian_noise(image: np.ndarray, std_dev: float) -> np.ndarray:
    """Return new image with Gaussian noise

    Args:
        image (np.ndarray): HxW input image
        std_dev (float): Gaussian standard deviation

    Returns:
        np.ndarray: Noisy Image
    """
    noise = np.random.normal(loc=0, scale=std_dev, size=image.shape)
    return image + noise

def add_poisson_noise(image: np.ndarray) -> np.ndarray:
    """Return new image with Poisson noise

    Args:
        image (np.ndarray): HxW input image

    Returns:
        np.ndarray: Noisy Image
    """
    return np.random.poisson(lam=(image * 255), size=image.shape) / 255


def ex2_1_noisy_images():
    print("Assignment Sheet 2, Exercise 2.1: Noisy Images")
    image = load_grayscale_image(Path(__file__).parent / "yosemite.png")

    # Figure with 6 subplots
    fig, axs = plt.subplots(3, 2, figsize=(8, 10))

    for axis in axs.flat:
        axis.axis('off')

    # Original in axs[0, 0]
    axs[0, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axs[0, 0].set_title("Original")

    # Poisson in axs[0, 1]
    poisson_img = add_poisson_noise(image)
    axs[0, 1].set_title("Poisson noise")
    axs[0, 1].imshow(poisson_img, cmap="gray")

    # Gaussians in axs[1, 0], axs[1, 1], axs[2, 0], and axs[2, 1]
    std_devs = [0.05, 0.1, 0.2, 0.5]
    gauss_img1 = add_gaussian_noise(image, std_devs[0])
    gauss_img2 = add_gaussian_noise(image, std_devs[1])
    gauss_img3 = add_gaussian_noise(image, std_devs[2])
    gauss_img4 = add_gaussian_noise(image, std_devs[3])
    axs[1, 0].imshow(gauss_img1, cmap="gray")
    axs[1, 0].set_title(f"Gaussian noise (std. dev = {std_devs[0]})")
    axs[1, 1].imshow(gauss_img2, cmap="gray")
    axs[1, 1].set_title(f"Gaussian noise (std. dev = {std_devs[1]})")
    axs[2, 0].imshow(gauss_img3, cmap="gray")
    axs[2, 0].set_title(f"Gaussian noise (std. dev = {std_devs[2]})")
    axs[2, 1].imshow(gauss_img4, cmap="gray")
    axs[2, 1].set_title(f"Gaussian noise (std. dev = {std_devs[3]})")

    plt.tight_layout()
    os.makedirs(Path(__file__).parent / "out", exist_ok=True)
    fig.savefig(Path(__file__).parent / "out" / "ex2_1.png")


def generate_difference_images(perp: np.ndarray, parallel: np.ndarray) -> np.ndarray:
    """Generate 100 difference images with Poisson noise

    Args:
        perp (np.ndarray): HxW perpendicular polarization image
        parallel (np.ndarray): HxW parallel polarization image

    Returns:
        np.ndarray: HxWx100 difference images
    """
    difference_images = np.zeros([perp.shape[0], perp.shape[1], 100])
    for i in range(100):
        noisy_perp = add_poisson_noise(perp)
        noisy_para = add_poisson_noise(parallel)
        difference_images[:, :, i] = noisy_para - noisy_perp
    return difference_images


def ex2_2_skellam_separation():
    print("Assignment Sheet 2, Exercise 2.2: Skellam Separation")
    # Load Images
    parallel = load_grayscale_image(Path(__file__).parent / "parallel.png")
    perp = load_grayscale_image(Path(__file__).parent / "perp.png")

    # Start plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for a in axs.flat:
        a.axis('off')
    axs[0, 0].imshow(parallel.clip(0,1), cmap="gray", vmin=0, vmax=1)
    axs[0, 0].set_title("Parallel")
    axs[1, 0].imshow(perp.clip(0,1), cmap="gray", vmin=0, vmax=1)
    axs[1, 0].set_title("Perpendicular")

    # Simulate 100 difference images
    difference_images = generate_difference_images(perp, parallel)

    # Calculate mean and variance for each pixel
    mean = np.mean(difference_images, axis=2)
    variance = np.var(difference_images, axis=2) * 255 # need to scale variance back up since we
                                                       # scaled the poisson distribution down
                                                       # (mean / 255 -> variance / 255^2)

    # Reconstruct perpendicular and parallel illumination
    # Plot as axs[0, 1] and axs[1, 1]
    # Formula:
    # | mu_diff  | = |  nu   -nu  | * | mu_1 |
    # | var_diff |   | nu^2  nu^2 |   | mu_2 |
    nu = 1.1
    mat = np.linalg.inv([
        [nu, -nu],
        [nu * nu, nu * nu]
    ])
    para_rec = mat[0, 0] * mean + mat[0, 1] * variance
    perp_rec = mat[1, 0] * mean + mat[1, 1] * variance
    axs[0, 1].imshow(para_rec, cmap="gray")
    axs[0, 1].set_title("Parallel (reconstruction)")
    axs[1, 1].imshow(perp_rec, cmap="gray")
    axs[1, 1].set_title("Perpendicular (reconstruction)")

    plt.tight_layout()
    os.makedirs(Path(__file__).parent / "out", exist_ok=True)
    fig.savefig(Path(__file__).parent / "out" / "ex2_2.png")


if __name__ == '__main__':
    ex2_1_noisy_images()
    ex2_2_skellam_separation()
