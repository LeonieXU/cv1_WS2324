from copy import deepcopy
import numpy as np
from PIL import Image
from scipy.special import binom
from scipy.ndimage import convolve


def load_img(path):
    """ Load image file

    Args:
        path: path to image file
    Returns:
        image as (H, W) np.array normalized to [0, 1]
    """
    #
    # You code here
    #
    image = Image.open(path).convert('L')  # Convert image to grayscale
    image_array = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
    return image_array


def gauss_2d(sigma, fsize):
    """ Create a 2D Gaussian filter

    Args:
        sigma: width of the Gaussian filter
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (H, W) np.array
    """
    #
    # You code here
    #
    fsize = np.array(fsize)
    fsize += (1 - fsize % 2)
    i, j = [(ss - 1.) / 2. for ss in fsize]
    y, x = np.ogrid[-i:i + 1, -j:j + 1]
    gaussian = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    return gaussian / gaussian.sum()  # Normalize


def binomial_2d(fsize):
    """ Create a 2D binomial filter

    Args:
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* binomial filter as (H, W) np.array
    """
    #
    # You code here
    #
    size = fsize[0]
    pascal_row = np.array([binom(size - 1, i) for i in range(size)])
    binomial_filter = np.outer(pascal_row, pascal_row)
    return binomial_filter / binomial_filter.sum()  # Normalize



def downsample2(img, f):
    """ Downsample image by a factor of 2
    Filter with Gaussian filter then take every other row/column

    Args:
        img: image to downsample
        f: 2d filter kernel
    Returns:
        downsampled image as (H, W) np.array
    """
    #
    # You code here
    #
    smoothed_image = convolve(img, f, mode='mirror')
    return smoothed_image[::2, ::2]



def upsample2(img, f):
    """ Upsample image by factor of 2

    Args:
        img: image to upsample
        f: 2d filter kernel
    Returns:
        upsampled image as (H, W) np.array
    """
    #
    # You code here
    #
    upsampled = np.zeros((img.shape[0] * 2, img.shape[1] * 2))
    upsampled[::2, ::2] = img
    filtered = convolve(upsampled, f, mode='mirror') * 4
    return filtered



def gaussian_pyramid(img, nlevel, f):
    """ Build Gaussian pyramid from image

    Args:
        img: input image for Gaussian pyramid
        nlevel: number of pyramid levels
        f: 2d filter kernel
    Returns:
        Gaussian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """
    #
    # You code here
    #
    pyramid = [img]
    for _ in range(1, nlevel):
        img = downsample2(img, f)
        pyramid.append(img)
    return pyramid




def laplacian_pyramid(gpyramid, f):
    """ Build Laplacian pyramid from Gaussian pyramid

    Args:
        gpyramid: Gaussian pyramid
        f: 2d filter kernel
    Returns:
        Laplacian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """
    #
    # You code here
    #
    lpyramid = []
    for i in range(len(gpyramid) - 1):
        gu = upsample2(gpyramid[i + 1], f)
        gu = gu[:gpyramid[i].shape[0], :gpyramid[i].shape[1]]
        lpyramid.append(gpyramid[i] - gu)
    lpyramid.append(gpyramid[-1])
    return lpyramid

def create_composite_image(pyramid):
    """ Create composite image from image pyramid
    Arrange from finest to coarsest image left to right, pad images with
    zeros on the bottom to match the hight of the finest pyramid level.
    Normalize each pyramid level individually before adding to the composite image

    Args:
        pyramid: image pyramid
    Returns:
        composite image as (H, W) np.array
    """
    #
    # You code here
    #
    max_height = max(image.shape[0] for image in pyramid)
    total_width = sum(image.shape[1] for image in pyramid)
    composite_image = np.zeros((max_height, total_width), dtype=np.float32)

    current_x = 0
    for image in pyramid:
        normalized_image = image / image.max()  # Normalize each level individually
        composite_image[:image.shape[0], current_x:current_x + image.shape[1]] = normalized_image
        current_x += image.shape[1]

    return composite_image



def amplify_high_freq(lpyramid, l0_factor=1, l1_factor=5):
    """ Amplify frequencies of the finest two layers of the Laplacian pyramid

    Args:
        lpyramid: Laplacian pyramid
        l0_factor: amplification factor for the finest pyramid level
        l1_factor: amplification factor for the second finest pyramid level
    Returns:
        Amplified Laplacian pyramid, data format like input pyramid
    """
    #
    # You code here
    #
    amplified_pyramid = deepcopy(lpyramid)
    amplified_pyramid[0] *= l0_factor
    amplified_pyramid[1] *= l1_factor
    return amplified_pyramid


def reconstruct_image(lpyramid, f):
    """ Reconstruct image from Laplacian pyramid

    Args:
        lpyramid: Laplacian pyramid
        f: 2d filter kernel
    Returns:
        Reconstructed image as (H, W) np.array clipped to [0, 1]
    """
    #
    # You code here
    #
    reconstructed = lpyramid[-1]
    for level in lpyramid[-2::-1]:
        reconstructed = upsample2(reconstructed, f)

        reconstructed = reconstructed[:level.shape[0], :level.shape[1]]
        reconstructed += level
        print(reconstructed.shape)
    return np.clip(reconstructed, 0, 1)