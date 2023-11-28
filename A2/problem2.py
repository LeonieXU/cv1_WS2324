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
    img = Image.open(path)
    print(type(img))
    img = np.array(img)
    return (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalization


def gauss_2d(sigma, fsize):
    """ Create a 2D Gaussian filter

    Args:
        sigma: width of the Gaussian filter
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (H, W) np.array
    """
    # Initialization
    x_size = fsize[0] // 2
    y_size = fsize[1] // 2

    # build gaussian filter: f(x,y)=(1/(sigma **2)*2 *pi)exp(-(x**2+y**2)/(2*(sigma**2)))
    x = np.arange(-y_size, y_size + 1, dtype=float)
    y = np.arange(-x_size, x_size + 1, dtype=float)
    xv, yv = np.meshgrid(x, y)
    gf = np.exp(-(xv ** 2 + yv ** 2) / (2 * (sigma ** 2)))
    gf /= (sigma ** 2) * 2 * np.pi

    # Normalization
    gf /= gf.sum()

    return gf


def binomial_2d(fsize):
    """ Create a 2D binomial filter

    Args:
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* binomial filter as (H, W) np.array
    """
    # pascal triangle
    pt = np.array([binom(fsize[1] - 1, np.arange(fsize[1]))])

    # build binomial filter
    bf = (pt.T) @ pt

    # Normalization
    bf /= bf.sum()
    return bf


def downsample2(img, f):
    """ Downsample image by a factor of 2
    Filter with Gaussian filter then take every other row/column

    Args:
        img: image to downsample
        f: 2d filter kernel
    Returns:
        downsampled image as (H, W) np.array
    """
    down_img = deepcopy(img)
    down_img = convolve(down_img, f, mode='mirror')
    return down_img[::2, ::2]


def upsample2(img, f):
    """ Upsample image by factor of 2

    Args:
        img: image to upsample
        f: 2d filter kernel
    Returns:
        upsampled image as (H, W) np.array
    """
    h, w = img.shape
    up_img = np.zeros((2 * h, 2 * w))

    up_img[::2, ::2] = deepcopy(img)

    up_img = convolve(up_img, f, mode='mirror')
    return up_img * 4


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
    img_list = []

    for i in range(nlevel):
        img_list += [img]
        img = downsample2(img, f)

    return img_list


def laplacian_pyramid(gpyramid, f):
    """ Build Laplacian pyramid from Gaussian pyramid

    Args:
        gpyramid: Gaussian pyramid
        f: 2d filter kernel
    Returns:
        Laplacian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """
    lpyramid = deepcopy(gpyramid)

    for i in range(len(lpyramid) - 1):
        lpyramid[i] = gpyramid[i] - upsample2(gpyramid[i + 1], f)

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
    # init
    length = len(pyramid)
    j = 0
    cpyramid = deepcopy(pyramid)

    # preparing Canvas for pyramid
    cop_img = np.zeros((max(pyramid[i].shape[0] for i in range(length)), sum(pyramid[i].shape[1] for i in range(length))))

    # write in data
    for i in cpyramid:
        i = (i - np.min(i)) / (np.max(i) - np.min(i))  # normalize
        cop_img[0:i.shape[0], j:(j + i.shape[1])] = deepcopy(i)
        j += i.shape[1]
    return cop_img


def amplify_high_freq(lpyramid, l0_factor=1, l1_factor=1):
    """ Amplify frequencies of the finest two layers of the Laplacian pyramid

    Args:
        lpyramid: Laplacian pyramid
        l0_factor: amplification factor for the finest pyramid level
        l1_factor: amplification factor for the second finest pyramid level
    Returns:
        Amplified Laplacian pyramid, data format like input pyramid
    """
    lpyramid_amp = deepcopy(lpyramid)
    lpyramid_amp[0] = lpyramid_amp[0] * l0_factor
    lpyramid_amp[1] = lpyramid_amp[1] * l1_factor
    return lpyramid_amp


def reconstruct_image(lpyramid, f):
    """ Reconstruct image from Laplacian pyramid

    Args:
        lpyramid: Laplacian pyramid
        f: 2d filter kernel
    Returns:
        Reconstructed image as (H, W) np.array clipped to [0, 1]
    """
    n = len(lpyramid) - 1
    rec_lpyramid = deepcopy(lpyramid[n])
    for i in range(n - 1, -1, -1):
        rec_lpyramid = lpyramid[i] + upsample2(rec_lpyramid, f)  # G[i]=L[i]+exbandG[i+1]
    rec_lpyramid = np.clip(rec_lpyramid, 0, 1)  # clipped to [0, 1]
    return rec_lpyramid
