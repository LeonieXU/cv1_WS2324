from functools import partial
import numpy as np
from scipy import interpolate
from scipy.ndimage import convolve
conv2d = partial(convolve, mode="mirror")


def compute_derivatives(img1, img2):
    """Compute dx, dy and dt derivatives

    Args:
        img1: first image as (H, W) np.array
        img2: second image as (H, W) np.array

    Returns:
        Ix, Iy, It: derivatives of img1 w.r.t. x, y and t as (H, W) np.array
    
    Hint: the provided conv2d function might be useful
    """
    #
    # You code here
    #


def compute_motion(Ix, Iy, It, patch_size=15):
    """Computes one iteration of optical flow estimation.

    Args:
        Ix, Iy, It: image derivatives w.r.t. x, y and t each as (H, W) np.array
        patch_size: specifies the side of the square region R in Eq. (1)
    Returns:
        u: optical flow in x direction as (H, W) np.array
        v: optical flow in y direction as (H, W) np.array
    
    Hint: the provided conv2d function might be useful
    """
    #
    # You code here
    #


def warp(img, u, v):
    """Warping of a given image using provided optical flow.

    Args:
        img: input image as (H, W) np.array
        u, v: optical flow in x and y direction each as (H, W) np.array

    Returns:
        im_warp: warped image as (H, W) np.array
    """
    #
    # You code here
    #
