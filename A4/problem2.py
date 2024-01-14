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
    sobel_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 8
    sobel_Y = sobel_X.T
    Ix = conv2d(img1, sobel_X)
    Iy = conv2d(img1, sobel_Y)
    It = img1 - img2
    return Ix, Iy, It


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
    # h, w = Ix.shape
    # u = np.zeros((h,w))
    # v = np.zeros((h,w))

    kernel = np.ones((patch_size, patch_size))
    Ix_window = conv2d(Ix ** 2, kernel)
    Iy_window = conv2d(Iy ** 2, kernel)
    Ixy_window = conv2d(Ix * Iy, kernel)
    Ixt_window = conv2d(Ix * It, kernel)
    Iyt_window = conv2d(Iy * It, kernel)

    det = Ix_window * Iy_window - Ixy_window ** 2
    u = (Iy_window * Ixt_window - Ixy_window * Iyt_window) / det
    v = (Ixy_window * Ixt_window - Ix_window * Iyt_window) / det
    return u, v


def warp(img, u, v):
    """Warping of a given image using provided optical flow.

    Args:
        img: input image as (H, W) np.array
        u, v: optical flow in x and y direction each as (H, W) np.array

    Returns:
        im_warp: warped image as (H, W) np.array
    """
    # Create a grid of coordinates
    h, w = img.shape
    y, x = np.mgrid[0:h, 0:w]

    # Warp the coordinates based on the optical flow
    warped_x = x + u
    warped_y = y + v

    # Flatten the arrays for griddata
    flat_x = x.flatten()
    flat_y = y.flatten()
    flat_warped_x = warped_x.flatten()
    flat_warped_y = warped_y.flatten()

    # Interpolate the values at the warped coordinates
    warped_image = interpolate.griddata((flat_x, flat_y), img.flatten(), (flat_warped_x, flat_warped_y), method='linear')

    # Reshape the interpolated values back to the original image shape
    warped_image = warped_image.reshape(h, w)

    return warped_image
