import numpy as np
from scipy.ndimage import convolve, maximum_filter

'''
    Citations:
    Harris Corner and Edge Detector(Isaac Berrios): https://medium.com/@itberrios6/harris-corner-and-edge-detector-4169312aa2f8
'''

def derivative_filters():
    """ Create derivative filters for x and y direction

    Returns:
        fx: derivative filter in x direction
        fy: derivative filter in y direction
    """
    # sobel kernels
    fx = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]])

    fy = fx.T
    return fx, fy


def gauss_2d(sigma, fsize):
    """ Create a 2D Gaussian filter

    Args:
        sigma: width of the Gaussian filter
        fsize: (h, w) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (h, w) np.array
    """
    m, n = fsize
    x = np.arange(-m / 2 + 0.5, m / 2)
    y = np.arange(-n / 2 + 0.5, n / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)

    # Calculate the 2D Gaussian filter
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g / np.sum(g)  # normalization


def compute_hessian(img, gauss, fx, fy):
    """ Compute elements of the Hessian matrix

    Args:
        img: numpy array with the image
        gauss: Gaussian filter
        fx: derivative filter in x direction
        fy: derivative filter in y direction

    Returns:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
    """
    # Smooth the image with the Gaussian filter
    smooth = convolve(img, gauss, mode='mirror')

    # Compute the first derivatives
    I_x = convolve(smooth, fx, mode='mirror')
    I_y = convolve(smooth, fy, mode='mirror')

    # Compute the second derivatives
    I_xx = convolve(I_x, fx, mode='mirror')
    I_yy = convolve(I_y, fy, mode='mirror')
    I_xy = convolve(I_x, fy, mode='mirror')
    return I_xx, I_yy, I_xy


def compute_criterion(I_xx, I_yy, I_xy, sigma):
    """ Compute criterion function

    Args:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
        sigma: scaling factor

    Returns:
        criterion: (h, w) np.array of scaled determinant of Hessian matrix
    """
    criterion = sigma ** 4 * (I_xx * I_yy - I_xy ** 2)
    return criterion


def non_max_suppression(criterion, threshold):
    """ Apply non-maximum suppression to criterion values
        and return Hessian interest points

        Args:
            criterion: (h, w) np.array of criterion function values
            threshold: criterion threshold
        Returns:
            rows: (n,) np.array with y-positions of interest points
            cols: (n,) np.array with x-positions of interest points
    """
    # Apply non-maximum suppression using a 5x5 maximum filter
    local_max = maximum_filter(criterion, size=5, mode='mirror')

    # Find locations of interest points
    rows, cols = np.nonzero((criterion == local_max) & (criterion > threshold))
    return rows, cols
