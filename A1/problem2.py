import numpy as np
from scipy.ndimage import convolve


def loadbayer(path):
    """ Load data from file

    Args:
        path: Path of the .npy file
    Returns:
        Data as numpy array (H,W)
    """
    return np.load(path)


def separatechannels(bayerdata):
    """ Separate bayer data into RGB channels so that
    each color channel retains only the respective
    values given by the bayer pattern and missing values
    are filled with zero

    Args:
        bayerdata: Numpy array containing bayer data (H,W)
    Returns:
        red, green, and blue channel as numpy array (H,W)
    """

    print(bayerdata.shape)  # debug
    # H,W = (1024, 1024)
    # initialization
    r = bayerdata.copy()  # deep copy, in order to keep original bayerdata clean (no change)
    g = bayerdata.copy()
    b = bayerdata.copy()

    # missing color = 0, turn pixels no need 0
    r[0::2, 0::2] = 0  # slicing: start from element in 1st row and 1st col, every other row, column, set to zero
    r[1::2] = 0  # slicing: start from 2nd row, every other row, set to zero
    g[0::2, 1::2] = 0
    g[1::2, 0::2] = 0
    b[1::2, 1::2] = 0
    b[0::2] = 0
    return np.array(r), np.array(g), np.array(b)



def assembleimage(r, g, b):
    """ Assemble separate channels into image

    Args:
        r: red channel as numpy array (H,W)
        g: green channel as numpy array (H,W)
        b: blue channel as numpy array (H,W)
    Returns:
        Image as numpy array (H,W,3)
    """
    h, w = r.shape
    img = np.zeros(shape=(h, w, 3))
    if r.size == g.size == b.size:  # check shape consistent
        img = np.stack([r, g, b], axis=2)
        print(img.shape)  # debug
    else:
        print('r,g,b not same shape.')
    return img


def interpolate(r, g, b):
    """ Interpolate missing values in the bayer pattern
    by using bilinear interpolation

    Args:
        r: red channel as numpy array (H,W)
        g: green channel as numpy array (H,W)
        b: blue channel as numpy array (H,W)
    Returns:
        Interpolated image as numpy array (H,W,3)
    """
    # initialization
    kernel_1 = np.array([[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]])  # kernel_1 for r and b channel
    kernel_2 = np.array([[0, 0.25, 0], [0.25, 1, 0.25], [0, 0.25, 0]])  # kernel_2 for g channel

    # convolve with nearest mode
    new_r = convolve(r, kernel_1, mode='mirror')
    new_g = convolve(g, kernel_2, mode='mirror')
    new_b = convolve(b, kernel_1, mode='mirror')
    h, w = new_r.shape

    # output
    img = np.zeros(shape=(h, w, 3))
    if new_r.size == new_g.size == new_b.size:
        img = np.stack([new_r, new_g, new_b], axis=2)
    else:
        print('not same shape.')
    return img
