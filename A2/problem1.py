import numpy as np
from scipy.ndimage import convolve


def generate_image():
    """ Generates cocentric simulated image in Figure 1.

    Returns:
        Concentric simulated image with the size (210, 210) with increasing intesity through the center
        as np.array.
    """
    # initialization black image square
    img = np.zeros((210, 210))

    # decreasing intensity, easier to be indexed
    for i in range(6):
        k = 15 * (i + 1)
        img[k:(210 - k), k:(210 - k)] += 30
    return img


def sobel_edge(img):
    """ Applies sobel edge filter on the image to obtain gradients in x and y directions and gradient map.
    (see lecture 5 slide 30 for filter coefficients)

    Args:
        img: image to be convolved
    Returns:
        Ix derivatives of the source image in x-direction as np.array
        Iy derivatives of the source image in y-direction as np.array
        Ig gradient magnitude map computed by sqrt(Ix^2+Iy^2) for each pixel
    """
    # sobel operator
    sobel_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 8
    sobel_Y = sobel_X.T

    # derivatives and gradient
    Ix = convolve(img, sobel_X)
    Iy = convolve(img, sobel_Y)
    Ig = np.sqrt(Ix ** 2 + Iy ** 2)
    return Ix, Iy, Ig


def detect_edges(grad_map, threshold=9):
    """ Applies threshold on the edge map to detect edges.

    Args:
        grad_map: gradient map.
        threshold: threshold to be applied.
    Returns:
        edge_map: thresholded gradient map.
    """
    # alternative with no deep copy
    term = grad_map
    edge_map = term.copy()
    edge_map[edge_map < threshold] = 0

    # grad_map with noise
    # print(grad_map.min(), grad_map.max())  # (0.0, 15.90990257669732)

    # In addition to the code, please include your response as a comment to 
    # the following questions: Which threshold recovers the edge map of the 
    # original image when working with the noisy image? How did you 
    # determine this threshold value, and why did you choose it?

    # ANSWER #
    # Threshold = 9
    # by print one sample (grad_map_noisy.min, grad_map_noisy.max) = (0.016308414012899137, 21.15077545375202) from image with noise,
    # we got an approximate pixel range of the gradient map from image with noise.
    # Therefore, the threshold ought to be in it.
    # Try Threshold <= 8: still has clear noise
    # Try Threshold >= 11: unclear edges, not enough precise to detect the edges
    # with threshold in range(9,11), they all have good performance. But finally we got best result with threshold=9

    return edge_map

def add_noise(img, mean=0, variance=15):
    """ Applies Gaussian noise on the image.

    Args:
        img: image in np.array
        mean: mean of the noise distribution.
        variance: variance of the noise distribution.
    Returns:
        noisy_image: gaussian noise applied image.
    """
    # add noisy with Gaussian distribution
    noisy_image = img + np.random.normal(mean, np.sqrt(variance), (210, 210))
    return noisy_image
