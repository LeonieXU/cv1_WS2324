import numpy as np
import matplotlib.pyplot as plt


################################################################
#             DO NOT EDIT THIS HELPER FUNCTION                 #
################################################################

def load_image(path):
    return plt.imread(path)

################################################################


def display_image(img):
    """ Show an image with matplotlib

    Args:
        img: Image as numpy array (H,W,3)
    """
    plt.imshow(img)
    return plt.show()


def save_as_npy(path, img):
    """ Save the image array as a .npy file

    Args:
        img: Image as numpy array (H,W,3)
    """
    return np.save(path, img)


def load_npy(path):
    """ Load and return the .npy file

    Args:
        path: Path of the .npy file
    Returns:
        Image as numpy array (H,W,3)
    """
    return np.load(path)  # H, W, 3 = (1024, 1024, 3)


def mirror_horizontal(img):
    """ Create and return a horizontally mirrored image

    Args:
        img: Image as numpy array (H,W,3)

    Returns:
        A horizontally mirrored numpy array (H,W,3).
    """
    img_h = np.fliplr(img)
    return img_h


def display_images(img1, img2):
    """ Display the normal and the mirrored image in one plot

    Args:
        img1: First image to display
        img2: Second image to display
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)  # 1 row, 2 cols
    ax1.imshow(img1)
    ax1.set_title('Original')
    ax2.imshow(img2)
    ax2.set_title('Mirrored')
    return plt.show()
