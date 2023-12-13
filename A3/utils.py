import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def show_images(ims, hw, title="", size=(8, 2)):
    assert ims.shape[0] < 10, "Too many images to display"
    n = ims.shape[0]
    
    # visualising the result
    fig = plt.figure(figsize=size)
    for i, im in enumerate(ims):
        fig.add_subplot(1, n, i + 1)
        plt.imshow(im.reshape(*hw), "gray")
        plt.axis("off")
    fig.suptitle(title)
 
 
def load_img(path):
    color = Image.open(path)
    gray = color.convert("L")
    color = np.array(color) / 255
    gray = np.array(gray) / 255
    return color, gray


def plot_heatmap(img, title=""):
    plt.imshow(img, "jet", interpolation="none")
    plt.axis("off")
    plt.title(title)
    
    
def show_points(img, rows, cols):
    plt.figure()
    plt.imshow(img, interpolation="none")
    plt.plot(cols, rows ,"xr", linewidth=8)
    plt.axis("off")
