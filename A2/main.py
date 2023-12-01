import matplotlib.pyplot as plt

def show_image(img):
    plt.figure()
    plt.imshow(img, cmap="gray", interpolation="none")
    plt.axis("off")
    plt.show()

    
#
# Problem 1
#
from problem1 import *
def problem1():
    """Example code implementing the steps in problem 1"""
    # generate image
    img = generate_image()
    # detect horizontal and vertical edge maps.
    x_map, y_map, grad_map = sobel_edge(img)
    # show gradient map in x direction
    show_image(x_map)
    # show gradient map in y direction
    show_image(y_map)
    # apply thresholding to obtain edge map from the gradient map.
    edge_map = detect_edges(grad_map)
    # show edge map
    show_image(edge_map)
    # add noise to image
    noisy_image = add_noise(img)
    # get gradient map of the noisy image
    _, _, grad_map_noisy = sobel_edge(noisy_image)
    # apply thesholding to obtain edge map from the gradient map for noisy image.
    # search for the threshold value to obtain a clean edge map
    # shortly explain how you decided the threshold value.
    edge_map_noisy = detect_edges(grad_map_noisy)
    show_image(edge_map_noisy)

# Problem 2
#
from problem2 import *

def problem2():
    """Example code implementing the steps in problem 2"""
    # default values
    fsize = (5, 5)
    sigma = 1.5
    nlevel = 6

    # load image and build Gaussian pyramid
    img = load_img("data/a2p2.png")
    gf = gauss_2d(sigma, fsize)
    gpyramid = gaussian_pyramid(img, nlevel, gf)
    show_image(create_composite_image(gpyramid))

    # build Laplacian pyramid from Gaussian pyramid
    bf = binomial_2d(fsize)
    lpyramid = laplacian_pyramid(gpyramid, bf)

    # amplifiy high frequencies of Laplacian pyramid
    lpyramid_amp = amplify_high_freq(lpyramid)
    show_image(create_composite_image(lpyramid_amp))

    # reconstruct sharpened image from amplified Laplacian pyramid
    img_rec = reconstruct_image(lpyramid_amp, bf)
    show_image(create_composite_image((img, img_rec, img_rec - img)))


if __name__ == "__main__":
    # problem1()
    problem2()
