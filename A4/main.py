from utils import *

#
# Problem 1
#
from problem1 import *

def problem1():
    """Example code implementing the steps in Problem 1"""

    # Given parameter. No need to change
    max_disp = 15

    alpha = optimal_alpha()
    print("Alpha: {:4.3f}".format(alpha))

    # Window size. You can freely change, but it should be an odd number
    window_size = 11

    # Load data
    im_left = load_image_gray("data/a4p1_left.png")
    im_right = load_image_gray("data/a4p1_right.png")
    disparity_gt = disparity_read("data/a4p1_gt.png")
   
    # Padding the images
    padded_img_l = pad_image(im_left, window_size, padding_mode='symmetric')
    padded_img_r = pad_image(im_right, window_size, padding_mode='symmetric') 

    # Compute disparity
    disparity_res = compute_disparity(padded_img_l, padded_img_r, max_disp, window_size, alpha)
    aepe = compute_aepe(disparity_gt, disparity_res)
    print("AEPE: {:4.3f}".format(aepe))
    
    show_disparity(disparity_gt, disparity_res)


#
# Problem 2
#
from problem2 import *

def problem2():
    # Loading the image and scaling them to [0, 1]
    img1 = load_img("data/a4p2a.png")
    img2 = load_img("data/a4p2b.png")

    Ix, Iy, It = compute_derivatives(img1, img2) # gradients
    u, v = compute_motion(Ix, Iy, It) # flow

    # stacking for visualization
    of = np.stack([u, v], axis=-1)
    # convert to RGB using wheel colour coding
    rgb_image = flow_to_color(of, clip_flow=5)
    # display
    show_flow(img1, rgb_image)

    # warping 1st image to the second
    img1_warped = warp(img1, u, v)
    show_warped(img2, img1_warped)


if __name__ == "__main__":
    problem1()
    problem2()
