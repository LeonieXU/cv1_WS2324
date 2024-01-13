import numpy as np
from numpy.linalg import norm


def cost_ssd(patch_l, patch_r):
    """Compute the Sum of Squared Pixel Differences (SSD):
    
    Args:
        patch_l: input patch 1 as (m, m) numpy array
        patch_r: input patch 2 as (m, m) numpy array
    
    Returns:
        cost_ssd: the calcuated SSD cost as a floating point value
    """
    #
    # You code here
    #


def cost_nc(patch_l, patch_r):
    """Compute the normalized correlation cost (NC):
    
    Args:
        patch_l: input patch 1 as (m, m) numpy array
        patch_r: input patch 2 as (m, m) numpy array
    
    Returns:
        cost_nc: the calcuated NC cost as a floating point value
    """
    #
    # You code here
    #


def cost_function(patch_l, patch_r, alpha):
    """Compute the cost between two input window patches
    
    Args:
        patch_l: input patch 1 as (m, m) numpy array
        patch_r: input patch 2 as (m, m) numpy array
        alpha: the weighting parameter for the cost function
    Returns:
        cost_val: the calculated cost value as a floating point value
    """
    #
    # You code here
    #


def pad_image(input_img, window_size, padding_mode='symmetric'):
    """Add padding to the input image based on the window size
    
    Args:
        input_img: input image as 2-dimensional (H,W) numpy array
        window_size: window size as a scalar value (always and odd number)
        padding_mode: padding scheme, it can be 'symmetric', 'reflect', or 'constant'.
            In the case of 'constant' assume zero padding.
        
    Returns:
        padded_img: padded image as a numpy array of the same type as input_img
    """
    #
    # You code here
    #


def compute_disparity(padded_img_l, padded_img_r, max_disp, window_size, alpha):
    """Compute the disparity map using the window-based matching strategy    
    
    Args:
        padded_img_l: The padded left-view input image as 2-dimensional (H,W) numpy array
        padded_img_r: The padded right-view input image as 2-dimensional (H,W) numpy array
        max_disp: the maximum disparity as a search range
        window_size: the patch size for window-based matching, odd number
        alpha: the weighting parameter for the cost function
    Returns:
        disparity: numpy array (H,W) of the same type as image
    """
    #
    # You code here
    #


def compute_aepe(disparity_gt, disparity_res):
    """Compute the average end-point error of the estimated disparity map
    
    Args:
        disparity_gt: ground truth of disparity map as (H, W) numpy array
        disparity_res: estimated disparity map as (H, W) numpy array
    
    Returns:
        aepe: the average end-point error as a floating point value
    """
    #
    # You code here
    #


def optimal_alpha():
    """Return alpha that leads to the smallest EPE (w.r.t. other values)
    Note:
    Remember to check that max_disp = 15, window_size = 11, and padding_mode='symmetric'
    """
    #
    # Once you find the best alpha, you have to fix it
    #
    alpha = np.random.choice([-0.001, -0.01, -0.1, 0.1, 1, 10])
    return alpha


"""
This is a multiple-choice question
"""
def window_based_disparity_matching():
    """Complete the following sentence by choosing the most appropriate answer 
    and return the value as a tuple.
    (Element with index 0 corresponds to Q1, with index 1 to Q2 and so on.)
    
    Q1. [?] is better for estimating disparity values on sharp objects and object boundaries
        1: Using a bigger window size (e.g., 11x11)
        2: Using a smaller window size (e.g., 3x3)
        
    Q2. When using a [?] padding scheme, the artifacts on the right border of the estimated disparity map become the worst.
        1: symmetric
        2: reflect
        3: constant

    Q3. The inaccurate disparity estimation on the left image border happens due to [?].
        1: the inappropriate padding scheme
        2: the limitations of the fixed window size
        3: the absence of corresponding pixels
        
    Example or reponse: (1,1,1)
    """
    return
