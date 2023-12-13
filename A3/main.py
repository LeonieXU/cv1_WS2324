from utils import *
               
#
# Problem 1
#
from problem1 import *
def problem1():
    """Example code implementing the steps in problem 1"""

    # Load images
    y, hw = load_faces("./data/yale_faces")
    mean_face = y.mean(0)
    print("Loaded array: ", y.shape)

    # Using 2 random images for testing
    test_face2 = y[0, :]
    test_face = y[-1, :]
    show_images(np.stack([test_face, test_face2], 0), hw, title="Sample images")

    # Compute PCA
    u, var = compute_pca(y)

    # Compute PCA reconstruction
    ps = [0.5, 0.7, 0.8, 0.95]
    ims = []
    for _, p in enumerate(ps):
        b = basis(u, var, p)
        ims.append(reconstruct(test_face, mean_face, b))

    show_images(np.stack(ims, 0), hw, title="PCA reconstruction")

    # fix some basis
    b = basis(u, var, 0.95)

    # Image search
    top5 = search(y, test_face, b, mean_face, 4)
    show_images(top5, hw, title="Image Search")

    # Interpolation
    ints = interpolate(test_face, test_face2, b, mean_face, 5)
    show_images(ints, hw, title="Interpolation")

    plt.show()


#
# Problem 2
#
from problem2 import *
def problem2():
    # Set paramters and load the image
    sigma = 5
    threshold = 1.5e-3
    color, gray = load_img("data/a3p2.png")

    # Generate filters and compute Hessian
    fx, fy = derivative_filters()
    gauss = gauss_2d(sigma, (25, 25))
    I_xx, I_yy, I_xy = compute_hessian(gray, gauss, fx, fy)

    # Show components of Hessian matrix
    plt.figure()
    plt.subplot(1,4,1)
    plot_heatmap(I_xx, "I_xx")
    plt.subplot(1,4,2)
    plot_heatmap(I_yy, "I_yy")
    plt.subplot(1,4,3)
    plot_heatmap(I_xy, "I_xy")

    # Compute and show Hessian criterion
    criterion = compute_criterion(I_xx, I_yy, I_xy, sigma)
    plt.subplot(1,4,4)
    plot_heatmap(criterion, "Determinant of Hessian")

    # Show all interest points where criterion is greater than threshold
    rows, cols = np.nonzero(criterion > threshold)
    show_points(color, rows, cols)

    # Apply non-maximum suppression and show remaining interest points
    rows, cols = non_max_suppression(criterion, threshold)
    show_points(color, rows, cols)
    plt.show()


#
# Problem 3
#
import problem3 as p3

def problem3():
    """Example code implementing the steps in Problem 2"""
    
    pts_array, feats_array = p3.load_pts_features('data/pts_feats.npz')

    # points and features for image1 and image2
    pts1, pts2 = pts_array
    fts1, fts2 = feats_array

    # Loading images
    img1 = Image.open('data/img1.png')
    img2 = Image.open('data/img2.png')

    im1 = np.array(img1)
    im2 = np.array(img2)

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.imshow(im1)
    plt.plot(pts1[:, 0], pts1[:, 1], 'ro', markersize=1.3)
    plt.subplot(1, 2, 2)
    plt.imshow(im2)
    plt.plot(pts2[:, 0], pts2[:, 1], 'ro', markersize=1.3)

    # display algined image
    H, ix1, ix2 = p3.final_homography(pts1, pts2, feats_array[0],
                                      feats_array[1])

    pts1 = pts1[ix1]
    pts2 = pts2[ix2]

    plt.figure(2)
    plt.subplot(1, 3, 1).set_title('Image 1')
    plt.imshow(im1)
    plt.plot(pts1[:, 0],
             pts1[:, 1],
             'ro',
             markersize=2.3,
             markerfacecolor='none')
    plt.subplot(1, 3, 2).set_title('Image 2')
    plt.imshow(im2)
    plt.plot(pts2[:, 0],
             pts2[:, 1],
             'ro',
             markersize=2.3,
             markerfacecolor='none')
    plt.subplot(1, 3, 3).set_title('Algined image 1')

    H_inv = np.linalg.inv(H)
    H_inv /= H_inv[2, 2]
    im3 = img1.transform(size=(im1.shape[1], im1.shape[0]),
                                     method=Image.PERSPECTIVE,
                                     data=H_inv.ravel(),
                                     resample=Image.BICUBIC)

    plt.show()


if __name__ == "__main__":
    problem1()
    problem2()
    problem3()
