import numpy as np
import os
from PIL import Image

"""
source: https://www.kaggle.com/code/vishvaachhatrara/pca-on-face-images-from-scratch
"""


def load_faces(path, ext=".pgm"):
    """Load faces into an array (N, M),
    where N is the number of face images and M is the dimensionality 
    (height*width for greyscale).
    
    Hint: os.walk() supports recursive listing of files 
    and directories in a path
    
    Args:
        path: path to the directory with face images
        ext: extension of the image files (you can assume .pgm only)
    
    Returns:
        x: (N, M) array
        hw: (H, W) tuple
    """
    x = []

    # Load faces
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(ext):
                x += [np.array(Image.open(os.path.join(root, file)))]

    x = np.array(x)
    n = x.shape[0]
    hw = x.shape[1:]  # image dimension
    x = x.reshape(n, -1)  # vectorize images

    return x, hw


def PCA_mcq():
    """
    Which method will be a more reasonable choice in your implementation
    0: SVD
    1: Eigendecomposition
    Why?
    0: It is more computationally efficient for our problem
    1: It allows to compute eigenvectors and eigenvalues of any matrix
    2: It can be applied to any matrix and is more numerically stable
    3: We can find the eigenvalues we need for our problem from the singular values
    4: We can find the singular values we need for our problem from the eigenvalues

    Return your answer as a tuple, e.g., return (0,0,1) means that the more reasonable
    method is SVD because it is more computationally efficient for our problem, and
    allows to compute eigenvector and eigenvalues of any matrix.
    """
    return tuple([0, 2, 3])


def compute_pca(X):
    """PCA implementation
    
    Args:
        X: (N, M) an array with N M-dimensional features
    
    Returns:
        u: (M, M) bases with principal components
        var: (N,) corresponding variance
    """
    # center the data
    mean_face = np.mean(X, axis=0)
    X_hat = X - mean_face  # each row represents a variable

    # find the singular values through SVD
    u, s, vh = np.linalg.svd(X_hat)

    # compute variance
    var = s ** 2 / X.shape[0]
    print(var.shape)
    return vh.T, var


def basis(u, var, p=0.5):
    """Return the minimum number of basis vectors from matrix U such 
    that they account for at least p percent of total variance.
    
    Hint: Do the singular values really represent the variance?
    
    Args:
        u: (M, M) numpy array containing principal components.
        For example, i'th vector is u[:, i]
        var: (N, ) numpy array containing the variance along the principal components.
        p: percent of total variance that should be contained.
        
    Returns:
        v: (M, D) numpy array that contains M principal components containing at most 
        p (percentile) of the variance.
    
    """
    # compute target variance
    total_var = np.sum(var)
    p_var = p * total_var

    # Find the minimum number of principal components that explain at least p percent of the variance
    cumul_var = np.cumsum(var)
    D = np.argmax(cumul_var >= p_var)

    return u[:, :D]



def reconstruct(face_image, mean_face, u):
    """Reconstructs the face image with respect to the first D 
    principal components u.
    
    Args:
        face_image: (M, ) numpy array (M=H*W) of the face.
        mean_face: (M, ) numpy array (M=H*W) mean face.
        u: (M, D) matrix containing D principal components. 
    
    Returns:
        reconstructed_img: (M, ) numpy array of reconstructed face image
    """
    # generating eigen faces: project the centered face image onto the principal components
    eigen_faces = u.T @ (face_image - mean_face)

    # reconstruct the face image using the principal components
    reconstructed_img = mean_face + u @ eigen_faces

    return reconstructed_img


def components_mcq():
    """
    Select the right answer (only one option):
    0: The first principal components mostly correspond to local features, e.g., nose, mouth, eyes
    1: The first principal components predominantly contain global structure, e.g., complete face
    2: The fewer principal components we use, the smaller is the re-projection error
    3: The more principal components we use, the sharper is the image
    4: The variations in the last principal components is perceptually insignificant; these bases can be neglected in the projection
    """
    return 1


def search(Y, x, u, mean_face, top_n):
    """Search for the top most similar images based on a given number of
    components in their PCA decomposition.
    
    Args:
        Y: (N, M) numpy array with N M-dimensional features
        x: (M, ) numpy array image we would like to retrieve
        u: (M, D) numpy arrray, bases vectors. Note, we already assume D has been selected
        mean_face: (M, ) numpy array, mean face as a vector
        top_n: integer, number of n closest images using L2 distance to return
    
    Returns:
        Y: (top_n, M)
    """
    # project the input image and entire dataset onto the selected principal components
    x_eigenfaces = u.T @ (x - mean_face)  # (D, )
    Y_eigenfaces = (Y - mean_face) @ u  # (N, D)
    print("Y_eigenfaces:", Y_eigenfaces.shape)

    # calculate the L2 distance between the input image and all images in the dataset
    dist = np.linalg.norm(Y_eigenfaces - x_eigenfaces, axis=1)  # (N, )

    # the top_n closest images
    top_n_faces = np.argsort(dist)[:top_n]

    return Y[top_n_faces, :]


def interpolate(x1, x2, u, mean_face, n):
    """Interpolates from x1 to x2.
    
    Args:
        x1: (M, ) numpy array, the first image
        x2: (M, ) numpy array, the second image
        u: (M, D) numpy array, bases vectors. Note, we already assume D has been selected.
        mean_face: (M, ) numpy array, mean face as a vector
        n: number of interpolation steps (including x1 and x2)

    Hint: you can use np.linspace to generate N equally-spaced points on a line
    
    Returns:
        Y: (n, M) numpy arrray, interpolated results.
        The first dimension is in the index into corresponding
        image; Y[0] == reconstruct(x1, mean_face, u); Y[-1] == reconstruct(x2, mean_face, u)
    """
    x_inpo = np.linspace(x1, x2, n)
    Y = [reconstruct(x, mean_face, u) for x in x_inpo]

    return np.array(Y)

