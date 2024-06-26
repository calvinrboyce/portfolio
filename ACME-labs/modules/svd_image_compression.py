"""Volume 1: The SVD and Image Compression."""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from imageio.v2 import imread


def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    # get the normalized eigenvalues of A^TA
    lamda, V = np.linalg.eig(A.conj().T@A)
    V = V/np.linalg.norm(V,axis=0)
    
    # sort the singular values
    sigma = np.sqrt(lamda)
    idx = sigma.argsort()[::-1]
    sigma = sigma[idx]
    V = V[:,idx]
    
    # find the singular values greater than tol
    r = sum(sigma>tol)
    sigma1 = sigma[:r]
    V1 = V[:,:r]
    
    # compute U
    U1 = A@V1/sigma1
    
    return U1,sigma1,V1.conj().T


def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    # strip the SVD
    U,E,VH = compact_svd(A)
    
    #value error
    if s>len(E):
        raise ValueError('s needs to be less than the rank of A')
    
    # clip the U Sigma and VH matrices
    U = U[:,:s]
    E = E[:s]
    VH = VH[:s]
    
    return U@np.diag(E)@VH, U.size+E.size+VH.size



def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    #import image
    image = imread(filename)/255
    hyperparams = dict()
    if len(image.shape) == 2:                           #gray
        hyperparams['cmap'] = 'gray'
        compressed, data = svd_approx(image,s)
    else:                                               #color
        R = image[:,:,0]
        G = image[:,:,1]
        B = image[:,:,2]
        
        # do SVD on each color
        R, Rdata = svd_approx(R,s)
        G, Gdata = svd_approx(G,s)
        B, Bdata = svd_approx(B,s)
        
        # clip and recombine
        compressed = np.clip(np.dstack((R,G,B)),0,1)
        data = Rdata + Gdata + Bdata
        
    #plot
    plt.subplot(121)
    plt.title('Original')
    plt.axis('off')
    plt.imshow(image,**hyperparams)
    
    plt.subplot(122)
    plt.title(f'Compressed: {s} dimensions')
    plt.axis('off')
    plt.imshow(compressed,**hyperparams)
    
    plt.tight_layout()
    plt.show()
    print(f'We saved {image.size-data} entries by compressing to {s} dimensions')
