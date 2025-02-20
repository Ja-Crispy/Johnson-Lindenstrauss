import numpy as np
from scipy.sparse import random as sparse_random
from scipy.sparse import csr_matrix
import csv

def generate_sparse_matrix(k, N):
    density = 3/np.sqrt(N)
    """Generate a sparse k×N matrix with specified density."""
    S = sparse_random(k, N, density=density, data_rvs=lambda x: np.ones(x))
    return S.toarray() * np.sqrt(N/(3*k))

def hadamard_transform(v, m):
    """Apply Hadamard transform to vector v using Sylvester construction."""
    n = len(v)
    if n == 1:
        return v
    
    # Split vector into even and odd indices
    v_even = v[::2]
    v_odd = v[1::2]
    
    # Recursive calls
    h_even = hadamard_transform(v_even, m-1)
    h_odd = hadamard_transform(v_odd, m-1)
    
    # Combine results
    return np.concatenate([h_even + h_odd, h_even - h_odd]) / np.sqrt(2)

def compute_projection(m, b):
    """
    Compute projection matrix P using FJLT.
    Parameters:
        m: power of 2 for N (N = 2^m)
        b: power of 2 for k (k = 2^b)
    Returns:
        P: Projection matrix of shape (N, k)
    """
    N = 2**m
    k = 2**b
    
    # Generate sparse matrix S
    S = generate_sparse_matrix(k, N)
    
    # Generate random diagonal matrix elements
    D = np.random.choice([-1, 1], size=N)
    
    # Initialize projection matrix
    P = np.zeros((N, k))
    
    # Compute P = HD one row at a time
    for i in range(N):
        # Create standard basis vector
        e_i = np.zeros(N)
        e_i[i] = 1
        
        # Apply D
        v = e_i * D
        
        # Apply H using Hadamard transform
        v = hadamard_transform(v, m)
        
        # Apply S
        P[i] = np.dot(v, S.T)
    
    # Normalize rows
    row_norms = np.sqrt(np.sum(P**2, axis=1))
    P = P / row_norms[:, np.newaxis]
    
    return P

def find_max_dot_product(P):
    """Find the maximum absolute dot product between any pair of rows in P."""
    N = P.shape[0]
    max_dot = 0
    
    for i in range(N):
        # Compute dot products with all subsequent rows
        dots = np.abs(np.dot(P[i], P[i+1:].T))
        if len(dots) > 0:
            max_dot = max(max_dot, np.max(dots))
    
    return max_dot

def save_matrix(P, filename):
    """Save matrix P to CSV file."""
    np.savetxt(filename, P, delimiter=',')

def main(m, b):
    """Main function to run the JL transform and analysis."""
    print(f"Computing projection for N=2^{m}={2**m}, k=2^{b}={2**b}")
    
    # Compute projection
    P = compute_projection(m, b)
    
    # Save matrix
    filename = f"projection_m{m}_b{b}.csv"
    save_matrix(P, filename)
    print(f"Saved projection matrix to {filename}")
    
    # Find maximum dot product
    max_dot = find_max_dot_product(P)
    print(f"Maximum absolute dot product between any pair of vectors: {max_dot:.6f}")
    print(f"Corresponding to worst-case distortion factor: {2/(2-2*max_dot):.6f}")
    C = (2**b) / np.log(2**m) * max_dot**2
    print ("C value: "+str(C))

if __name__ == "__main__":
    # Example usage
    m = 10  # N = 1024
    b = 7   # k = 128
    main(m, b)
