def dirichlet_zero(matrix, dirichlet_zero_dofs):
    """Enforces dirichlet zero B.C.'s by zeroing-out rows and columns 
    associated with dirichlet dofs, and putting 1 on the diagonal there."""
    
    import numpy as np
    import scipy.sparse

    N = matrix.shape[1]

    chi_interior = np.ones(N)
    chi_interior[dirichlet_zero_dofs] = 0.0
    I_interior = scipy.sparse.spdiags(chi_interior, [0], N, N).tocsc()

    chi_boundary = np.zeros(N)
    chi_boundary[dirichlet_zero_dofs] = 1.0
    I_boundary = scipy.sparse.spdiags(chi_boundary, [0], N, N).tocsc()

    matrix_modified = I_interior * matrix * I_interior + I_boundary
    return matrix_modified