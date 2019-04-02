import numpy as np
from numpy import ndarray as Tensor


def Diagonal_Matrix(input_vector: Tensor, spec: int) -> Tensor:
    return np.multiply(np.identity(spec), input_vector)


def GLV_Model(solution_point: Tensor, A_m: Tensor, R_vec: Tensor) -> Tensor:

    n = A_m.shape[0]
    A = Diagonal_Matrix(solution_point.reshape(1, n), n)

    solution_point = solution_point.reshape(n,1)
    R_vec = R_vec.reshape(n,1)
    
    B = np.matmul(A_m, solution_point)+R_vec

    glv_vec = np.matmul(A, B)

    # We are using norm |v|_1
    return np.linalg.norm(glv_vec, ord=1) / glv_vec.shape[0]
