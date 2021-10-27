#!/usr/bin/env python3

import sys
import numpy as np
from math import pi, sqrt
import cvxopt
from cvxopt import matrix, solvers

def gen_eq_bloch_states(n, C=3.6, return_angles=True):
    '''
    Generates equally spaced points on the bloch sphere through a spiral. 
    See here: https://www.intlpress.com/site/pub/files/_fulltext/journals/mrl/1994/0001/0006/MRL-1994-0001-0006-a003.pdf

    ==========
    Parameters
    ----------
    n (type=int): the number of points to be generated evenly on the sphere
    C (type=float, default=3.6): constant that makes sure that succesive points 
                                 on S^2 will be the same Euclidean distance apart
    return_angles (type=boolean, default=True): whether to also return the angles 
                                                associated with each state
    ==========
    
    ======
    Return
    ------
    (type=numpy.array): an array for n evenly spaced coordinates on the block sphere, 
                         with their associated angles if needed
    ======
    '''
    states = np.zeros((n, 2), dtype=complex)
    h = np.zeros(n)
    theta = np.zeros(n)
    phi = np.zeros(n)
    
    # initialize
    h[0] = -1
    h[n-1] = -1 + 2*(n-1)/(n-1)
    theta[0] = np.arccos(h[0])
    theta[n-1] = np.arccos(h[n-1]) 
    phi[0] = 0
    phi[n-1] = 0
    states[0] = [np.cos(theta[0]/2), np.sin(theta[0]/2)*np.exp(1j*phi[0])]
    states[n-1] = [np.cos(theta[n-1]/2), np.sin(theta[n-1]/2)*np.exp(1j*phi[n-1])]
    
    # the rest of the states
    for k in range(1, n-1):
        h[k] = -1 + 2*(k)/(n-1)
        theta[k] = np.arccos(h[k])
        phi[k] = np.mod(phi[k] + C/(sqrt(n)*sqrt(1 - h[k]**2)), 2*pi)
        states[k] = [np.cos(theta[k]/2), np.sin(theta[k]/2)*np.exp(1j*phi[k])]
    
    if return_angles==True:
        return [states, theta, phi]
    else:
        return states


def vec(M):
    '''
    Converts an n x m matrix into a column vector, where the first n entries
    corresponds to the first column of the matrix, the next n entries to the 
    second column, and so on.
    
    https://stackoverflow.com/a/25248378

    ==========
    Parameters
    ----------
    M (type=numpy.array): the matrix to use for the column vector
    ==========
    
    ======
    Return
    ------
    (type=numpy.array): the column vector
    ======
    '''
    return M.reshape((-1, 1), order="F")

def mat(v, n, m, dty=float):
    '''
    Converts a vector of length n*m to an n x m matrix where the first n entries
    corresponds to the first column of the matrix, the next n entries to the 
    second column, and so on.
    
    ==========
    Parameters
    ----------
    v (type=numpy.array): the array to use for the matrix
    n (type=int): number of rows for the matrix
    m (type=int): number of columns for the matrix
    dty (type=type): type of the matrix
    ==========
    
    ======
    Return
    ------
    (type=numpy.array): the matrix
    ======
    '''
    M = np.zeros((n, m), dtype=dty)
    for i in range(n):
        for j in range(m):
            M[i][j] = v[n*j + i]
    return M

def s_min(E, W, F, n, m):
    '''
    Function that minimizes the estimated state space, S, with a fixed effect space
    according to the following quadratic program:
    
    minimize_{S} vec(S)^T @ (E x I_n) @ W (E^T x I_n) vec(S) - 2 vec(S)^T @ (E x I_n) @ W @ vec(F)
    
    subject to 0 <= (E^T x I_n) @ vec(S) <= 1 (element wise inequality)
    
    ==========
    Parameters
    ----------
    E (type=numpy.array): fixed effect space matrix; dim = k x m
    W (type=numpy.array): matrix that encodes the uncertainties along the diagonal 
                          for each preparation/measurement pair; dim = n*m x n*m
    F (type=numpy.array): data matrix; dim = n x m
    n (type=int): number of preparations
    m (type=int): number of measurements
    ==========
    
    ======
    Return
    ------
    (type=numpy.array): the solution to the quadratic program
    ======
    '''
    I_n = np.identity(n, dtype=float)
    P = 2 * (np.kron(E, I_n)) @ W @ (np.kron(np.transpose(E), I_n))
    q = -2 * np.kron(E, I_n) @ W @ vec(F)
    G_0 = -np.kron(np.transpose(E), I_n)
    G_1 = np.kron(np.transpose(E), I_n)
    h_0 = (np.zeros(n*m))
    h_1 = (np.ones(n*m))
    
    P = matrix(P)
    q = matrix(q)
    G = matrix(np.concatenate([G_0, G_1]))
    h = matrix(np.concatenate([h_0, h_1]))
    
    return solvers.qp(P, q, G, h, kktsolver="chol")

def e_min(S, W, F, n, m):
    '''
    Function that minimizes the estimated effect space, E, with a fixed state space
    according to the following quadratic program:
    
    minimize_{E} vec(E)^T @ (I_m x S)^T @ W (I_m x S) vec(S) - 2 vec(S)^T @ (I_m x S)^T @ W @ vec(F)
    
    subject to 0 <= (I_m x S) @ vec(E) <= 1 (element wise inequality)
    
    ==========
    Parameters
    ----------
    S (type=numpy.array): fixed state space matrix; dim = k x m
    W (type=numpy.array): matrix that encodes the uncertainties along the diagonal 
                          for each preparation/measurement pair; dim = n*m x n*m
    F (type=numpy.array): data matrix; dim = n x m
    n (type=int): number of preparations
    m (type=int): number of measurements
    ==========
    
    ======
    Return
    ------
    (type=numpy.array): the solution to the quadratic program
    ======
    '''
    I_m = np.identity(m, dtype=float)
    P = 2 * (np.transpose(np.kron(I_m, S))) @ W @ (np.kron(I_m, S))
    q = -2 * np.transpose(np.kron(I_m, S)) @ W @ vec(F)
    G_0 = -np.kron(I_m, S)
    G_1 = np.kron(I_m, S)
    h_0 = (np.zeros(n*m))
    h_1 = (np.ones(n*m))
    
    P = matrix(P)
    q = matrix(q)
    G = matrix(np.concatenate([G_0, G_1]))
    h = matrix(np.concatenate([h_0, h_1]))
    
    return solvers.qp(P, q, G, h, kktsolver="chol")

def chi_squared(S, E, F, W, n, m):
    '''
    Calculates the weighted chi^2 value.

    ==========
    Parameters
    ----------
    S (type=numpy.array): state space matrix; dim = n x k
    E (type=numpy.array): effect space matrix; dim = k x m
    F (type=numpy.array): data matrix; dim = n x m
    W (type=numpy.array): matrix that encodes the uncertainties along the diagonal 
                          for each preparation/measurement pair; dim = n*m x n*m
    n (type=int): number of preparations
    m (type=int): number of measurements
    ==========
    
    ======
    Return
    ------
    (type=float): the chi^2 value
    ======
    '''
    sum = 0
    w_index = 0
    D = S @ E
    for i in range(n):
        for j in range(m):
            sum += ((F[i][j] - D[i][j]))**2 * W[w_index][w_index]
            w_index += 1
    return sum

def bfp(k, E_0, F, W, n, m, max_iterations=5000, convergence_threshold=10E-6):
    '''
    Finds the low-rank matrix that best fits the data matrix of frequencies.  

    ==========
    Parameters
    ----------
    k (type=int): rank of best-fit matrix
    E_0 (type=numpy.array): initial estimate of effect space matrix; dim = k x m
    F (type=numpy.array): data matrix; dim = n x m
    W (type=numpy.array): matrix that encodes the uncertainties along the diagonal 
                          for each preparation/measurement pair; dim = n*m x n*m
    n (type=int): number of preparations
    m (type=int): number of measurements
    max_iterations (type=int, default=5000): the max number of iterations to go through
                                             for the optimization
    convergence_threshold (type=float, default=10E-16): the convergence threshold for 
                                                        the optimization
    ==========
    
    ======
    Return
    ------
    (type=numpy.array): an array that contains the estimated state space and effect space matrices
    ======
    '''
    S = np.zeros((n, k), dtype=float)
    E = E_0
    chi_squared_prev = 0
    chi_squared_curr = 0
    iteration = 1
    while (True):
        chi_squared_prev = chi_squared_curr
        S = mat(s_min(E, W, F, n, m)['x'], n, k)
        E = mat(e_min(S, W, F, n, m)['x'], k, m)
        chi_squared_curr = chi_squared(S, E, F, W, n, m)
        if (iteration == max_iterations or (chi_squared_curr -  chi_squared_prev) < convergence_threshold):
            break
        iteration += 1
    return [S, E]

####################################################
#==================================================#
####################################################

def aic(k, chi_squared_k, m, n):
    r_k = k*(m + n - k)
    return chi_squared_k + r_k

def is_pos_semi_def(A, tol=1e-8):
    E = np.linalg.eigvalsh(A)
    return np.all(E > -tol)

def S_min(S, E, F, W, m):
    I_m = np.identity(m)
    A_1 = np.transpose(vec(S)) @ (np.kron(E, I_m)) @ W @ (np.kron(np.transpose(E), I_m)) @ vec(S)
    A_2 = 2 * np.transpose(vec(S)) @ (np.kron(E, I_m)) @ W @ vec(F)
    A = A_1 - A_2
    return A[0][0]

def E_min(E, S, F, W, n):
    I_n = np.identity(n)
    A_1 = np.transpose(vec(E)) @ np.transpose(np.kron(I_n, S)) @ W @ (np.kron(I_n, S)) @ vec(E)
    A_2 = 2 * np.transpose(vec(E)) @ np.transpose(np.kron(I_n, S)) @ W @ vec(F)
    A = A_1 - A_2
    return A[0][0]

def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))

def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -numpy.vstack([A, G]).T
        qp_b = -numpy.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

def wrla_1(U, V, A, W):
    A_tilda = U @ V
    summation = 0
    w = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            summation += W[w][w] * ((A[i][j] - A_tilda[i][j])**2)
    return summation

def wrla_2(V, U, A, W):
    A_tilda = U @ V
    summation = 0
    w = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            summation += W[w][w] * ((A[i][j] - A_tilda[i][j])**2)
    return summation