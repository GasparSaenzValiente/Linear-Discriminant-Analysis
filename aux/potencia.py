import numpy as np
def autpotencias2(A, q0, eps = 1e-10, m = 500):
    A = np.array(A) 
    q = q0 / np.linalg.norm(q0)

    l_old = 0
    l_new = np.inf
    c = 0
    while(c < m and abs(l_old - l_new) > eps):
        c += 1
        z = A @ q
        l_old = l_new
        l_new = np.conj(q) @ z
        q = z / np.linalg.norm(z)
    
    return l_new, q