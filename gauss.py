import numpy as np
def sol_trsupfil(A,b):
    n = len(A)
    x = np.zeros(n)
    
    x[n-1] = b[n-1]/A[n-1][n-1]
    
    for i in range(n-2,-1,-1):
        # np.any true si existe elem > 0
        if b[i] == 0 and not np.any(x[i+1:]):
            x[i] = 0
        else:
            x[i] = (b[i] - (A[i][i+1:] @ x[i+1:]))/A[i][i] 
    return x

def egaussp(A, b):
    A = A.astype(float)
    b = b.astype(float)
    n = len(A)
    for k in range(n-1):
        # l es index
        l = k + np.argmax(np.abs(A[k:, k]))
        
        if l != k:
            A[[k, l]] = A[[l, k]]
            b[[k,l]] = b[[l,k]]
            
        v = A[k+1:,k] / A[k,k]
        A[k+1:, k:] -=  np.outer(v,A[k, k:])
        b[k+1:] -= v * b[k]
    
    return A, b

def sol_egauss(A, b):
    U, y = egaussp(A, b)
    return sol_trsupfil(U,y)