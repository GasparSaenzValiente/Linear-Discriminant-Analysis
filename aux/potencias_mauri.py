import numpy as np
import sys
sys.path.append("..")

def sol_trisup_fil(A,b):
    n = len(b)
    x = np.zeros(n)
    k = -1 #Para que no se rompa si b = 0

    for i in reversed(range(n)):
        if b[i]!=0:
            k = i
            break

    for i in range(k, -1, -1):
        if A[i,i] == 0:
            print("La matriz tiene ceros en la diagonal")
            break
        else:
            x[i] = (b[i] - A[i,i+1:] @ x[i+1:])/A[i,i]

    return x

def egaussp_2(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray,np.ndarray] :
    b = b.astype(float)
    A = A.astype(float)

    m,n = A.shape
    for k in range(min(m-1,n)):
        l = np.argmax(np.abs(A[k:,k])) + k #El argmax te da la posicion del maximo, pero desde la posicion en la que empieza(En este caso es k), el valor absoluto tambien es necesario
        if l != k:
            A[[k,l]] = A[[l,k]]
            b[[k,l]] = b[[l,k]]
        assert A[k,k]!= 0, f"En la posicion {k,k} toda la columna desde k para abajo ya es cero"
        v = A[k+1:,k]/A[k,k]
        A[k+1:,k:] -= np.outer(v,A[k,k:])
        A[k+1:,k] = 0.0 #(Esto sería para garantizar que sea triangular superior, por la arimetica del punto flotante)
        b[k+1:] -= v * b[k] 

    U = A
    y = b

    return U, y

def sol_egauss(A,b):
    U, y = egaussp_2(A,b)
    x = sol_trisup_fil(U,y)

    return x


def autpotenciasinf(A: np.ndarray, q0: np.ndarray, eps: float = 1e-10, m: int = 500):
    """
    Algoritmo 23: método de las potencias en norma infinito.

    Parametros:
    ----------------------------------------------
    Entradas:
    A, matriz n x n, np.ndarray
    q0, vector inicial de R^n, np.ndarray
    eps, tolerancia, float
    m, cantidad máxima de iteraciones, int

    Salidas:
    q, autovector aproximado (con ||q||_inf = 1), np.ndarray
    rho, autovalor aproximado (dominante), float
    """
    A = np.asarray(A, dtype=float)
    q0 = np.asarray(q0, dtype=float) 
    q_hat = A @ q0      # Paso 0
    rho_hat = np.inf
    for k in range(m):                            # Paso 1
        j = np.argmax(np.abs(q_hat))              # |q_hat_j| = ||q_hat||_inf
        q = q_hat / q_hat[j]
        q_hat = A @ q
        rho = q_hat[j]
        if np.abs(rho - rho_hat) <= eps:
            break
        rho_hat = rho
    return q, rho                                 # Paso 2


def autpotencias2(A: np.ndarray, q0: np.ndarray, eps: float = 1e-10, m: int = 500):
    """
    Algoritmo 24: método de las potencias en norma 2.

    Parametros:
    ----------------------------------------------
    Entradas:
    A, matriz n x n, np.ndarray
    q0, vector inicial de R^n, np.ndarray
    eps, tolerancia, float
    m, cantidad máxima de iteraciones, int

    Salidas:
    q, autovector aproximado (con ||q||_2 = 1), np.ndarray
    rho, autovalor aproximado (dominante), float
    """
    A = np.asarray(A, dtype=float)
    q0 = np.asarray(q0, dtype=float)
    q_hat = A @ q0                                # Paso 0
    rho_hat = (q0.T @ q_hat) / (q0.T @ q0)
    for k in range(m):                            # Paso 1
        q = q_hat / np.linalg.norm(q_hat)
        q_hat = A @ q
        rho = q.T @ q_hat                           # cociente de Rayleigh q^T A q
        if np.abs(rho - rho_hat) <= eps:
            break
        rho_hat = rho
    return q, rho                                 # Paso 2


def autrayleigh(A: np.ndarray, q0: np.ndarray, eps: float = 1e-10, m: int = 500):
    """
    Algoritmo 25: iteración del cociente de Rayleigh.
    Converge al autovalor más cercano a rho_0 = q0^T A q0 / ||q0||^2
    (no necesariamente el dominante).

    Parametros:
    ----------------------------------------------
    Entradas:
    A, matriz n x n, np.ndarray
    q0, vector inicial de R^n, np.ndarray
    eps, tolerancia, float
    m, cantidad máxima de iteraciones, int

    Salidas:
    q, autovector aproximado (con ||q||_2 = 1), np.ndarray
    rho, autovalor aproximado, float
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    q = np.asarray(q0, dtype=float)
    q = q / np.linalg.norm(q)                     # Paso 0
    q_hat = q.copy()
    rho = q.T @ A @ q
    theta = 0.0
    for k in range(m):                            # Paso 1
        try:
            z = sol_egauss(A - rho * np.eye(n), q_hat)   # (A - rho I) z = q_hat
        except AssertionError:
            # pivote nulo en la eliminación: A - rho I singular, rho ya es autovalor
            theta = 0.0
            break
        sigma = np.linalg.norm(z, ord=2)
        if sigma == 0 or not np.isfinite(sigma):
            # cero en la diagonal de U (sol_trisup_fil corta): rho ya es autovalor
            theta = 0.0
            break
        q = z / sigma
        theta = (q.T @ q_hat) / sigma
        if np.abs(theta) <= eps:
            break
        q_hat = q
        rho = rho + theta
    return q, rho + theta                         # Paso 2

