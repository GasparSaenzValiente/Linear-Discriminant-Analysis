import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from aux.potencias_mauri import autpotenciasinf, autpotencias2, autrayleigh
from aux.plot import graficar_resultados_lda
from aux.gauss import sol_egauss
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# maligno
X_train_m = X_train[y_train == 0]

# benigno
X_train_b = X_train[y_train == 1]

# mu1 mu2
media_m = np.mean(X_train_m, axis=0)
media_b = np.mean(X_train_b, axis=0)

# probs
p_m = len(X_train_m)/len(X_train)
p_b = len(X_train_b)/len(X_train)
print(f"Probabilidad m: {p_m}, b:{p_b}")

# mu
media = np.mean(X_train, axis=0)
print(f"Media: {media}")

# covs
sigmaw_b = X_train_b - media_b
sigmaw_m = X_train_m - media_m

cov_b = sigmaw_b.T @ sigmaw_b
cov_m = sigmaw_m.T @ sigmaw_m

# varianza intra clase
Sw = p_b * cov_b/len(X_train_b) + p_m * cov_m/len(X_train_m)
print(f"sw shape {np.shape(Sw)}")
# varianza entre clases
sigmab_b = media_b - media
sigmab_m = media_m - media

covb_b = np.outer(sigmab_b, sigmab_b)
covb_m = np.outer(sigmab_m, sigmab_m)

Sb = p_b * covb_b + p_m * covb_m
print(f"sb shape {np.shape(Sb)}")

# criterion
#criterion = np.linalg.inv(Sw) @ Sb
criterion = np.zeros(np.shape(Sw))
for i in range(len(criterion)):
    criterion[:, i] = sol_egauss(Sw, Sb[:, i])


# =============================================================================
# COMPARACIÓN DE LOS MÉTODOS DE POTENCIAS
# =============================================================================
# Métodos de aux/potencias_mauri.py; todos tienen la interfaz f(A, q0, eps, m) -> (q, rho).
METODOS = {
    "Potencias norma inf": autpotenciasinf,
    "Potencias norma 2": autpotencias2,
    "Cociente de Rayleigh": autrayleigh,
}


def historial(f, A, q0, eps=1e-10, m_max=500):
    """
    Reconstruye la sucesión de iterados (q_k, rho_k) de un método SIN modificarlo:
    como cada método es determinista, correrlo con m = k devuelve exactamente el
    iterado k. El método "se detuvo" en la primera k a partir de la cual la salida
    ya no cambia (es decir, cuando se cumplió |rho_k - rho_{k-1}| <= eps).
    """
    q_fin, rho_fin = f(A, q0, eps=eps, m=m_max)
    rhos, qs = [], []
    for k in range(1, m_max + 1):
        q_k, rho_k = f(A, q0, eps=eps, m=k)
        rhos.append(rho_k)
        qs.append(q_k)
        if rho_k == rho_fin and np.array_equal(q_k, q_fin):
            break
    return np.array(rhos), qs, q_fin, rho_fin


def tiempo_medio(f, A, q0, eps=1e-10, repeticiones=20):
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        f(A, q0, eps=eps)
        tiempos.append(time.perf_counter() - t0)
    return np.median(tiempos)


def error_angular(q, v):
    """
    Ángulo (en radianes) entre las rectas generadas por q y v (no importa el signo).
    No usamos arccos(|q^T v|): cerca de 1 pierde precisión y el ángulo más chico que
    puede devolver es arccos(1 - eps_maq) ~ 1.5e-8. Usamos atan2(||q_perp||, |q^T v|),
    que es preciso también para ángulos chicos.
    """
    q = q / np.linalg.norm(q)
    v = v / np.linalg.norm(v)
    c = q @ v
    q_perp = q - c * v              # componente de q ortogonal a v
    return np.arctan2(np.linalg.norm(q_perp), np.abs(c))


def comparar_metodos(A, q0, titulo, eps=1e-10):
    """Corre todos los métodos sobre A y devuelve un dict con los resultados."""
    autovals, autovecs = np.linalg.eig(A)
    i_dom = np.argmax(np.abs(autovals))
    lam_ref = np.real(autovals[i_dom])
    v_ref = np.real(autovecs[:, i_dom])
    orden = np.sort(np.abs(autovals))[::-1]
    cociente = orden[1] / orden[0]

    print("\n" + "=" * 92)
    print(f"{titulo}")
    print(f"  lambda_1 (referencia numpy) = {lam_ref:.12g}   |lambda_2 / lambda_1| = {cociente:.3e}")
    print("=" * 92)
    print(f"{'Método':<32}{'iter':>6}{'tiempo [ms]':>13}{'|rho-l1|/|l1|':>16}{'ángulo(q,v1)':>15}{'residuo':>11}")

    resultados = {}
    for nombre, f in METODOS.items():
        rhos, qs, q, rho = historial(f, A, q0, eps=eps)
        t = tiempo_medio(f, A, q0, eps=eps)
        err_rel = np.abs(rhos - lam_ref) / np.abs(lam_ref)
        err_ang = np.array([error_angular(qk, v_ref) for qk in qs])
        residuo = np.linalg.norm(A @ q - rho * q) / np.linalg.norm(q)
        resultados[nombre] = dict(q=q, rho=rho, iter=len(rhos), tiempo=t,
                                  err_rel=err_rel, err_ang=err_ang, residuo=residuo)
        print(f"{nombre:<32}{len(rhos):>6}{1e3 * t:>13.3f}{err_rel[-1]:>16.2e}"
              f"{err_ang[-1]:>15.2e}{residuo:>11.2e}")

    # El más rápido entre los que convergieron al autovalor DOMINANTE
    # (Rayleigh puede converger a otro autovalor según el q0).
    convergidos = {n: r for n, r in resultados.items() if r["err_rel"][-1] < 1e-6}
    if convergidos:
        mas_rapido = min(convergidos, key=lambda n: (convergidos[n]["iter"], convergidos[n]["tiempo"]))
        print(f"\n  -> Converge en menos iteraciones: {mas_rapido} ({convergidos[mas_rapido]['iter']} iter.)")
    no_conv = set(resultados) - set(convergidos)
    for n in no_conv:
        print(f"  -> {n} NO convergió al autovalor dominante (rho = {resultados[n]['rho']:.6g})")
    return resultados, lam_ref, cociente


CARPETA_FIGURAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras")


def guardar_paneles(fig, nombres, carpeta=CARPETA_FIGURAS, dpi=200):
    """
    Guarda cada subplot de `fig` en un .png aparte (sin tocar la figura).
    `nombres` sigue el orden de los subplots. Los ejes extra (p. ej. la barra de
    color de la matriz de confusión) se guardan junto con el subplot más cercano.
    """
    os.makedirs(carpeta, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    principales = fig.axes[:len(nombres)]
    grupos = [[ax] for ax in principales]
    for extra in fig.axes[len(nombres):]:
        c = np.mean(extra.get_position().get_points(), axis=0)
        dist = [np.linalg.norm(c - np.mean(ax.get_position().get_points(), axis=0)) for ax in principales]
        grupos[int(np.argmin(dist))].append(extra)
    for nombre, grupo in zip(nombres, grupos):
        caja = Bbox.union([ax.get_tightbbox(renderer) for ax in grupo])
        caja = caja.transformed(fig.dpi_scale_trans.inverted()).expanded(1.03, 1.03)
        ruta = os.path.join(carpeta, nombre)
        fig.savefig(ruta, bbox_inches=caja, dpi=dpi)
        print(f"  guardado: {ruta}")


def graficar_convergencia(resultados):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Convergencia de los métodos de potencias sobre Sw^{-1} Sb", fontsize=15, fontweight="bold")
    marcadores = ["o", "s", "^"]
    for (nombre, r), mk in zip(resultados.items(), marcadores):
        k = np.arange(1, r["iter"] + 1)
        # piso de 1e-17 para poder graficar ceros en escala logarítmica
        axes[0].semilogy(k, np.maximum(r["err_rel"], 1e-17), marker=mk, ms=4, label=nombre)
        axes[1].semilogy(k, np.maximum(r["err_ang"], 1e-17), marker=mk, ms=4, label=nombre)
    axes[0].set_title("Error relativo del autovalor", fontweight="bold")
    axes[0].set_ylabel(r"$|\rho_k - \lambda_1| / |\lambda_1|$")
    axes[1].set_title("Error angular del autovector", fontweight="bold")
    axes[1].set_ylabel(r"$\angle(q_k, v_1)$ [rad]")
    for ax in axes:
        ax.set_xlabel("iteración k")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    guardar_paneles(fig, ["convergencia_error_autovalor.png", "convergencia_error_angular.png"])
    plt.show()


# Con dos clases, Sb = p_b p_m (mu_b - mu_m)(mu_b - mu_m)^T tiene rango 1, así que
# Sw^{-1} Sb tiene un único autovalor no nulo y |l2/l1| ~ 0: los métodos de potencias
# convergen prácticamente en una iteración.
res_lda, lamda_ref, cociente_lda = comparar_metodos(criterion, media, "Matriz del criterio Sw^{-1} Sb (LDA)")
graficar_convergencia(res_lda)


# =============================================================================
# CLASIFICACIÓN CON CADA MÉTODO
# =============================================================================
print("\nExactitud del clasificador según el autovector de cada método:")
for nombre, r in res_lda.items():
    w_k = r["q"]
    y_pred_k = np.where(np.abs(X_test @ w_k - media_m @ w_k) < np.abs(X_test @ w_k - media_b @ w_k), 0, 1)
    print(f"  {nombre:<32} {np.mean(y_pred_k == y_test) * 100:.2f}%")

# Para el resto usamos el método que convergió en menos iteraciones al autovalor dominante
convergidos = {n: r for n, r in res_lda.items() if r["err_rel"][-1] < 1e-6}
metodo_elegido = min(convergidos, key=lambda n: (convergidos[n]["iter"], convergidos[n]["tiempo"]))
w, lamda = res_lda[metodo_elegido]["q"], res_lda[metodo_elegido]["rho"]
print(f"\nMétodo usado para el LDA: {metodo_elegido}")
print(f"lamda:{lamda}, dir:{w}")

# proyecciones
X_train_projection = X_train  @ w
media_proj = np.mean(X_train_projection)
media_b_projection = media_b @ w
media_m_projection = media_m @ w


# clasificacion
X_test_projection = X_test @ w

distancia_a_maligno = np.abs(X_test_projection - media_m_projection)
distancia_a_benigno = np.abs(X_test_projection - media_b_projection)

y_pred = np.where(distancia_a_maligno < distancia_a_benigno, 0, 1)

exactitud = np.mean(y_pred == y_test) * 100
print(f"Exactitud del clasificador LDA: {exactitud:.2f}%")

# El gráfico final no se modifica: solo interceptamos plt.show para guardar
# cada uno de sus cuatro paneles por separado antes de mostrarlo.
_plt_show = plt.show
def _guardar_y_mostrar(*args, **kwargs):
    guardar_paneles(plt.gcf(), [
        "lda_1_histograma_proyeccion.png",
        "lda_2_espacio_original.png",
        "lda_3_clasificacion_test.png",
        "lda_4_matriz_confusion.png",
    ])
    _plt_show(*args, **kwargs)

plt.show = _guardar_y_mostrar
try:
    graficar_resultados_lda(X_train, y_train, X_test, y_test, w, media_m_projection, media_b_projection, y_pred, data.feature_names)
finally:
    plt.show = _plt_show
