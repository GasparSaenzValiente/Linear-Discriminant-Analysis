"""Comparación de métodos de potencias y selección del autovector para LDA."""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from .potencias_mauri import autpotenciasinf, autpotencias2, autrayleigh


METODOS = {
    "Potencias norma inf": autpotenciasinf,
    "Potencias norma 2": autpotencias2,
    "Cociente de Rayleigh": autrayleigh,
}


def historial(f, A, q0, eps=1e-10, m_max=500):
    """Reconstruye los iterados ejecutando el método con m = k."""
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
    """Ángulo entre las rectas generadas por q y v, sin importar el signo."""
    q = q / np.linalg.norm(q)
    v = v / np.linalg.norm(v)

    c = q @ v
    q_perp = q - c * v

    return np.arctan2(np.linalg.norm(q_perp), np.abs(c))


def comparar_metodos(A, q0, titulo, eps=1e-10):
    """Ejecuta todos los métodos y devuelve resultados y referencias."""
    autovals, autovecs = np.linalg.eig(A)
    i_dom = np.argmax(np.abs(autovals))
    lam_ref = np.real(autovals[i_dom])
    v_ref = np.real(autovecs[:, i_dom])

    orden = np.sort(np.abs(autovals))[::-1]
    cociente = orden[1] / orden[0]

    print("\n" + "=" * 92)
    print(titulo)
    print(
        f"  lambda_1 (referencia numpy) = {lam_ref:.12g}"
        f"   |lambda_2 / lambda_1| = {cociente:.3e}"
    )
    print("=" * 92)
    print(
        f"{'Método':<32}{'iter':>6}{'tiempo [ms]':>13}"
        f"{'|rho-l1|/|l1|':>16}{'ángulo(q,v1)':>15}{'residuo':>11}"
    )

    resultados = {}

    for nombre, f in METODOS.items():
        rhos, qs, q, rho = historial(f, A, q0, eps=eps)
        t = tiempo_medio(f, A, q0, eps=eps)

        err_rel = np.abs(rhos - lam_ref) / np.abs(lam_ref)
        err_ang = np.array([error_angular(qk, v_ref) for qk in qs])
        residuo = np.linalg.norm(A @ q - rho * q) / np.linalg.norm(q)

        resultados[nombre] = dict(
            q=q,
            rho=rho,
            iter=len(rhos),
            tiempo=t,
            err_rel=err_rel,
            err_ang=err_ang,
            residuo=residuo,
        )

        print(
            f"{nombre:<32}{len(rhos):>6}{1e3 * t:>13.3f}"
            f"{err_rel[-1]:>16.2e}{err_ang[-1]:>15.2e}{residuo:>11.2e}"
        )

    # Rayleigh puede converger a otro autovalor según el vector inicial.
    convergidos = {
        nombre: r
        for nombre, r in resultados.items()
        if r["err_rel"][-1] < 1e-6
    }

    if convergidos:
        mas_rapido = min(
            convergidos,
            key=lambda nombre: (
                convergidos[nombre]["iter"],
                convergidos[nombre]["tiempo"],
            ),
        )
        print(
            f"\n  -> Converge en menos iteraciones: {mas_rapido}"
            f" ({convergidos[mas_rapido]['iter']} iter.)"
        )

    for nombre in set(resultados) - set(convergidos):
        print(
            f"  -> {nombre} NO convergió al autovalor dominante"
            f" (rho = {resultados[nombre]['rho']:.6g})"
        )

    return resultados, lam_ref, cociente


# La carpeta figuras queda al mismo nivel que project.py.
CARPETA_FIGURAS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "figuras",
)


def guardar_paneles(fig, nombres, carpeta=CARPETA_FIGURAS, dpi=200):
    """Guarda cada subplot en un PNG, incluyendo sus ejes adicionales."""
    os.makedirs(carpeta, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    principales = fig.axes[:len(nombres)]
    grupos = [[ax] for ax in principales]

    for extra in fig.axes[len(nombres):]:
        centro = np.mean(extra.get_position().get_points(), axis=0)
        distancias = [
            np.linalg.norm(
                centro - np.mean(ax.get_position().get_points(), axis=0)
            )
            for ax in principales
        ]
        grupos[int(np.argmin(distancias))].append(extra)

    for nombre, grupo in zip(nombres, grupos):
        caja = Bbox.union([ax.get_tightbbox(renderer) for ax in grupo])
        caja = caja.transformed(
            fig.dpi_scale_trans.inverted()
        ).expanded(1.03, 1.03)

        ruta = os.path.join(carpeta, nombre)
        fig.savefig(ruta, bbox_inches=caja, dpi=dpi)
        print(f"  guardado: {ruta}")


def graficar_convergencia(resultados):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Convergencia de los métodos de potencias sobre Sw^{-1} Sb",
        fontsize=15,
        fontweight="bold",
    )

    marcadores = ["o", "s", "^"]

    for (nombre, r), marcador in zip(resultados.items(), marcadores):
        k = np.arange(1, r["iter"] + 1)

        # Piso para representar ceros en escala logarítmica.
        axes[0].semilogy(
            k, np.maximum(r["err_rel"], 1e-17),
            marker=marcador, ms=4, label=nombre,
        )
        axes[1].semilogy(
            k, np.maximum(r["err_ang"], 1e-17),
            marker=marcador, ms=4, label=nombre,
        )

    axes[0].set_title("Error relativo del autovalor", fontweight="bold")
    axes[0].set_ylabel(r"$|\rho_k - \lambda_1| / |\lambda_1|$")

    axes[1].set_title("Error angular del autovector", fontweight="bold")
    axes[1].set_ylabel(r"$\angle(q_k, v_1)$ [rad]")

    for ax in axes:
        ax.set_xlabel("iteración k")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    guardar_paneles(
        fig,
        [
            "convergencia_error_autovalor.png",
            "convergencia_error_angular.png",
        ],
    )
    plt.show()


def ejecutar_comparacion(criterion, media, X_test, y_test, media_m, media_b):
    """Compara, grafica y devuelve (w, lamda) del método elegido para LDA."""
    res_lda, _, _ = comparar_metodos(
        criterion,
        media,
        "Matriz del criterio Sw^{-1} Sb (LDA)",
    )
    graficar_convergencia(res_lda)

    print("\nExactitud del clasificador según el autovector de cada método:")

    for nombre, r in res_lda.items():
        w_k = r["q"]
        y_pred_k = np.where(
            np.abs(X_test @ w_k - media_m @ w_k)
            < np.abs(X_test @ w_k - media_b @ w_k),
            0,
            1,
        )
        print(f"  {nombre:<32} {np.mean(y_pred_k == y_test) * 100:.2f}%")

    convergidos = {
        nombre: r
        for nombre, r in res_lda.items()
        if r["err_rel"][-1] < 1e-6
    }

    metodo_elegido = min(
        convergidos,
        key=lambda nombre: (
            convergidos[nombre]["iter"],
            convergidos[nombre]["tiempo"],
        ),
    )

    w = res_lda[metodo_elegido]["q"]
    lamda = res_lda[metodo_elegido]["rho"]

    print(f"\nMétodo usado para el LDA: {metodo_elegido}")
    print(f"lamda:{lamda}, dir:{w}")

    return w, lamda