import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from aux.plot import graficar_resultados_lda
from aux.gauss import sol_egauss
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from aux.comparacion_potencias import ejecutar_comparacion, guardar_paneles

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


# criterion = np.linalg.inv(Sw) @ Sb
criterion = np.zeros(np.shape(Sw))
for i in range(len(criterion)):
    criterion[:, i] = sol_egauss(Sw, Sb[:, i])

# comparaciones
w, lamda = ejecutar_comparacion(
    criterion, media, X_test, y_test, media_m, media_b
)

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


# UMBRAL
# el signo del autovector es arbitrario: lo orientamos para que los malignos
# proyecten más alto, así "s >= t  =>  maligno".
w_or = -w if media_m_projection < media_b_projection else w

s_train = X_train @ w_or
s_test  = X_test  @ w_or
t0 = (media_m @ w_or + media_b @ w_or) / 2   # equivalente a la regla de distancias

def metricas(y_true, y_pred):
    """Maligno (0) es la clase positiva."""
    tp = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 1) & (y_true == 0))
    fp = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 1) & (y_true == 1))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    especificidad = tn / (tn + fp)
    return recall, precision, especificidad, fn, fp

# criterio de recall
objetivo = 0.98
t = t0
for c in np.sort(s_train[y_train == 0])[::-1]:
    if metricas(y_train, np.where(s_train >= c, 0, 1))[0] >= objetivo:
        t = min(c, t0)
        break

y_pred_recall = np.where(s_test >= t, 0, 1)   # prediccion final con el umbral nuevo

for nombre, umbral in [("punto medio (original)", t0), (f"recall >= {objetivo}", t)]:
    y_u = np.where(s_test >= umbral, 0, 1)
    r, p, e, fn, fp = metricas(y_test, y_u)
    print(f"{nombre:<24} t={umbral:.4f}  recall={r:.3f}  precision={p:.3f}  "
          f"especif={e:.3f}  FN={fn}  FP={fp}")


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
    graficar_resultados_lda(X_train, y_train, X_test, y_test, w, media_m_projection, media_b_projection, y_pred_recall, data.feature_names)
finally:
    plt.show = _plt_show