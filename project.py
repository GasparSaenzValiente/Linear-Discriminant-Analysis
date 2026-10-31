import numpy as np
from potencia import autpotencias2
from plot import graficar_resultados_lda
from gauss import sol_egauss
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
Sw = p_b * cov_b + p_m * cov_m
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

lamda, w = autpotencias2(criterion, media)
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

graficar_resultados_lda(X_train, y_train, X_test, y_test, w, media_proj, media_b_projection, y_pred)