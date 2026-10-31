import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def graficar_resultados_lda(X_train, y_train, X_test, y_test, w, media_m_proj, media_b_proj, y_pred, feature_names=None):
    w_flat = w.flatten()
    
    # Proyecciones escalares (1D)
    X_train_proj = (X_train @ w).flatten()
    X_test_proj = (X_test @ w).flatten()
    
    # Umbral de decisión (punto medio entre las medias proyectadas)
    umbral = 0.5 * (media_m_proj + media_b_proj)
    if isinstance(umbral, np.ndarray):
        umbral = umbral.item()

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('Resultados del Análisis Discriminante Lineal (LDA)', fontsize=16, fontweight='bold')

    # =========================================================================
    # 1. HISTOGRAMA DE SEPARABILIDAD EN EL ESPACIO TRANSFORMADO 
    # =========================================================================
    ax1 = axes[0, 0]
    proj_m_train = X_train_proj[y_train == 0]
    proj_b_train = X_train_proj[y_train == 1]

    ax1.hist(proj_m_train, bins=25, alpha=0.6, color='crimson', label='Maligno (Train)', density=True, edgecolor='black')
    ax1.hist(proj_b_train, bins=25, alpha=0.6, color='royalblue', label='Benigno (Train)', density=True, edgecolor='black')
    
    ax1.axvline(media_m_proj, color='darkred', linestyle='--', linewidth=2, label=f'Media Maligno ({media_m_proj:.2f})')
    ax1.axvline(media_b_proj, color='darkblue', linestyle='--', linewidth=2, label=f'Media Benigno ({media_b_proj:.2f})')
    ax1.axvline(umbral, color='black', linestyle='-', linewidth=2.5, label=f'Frontera Decisión ({umbral:.2f})')
    
    ax1.set_title('1. Distribución de Clases en el Eje Proyectado w (Train)', fontweight='bold')
    ax1.set_xlabel('Coordenada en el eje discriminante (X @ w)')
    ax1.set_ylabel('Densidad')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # 2. ESPACIO ORIGINAL (2D) VS EJE DEL AUTOVECTOR
    # =========================================================================
    ax2 = axes[0, 1]
    top2_idx = np.argsort(np.abs(w_flat))[-2:]
    idx1, idx2 = top2_idx[0], top2_idx[1]
    
    nombre_x = feature_names[idx1] if feature_names is not None else f'Característica {idx1}'
    nombre_y = feature_names[idx2] if feature_names is not None else f'Característica {idx2}'

    ax2.scatter(X_train[y_train == 0, idx1], X_train[y_train == 0, idx2], 
                color='crimson', alpha=0.5, label='Maligno (Original)', edgecolors='k', s=40)
    ax2.scatter(X_train[y_train == 1, idx1], X_train[y_train == 1, idx2], 
                color='royalblue', alpha=0.5, label='Benigno (Original)', edgecolors='k', s=40)

    media_global_2d = np.mean(X_train[:, [idx1, idx2]], axis=0)
    w_2d = w_flat[[idx1, idx2]]
    w_2d_norm = w_2d / (np.linalg.norm(w_2d) + 1e-12)
    
    escala = np.std(X_train[:, [idx1, idx2]]) * 3
    p_inicio = media_global_2d - escala * w_2d_norm
    p_fin = media_global_2d + escala * w_2d_norm
    
    ax2.plot([p_inicio[0], p_fin[0]], [p_inicio[1], p_fin[1]], 
             color='black', linewidth=2.5, label='Dirección Autovector w')
    ax2.scatter(media_global_2d[0], media_global_2d[1], color='gold', edgecolor='black', s=100, zorder=5, label='Media Global')

    ax2.set_title('2. Espacio Original (Top 2 Variables) y Dirección de w', fontweight='bold')
    ax2.set_xlabel(nombre_x)
    ax2.set_ylabel(nombre_y)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # 3. EVALUACIÓN EN TEST: PROYECCIÓN 1D Y ERRORES DE CLASIFICACIÓN
    # =========================================================================
    ax3 = axes[1, 0]
    aciertos = (y_pred.flatten() == y_test.flatten())
    errores = ~aciertos

    np.random.seed(42)
    jitter_m = np.random.normal(0.15, 0.03, size=np.sum((y_test == 0) & aciertos))
    jitter_b = np.random.normal(-0.15, 0.03, size=np.sum((y_test == 1) & aciertos))
    jitter_err = np.zeros(np.sum(errores))

    ax3.scatter(X_test_proj[(y_test == 0) & aciertos], jitter_m, 
                color='crimson', alpha=0.7, label='Maligno Correcto', s=45)
    ax3.scatter(X_test_proj[(y_test == 1) & aciertos], jitter_b, 
                color='royalblue', alpha=0.7, label='Benigno Correcto', s=45)
    
    if np.sum(errores) > 0:
        ax3.scatter(X_test_proj[errores], jitter_err, 
                    color='black', marker='x', s=100, linewidths=2.5, label=f'Errores ({np.sum(errores)})', zorder=5)

    ax3.axvline(umbral, color='black', linestyle='-', linewidth=2, label='Frontera de Decisión')
    ax3.set_ylim(-0.4, 0.4)
    ax3.set_yticks([-0.15, 0.0, 0.15])
    ax3.set_yticklabels(['Benigno', 'Errores', 'Maligno'])
    ax3.set_title('3. Clasificación del Conjunto de Prueba (X_test)', fontweight='bold')
    ax3.set_xlabel('Coordenada proyectada (X_test @ w)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # =========================================================================
    # 4. MATRIZ DE CONFUSIÓN EN TEST
    # =========================================================================
    ax4 = axes[1, 1]
    cm = confusion_matrix(y_test, y_pred)
    im = ax4.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    clases = ['Maligno (0)', 'Benigno (1)']
    ax4.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=clases, yticklabels=clases,
            title='4. Matriz de Confusión (Datos de Prueba)',
            ylabel='Clase Real (y_test)',
            xlabel='Clase Predicha (y_pred)')
    ax4.title.set_fontweight('bold')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax4.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()