# Linear-Discriminant-Analysis
## Clasificación de Tumores de Mama mediante Análisis Discriminante Lineal (LDA)

Este proyecto implementa desde cero el método de **Análisis Discriminante Lineal (LDA)** para clasificar tumores de mama en benignos y malignos. Utilizando el conjunto de datos `breast_cancer` de `scikit-learn`, el modelo busca proyectar datos multidimensionales en un único eje que maximice la separación entre las clases.

**Autores:** Mauricio Nicolas Perez Acevedo y Gaspar Saenz Valiente.

## 📌 Descripción General
A diferencia de utilizar librerías preconstruidas para el núcleo del algoritmo, este proyecto resuelve el problema de clasificación implementando los fundamentos matemáticos de LDA. Para encontrar el eje de máxima separabilidad (maximizando el Criterio de Fisher), el proyecto elude la inversión directa de matrices mediante métodos numéricos propios:
1. **Eliminación Gaussiana con pivoteo parcial** para estructurar el problema generalizado de autovalores.
2. **Método de las Potencias** para encontrar el autovector dominante (la dirección óptima de proyección).

Finalmente, los nuevos datos son clasificados utilizando una regla de mínima distancia euclidiana hacia los centroides proyectados.

## 🗂️ Estructura del Proyecto

El repositorio está modularizado en los siguientes archivos:

* **`project.py`**: Es el script principal. Carga el *dataset*, divide los datos (entrenamiento/prueba), calcula las matrices de covarianza intra-clase ($S_w$) y entre clases ($S_b$), y evalúa la exactitud del modelo.
* **`gauss.py`**: Contiene la implementación numérica de la Eliminación Gaussiana (`egauusp` y `sol_egauss`) para resolver sistemas de ecuaciones lineales.
* **`potencia.py`**: Implementa el algoritmo iterativo del Método de las Potencias (`autpotencias2`) para extraer el autovalor y autovector principal.
* **`plot.py`**: Módulo gráfico que utiliza `matplotlib` para generar histogramas de separabilidad 1D, visualizaciones 2D del autovector y la matriz de confusión del conjunto de prueba.
* **`LDA_informe.pdf`** / **`LDA_informe.tex`**: Documento final y código fuente en LaTeX con el desarrollo teórico, demostraciones matemáticas y análisis riguroso del método aplicado.

## 🚀 Cómo ejecutar el proyecto

El proyecto incluye scripts automatizados para crear un entorno virtual, instalar las dependencias necesarias (`numpy`, `scikit-learn`, `matplotlib`) y ejecutar el análisis al instante. 

### En Linux / macOS
```bash
chmod +x setup.sh
./setup.sh
```

> **Nota:** El script verificará automáticamente si el entorno virtual ya existe. Si es la primera vez, descargará las librerías. En las siguientes ejecuciones, simplemente abrirá los resultados y los gráficos al instante.

## 📊 Resultados Esperados
Al ejecutar el proyecto, veras la exactitud final del modelo. Además, se abrirá una ventana con 4 gráficos fundamentales:
1. **Histograma de Separabilidad:** Densidad de los tumores benignos y malignos proyectados sobre el nuevo eje discriminante.
2. **Espacio Original 2D:** Visualización de las 2 características principales originales frente a la dirección del autovector calculado.
3. **Clasificación en Datos de Prueba:** Distribución de aciertos y errores respecto a la frontera de decisión en el espacio transformado.
4. **Matriz de Confusión:** Cantidad exacta de verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos en el set de prueba.
