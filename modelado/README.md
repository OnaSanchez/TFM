# Paso 3 · Preprocesamiento, validación cruzada anidada y comparación de modelos

`02_modelado.ipynb` (bloques 4 a 8 del flujo de análisis, tarea T3). Usa los resultados de los pasos 1 y 2 (`qc_eda/` y `correspondencia_genes/`) y las funciones de `tfm/modelos.py` y `tfm/metricas.py`. Tarda unos minutos con 12 núcleos.

| Apartado del notebook | Bloque | Qué hace |
|---|---|---|
| 0 | — | Entorno, versiones de las librerías y semilla |
| 1 | — | Carga de los datos y de la correspondencia de genes |
| 2 | 4 | Particiones de la validación cruzada (CV) anidada: 5 externas y 3 internas |
| 3 | 5 | Preprocesamiento: genes sin variación y escalado |
| 4 | 6 | Selección de variables: filtro ANOVA de *k* genes |
| 5 | 7 | Ajuste de hiperparámetros y entrenamiento de los cuatro modelos |
| 6 | 8 | Evaluación externa: métricas, intervalos de confianza, coste y análisis de errores |

La explicabilidad, la elección del modelo final y la exportación para la aplicación (bloques 9 a 11) se harán en la siguiente fase.

## Resultados

Tablas en `resultados/`, en el orden en que las genera el notebook:

| Fichero | Contenido | Apartado |
|---|---|---|
| `particiones_externas.csv` | Partición de prueba (0 a 4) de cada muestra | 2 |
| `predicciones_oof.csv` | Predicción de cada muestra con cada modelo, hecha por un modelo que no la vio al entrenar, y puntuación de cada clase | 5 |
| `hiperparametros_por_particion.csv` | Número de genes e hiperparámetros elegidos en cada partición, F1 macro interno y tiempo | 5 |
| `metricas_globales_ic95.csv` | F1 macro, exactitud equilibrada, exactitud, precisión y sensibilidad, con su IC 95 % | 6.1 |
| `exactitud_ic_exacto.csv` | Errores, exactitud e intervalo exacto de Clopper-Pearson | 6.1 |
| `metricas_por_particion.csv` | Las mismas métricas en cada partición externa | 6.1 |
| `metricas_por_clase.csv` | Precisión, sensibilidad y F1 de cada tipo tumoral | 6.1 |
| `coste_y_genes.csv` | Tiempo medio y número de genes de cada modelo | 6.2 |
| `coordenadas_pca_umap.csv` | Coordenadas de cada muestra en la PCA y el UMAP (las usa la aplicación) | 6.3 |
| `analisis_errores.csv` | Para cada muestra: si es atípica, su grupo en el UMAP y la predicción de cada modelo | 6.3 |
| `marcadores_grupos_umap.csv` | Expresión de marcadores conocidos en el grupo pequeño de BRCA y en las muestras LUAD mal clasificadas | 6.3 |

Figuras en `figuras/`: `fig6_matrices_confusion.png` (apartado 6.1) y `fig7_umap_grupos.png` (apartado 6.3).
