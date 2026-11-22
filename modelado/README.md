# Paso 3 · Modelado, evaluación, explicabilidad y exportación

`02_modelado.ipynb` (bloques 4 a 12 del flujo de análisis, tareas T3 y T4). Usa los resultados de los pasos 1 y 2 (`qc_eda/` y `correspondencia_genes/`) y las funciones de `tfm/modelos.py`, `tfm/metricas.py` y `tfm/genes.py`. Tarda unos 30 minutos con 12 núcleos (la mayor parte, la importancia por permutación del RF). La primera vez descarga la anotación de cromosomas de NCBI y los datos clínicos básicos de TCGA (Firehose) en `Dataset/`.

| Apartado del notebook | Bloque | Qué hace |
|---|---|---|
| 0 | — | Entorno, versiones de las librerías y semilla |
| 1 | — | Carga de los datos y de la correspondencia de genes |
| 2 | 4 | Particiones de la validación cruzada (CV) anidada: 5 externas y 3 internas |
| 3 | 5 | Preprocesamiento: genes sin variación y escalado |
| 4 | 6 | Selección de variables: filtro ANOVA de *k* genes |
| 5 | 7 | Ajuste de hiperparámetros y entrenamiento de los cuatro modelos (incluye la importancia por permutación en cada partición) |
| 6 | 8 | Evaluación externa: métricas, intervalos de confianza, coste y análisis de errores |
| 7 | 9 | Explicabilidad: importancia por permutación y propia, estabilidad (Nogueira y Jaccard), elección del modelo final, modelos finales, SHAP de los finalistas, interpretación biológica, posible confusión por sexo y datos clínicos de las muestras mal clasificadas |
| 8 | 10 | Figura y tabla de resumen para la memoria |
| 9 | 11 | Exportación de los modelos finales y de los metadatos para la aplicación |
| 10 | 12 | Conclusión |

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
| `matrices_confusion.csv` | Matriz de confusión conjunta de cada modelo | 6.1 |
| `coste_y_genes.csv` | Tiempo medio y número de genes de cada modelo | 6.2 |
| `coordenadas_pca_umap.csv` | Coordenadas de cada muestra en la PCA y el UMAP (las usa la aplicación) | 6.3 |
| `analisis_errores.csv` | Para cada muestra: si es atípica, su grupo en el UMAP, la predicción de cada modelo y el sexo (proxy y registrado en TCGA) | 6.3 y 7.7 |
| `marcadores_grupos_umap.csv` | Expresión de marcadores conocidos en el grupo pequeño de BRCA y en las muestras LUAD mal clasificadas | 6.3 |
| `importancia_genes.csv` | Importancia por permutación y propia de cada gen en cada modelo | 7.1 |
| `estabilidad_nogueira.csv` | Índice de Nogueira de cada selección de genes | 7.2 |
| `jaccard_top50_modelos.csv` | Coincidencia de los 50 genes principales entre modelos | 7.2 |
| `comparacion_modelos.csv` | Criterios de la regla de elección y orden de los modelos | 7.3 |
| `shap_resumen.csv` | \|SHAP\| medio de cada gen por tipo tumoral en los finalistas | 7.5 |
| `interpretacion_biologica.csv` | Genes con más SHAP y marcadores conocidos por tipo tumoral | 7.6 |
| `proxy_sexo.csv` | XIST, media de los genes del cromosoma Y, patrón del proxy y sexo registrado en TCGA | 7.7 |
| `sensibilidad_sexo.csv` | Rendimiento de los finalistas con y sin los genes de los cromosomas X e Y | 7.7 |
| `errores_datos_clinicos.csv` | Tipo histológico, estadio y sexo registrados en TCGA de las muestras mal clasificadas | 7.8 |
| `tabla_resumen_memoria.csv` | Tabla resumen de los cuatro modelos para la memoria | 8 |

Modelos en `modelos/`: `modelo_final_<modelo>.joblib` (los cuatro flujos entrenados con las 801 muestras) y `metadatos.json` (modelo final, finalistas, genes de entrada, escala, genes e hiperparámetros de cada modelo y versiones), que carga la aplicación.

Figuras en `figuras/`, cada una en PNG (600 ppp), PDF y SVG: `fig6_matrices_confusion` (6.1), `fig7_umap_grupos` (6.3), `fig8_importancia_modelos` (7.1), `fig9_shap_por_clase` y `fig10_shap_beeswarm` (7.5), `fig11_marcadores` (7.6), `fig12_proxy_sexo` (7.7) y `fig13_comparacion_modelos` (8).
