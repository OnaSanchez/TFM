# Paso 1 · Obtención de los datos, control de calidad y análisis exploratorio

`01_qc_eda.ipynb` (bloques 1 a 3 del flujo de análisis, tarea T2). La primera vez descarga los datos de UCI en `Dataset/`, en la raíz del repositorio.

| Apartado del notebook | Qué hace |
|---|---|
| 0 | Entorno, versiones de las librerías y semilla |
| 1 | Carga y validación de `data.csv` y `labels.csv` |
| 2 | Control de calidad: valores faltantes, duplicados, genes sin variación y muestras atípicas |
| 3 | Análisis exploratorio: clases, PCA (datos centrados y estandarizados), UMAP y mapa de calor |
| 4 | Conclusión |

## Resultados

| Fichero | Contenido |
|---|---|
| `resultados/qc_muestras.csv` | Para cada muestra: tipo tumoral, expresión media, % de ceros y si es atípica |
| `figuras/fig1_qc.png` | Control de calidad |
| `figuras/fig2_clases.png` | Número de muestras de cada tipo tumoral |
| `figuras/fig3_pca.png` | PCA con datos centrados y estandarizados |
| `figuras/fig4_umap.png` | UMAP |
| `figuras/fig5_heatmap.png` | Mapa de calor de los 500 genes más variables |

Siguiente paso: `correspondencia_genes/`.
