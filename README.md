# Desarrollo de un prototipo software interpretable para la clasificación de tipos tumorales a partir de datos de expresión génica mediante aprendizaje automático

Trabajo Final de Máster (M0.211 – TFM Desarrollo de Programas y Aplicaciones, UOC).

**Autora:** Ona Sánchez Núñez · **Tutor:** Giuseppe Tardiolo

**Palabras clave:** Gene expression, RNA-seq, Cancer classification, Machine learning, Explainable AI (SHAP), Nested cross-validation, Streamlit, Reproducible software

## Descripción

El TFM consiste en diseñar, implementar y validar un prototipo software demostrativo y reproducible, desarrollado en Python con una interfaz en Streamlit, para clasificar cinco tipos tumorales a partir de datos de expresión génica.

El trabajo combina dos partes:

- una de **análisis de datos**, en la que se construyen, validan y comparan varios modelos de clasificación y se estudian los genes que más contribuyen a sus predicciones;
- otra de **desarrollo de software**, en la que todo este proceso se integra en una aplicación interactiva y reproducible que permite explorar los datos, comparar los modelos e interpretar sus resultados.

Como extensión opcional, si el núcleo del prototipo está completo y validado, se podría incorporar un módulo con un modelo de lenguaje que redacte una interpretación preliminar de los resultados.

> El prototipo no es una herramienta diagnóstica ni clínica. Su finalidad es **demostrativa, metodológica y docente**.

## Objetivos

### Objetivos generales

- **OG1.** Desarrollar y validar un pipeline reproducible de aprendizaje automático para clasificar cinco tipos tumorales a partir de datos de expresión génica.
- **OG2.** Identificar e interpretar los genes que más contribuyen a la clasificación mediante técnicas de explicabilidad.
- **OG3.** Implementar un prototipo software interactivo y reproducible que integre la exploración de los datos, el entrenamiento y la comparación de modelos, su interpretación y (opcionalmente) la generación de informes.

### Objetivos específicos

- **OE1.** Caracterizar el conjunto de datos mediante un análisis exploratorio: distribución de clases, control de calidad y visualización con PCA y UMAP.
- **OE2.** Implementar una pipeline de preprocesamiento y selección de variables integrada dentro de cada partición de la validación, para evitar el *data leakage*.
- **OE3.** Comparar al menos tres modelos de clasificación mediante validación cruzada, reportando accuracy, F1 macro y la matriz de confusión.
- **OE4.** Identificar los genes que más contribuyen a la clasificación mediante permutation importance y valores SHAP, y evaluar su estabilidad entre particiones y modelos.
- **OE5.** Desarrollar una aplicación interactiva en Streamlit que integre la exploración de los datos, la comparación de modelos, la explicabilidad y la predicción de nuevas muestras.
- **OE6.** *(Opcional)* Integrar y evaluar un módulo con un modelo de lenguaje que redacte una interpretación preliminar de los resultados.

## Datos

Se utiliza el dataset *Gene expression cancer RNA-Seq* del UCI Machine Learning Repository ([Fiorini, 2016](https://doi.org/10.24432/C5R88H)), con licencia CC BY 4.0. Procede del proyecto TCGA Pan-Cancer ([Weinstein et al., 2013](https://doi.org/10.1038/ng.2764)) y contiene 801 muestras y 20.531 genes.

| Tipo tumoral | Abreviatura | Muestras |
|---|---|---|
| Carcinoma invasivo de mama | BRCA | 300 |
| Carcinoma renal de células claras | KIRC | 146 |
| Adenocarcinoma de pulmón | LUAD | 141 |
| Adenocarcinoma de próstata | PRAD | 136 |
| Adenocarcinoma de colon | COAD | 78 |

Las clases no están equilibradas, algo que se tiene en cuenta en la validación.

Los datos **no se incluyen en el repositorio**, porque superan el límite de tamaño de GitHub. El notebook los descarga automáticamente de UCI la primera vez que se ejecuta y los guarda en `Dataset/UCI/`.

## Enfoque y método

| Decisión | Estrategia |
|---|---|
| Datos | Dataset procesado UCI (TCGA PANCAN) |
| Modelos | Modelos clásicos (LR penalizada, RF, SVM, XGBoost) |
| Validación | CV anidada estratificada |
| Explicabilidad | Permutation importance + SHAP |
| Interfaz | Streamlit |
| Visualización | PCA (principal) + UMAP (exploratoria) |

**Metodología:**

1. **Obtención y control de calidad de los datos:** descarga de los ficheros originales (`data.csv`, `labels.csv`) y comprobación de dimensiones, valores faltantes y correspondencia entre muestras y etiquetas.
2. **Análisis exploratorio:** distribución de clases y visualización con PCA, UMAP y un heatmap de los genes más variables.
3. **Preparación de los datos:** dentro de cada partición se eliminan los genes que apenas varían, se seleccionan los más informativos y se ajusta la escala de los valores.
4. **Validación cruzada anidada:** estratificada y con métricas adecuadas para clases desequilibradas (F1 macro); resultados expresados como media ± desviación estándar.
5. **Interpretabilidad:** permutation importance y valores SHAP, y análisis de la estabilidad de los genes más relevantes entre particiones y modelos.
6. **Desarrollo del prototipo:** aplicación Streamlit organizada en módulos (datos, modelos, explicabilidad e informes).
7. **Reproducibilidad y pruebas:** repositorio Git, `requirements.txt`/`environment.yml`, semillas fijas, pruebas unitarias con pytest y README.
8. *(Opcional)* **Módulo LLM**, validado con una rúbrica: coherencia numérica, afirmaciones no respaldadas y reconocimiento de limitaciones.

**Herramientas:** Python, pandas, scikit-learn, XGBoost, SHAP, umap-learn, matplotlib/plotly, Streamlit, Git/GitHub.

## Estructura del repositorio

```
├── TFM_código.ipynb   # T2: obtención de los datos, QC y análisis exploratorio (PCA/UMAP)
├── figuras/           # Figuras generadas por el notebook
├── requirements.txt   # Dependencias (pip)
└── environment.yml    # Entorno reproducible (conda)
```

## Instalación

Con **conda**:

```bash
conda env create -f environment.yml
conda activate tfm
```

O con **pip** (Python 3.9):

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

## Uso

```bash
jupyter notebook TFM_código.ipynb
```

También se puede abrir con VS Code (extensión de Jupyter). Las figuras se muestran en el notebook y se guardan en `figuras/`.

## Planificación y estado del proyecto

El semestre va del 23/09/2026 al 29/01/2027.

| Tarea | Descripción | Objetivo(s) | Inicio | Fin | Estado |
|---|---|---|---|---|---|
| T0 | Definición y plan de trabajo (PEC1) | — | 23/09 | 07/10 | En curso |
| T1 | Desarrollo del trabajo: estado del arte (PEC2) | OG1–OG3 | 08/10 | 04/11 | Pendiente |
| T2 | Obtención datos, QC y análisis exploratorio (PCA/UMAP) | OE1 | 12/10 | 25/10 | Hecho |
| T3 | Pipeline de preprocesamiento + CV anidada + comparación de modelos | OE2, OE3 | 26/10 | 15/11 | Pendiente |
| T4 | Explicabilidad (permutation importance y SHAP) | OE4 | 16/11 | 29/11 | Pendiente |
| T5 | Arquitectura e implementación del prototipo Streamlit | OE5 | 26/10 | 29/11 | Pendiente |
| T6 | Informe de resultados, pruebas, documentación y reproducibilidad | OE5 | 23/11 | 01/12 | Pendiente |
| T7 | Análisis de resultados y redacción de la PEC3 (implementación) | OE1–OE5 | 23/11 | 02/12 | Pendiente |
| T8 | *(Opcional)* Módulo LLM y evaluación crítica | OE6 | 03/12 | 16/12 | Pendiente |
| T9 | Redacción de la memoria final (PEC4) | Todos | 12/10 | 07/01 | Pendiente |
| T10 | Presentación y grabación (PEC5) | Todos | 08/01 | 13/01 | Pendiente |
| T11 | Preparación de la defensa del TFM | Todos | 14/01 | 29/01 | Pendiente |

### Hitos

| Hito | Fecha | Descripción |
|---|---|---|
| H1 | 07/10/2026 | Entrega del plan de trabajo (PEC1) |
| H2 | 04/11/2026 | Entrega del estado del arte (PEC2) y análisis exploratorio terminado |
| H3 | 15/11/2026 | Pipeline validada y tabla comparativa de modelos |
| H4 | 02/12/2026 | Prototipo funcional y entrega de la implementación (PEC3) |
| H5 | 07/01/2027 | Entrega de la memoria final (PEC4) |
| H6 | 13/01/2027 | Entrega de la presentación y grabación (PEC5) |
| H7 | 29/01/2027 | Defensa del TFM |

## Licencia de los datos

Los datos pertenecen a sus autores y se distribuyen bajo licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Si se reutilizan, deben citarse Fiorini (2016) y Weinstein et al. (2013).
