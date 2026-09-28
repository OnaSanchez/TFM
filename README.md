# Desarrollo de un prototipo software interpretable para la clasificación de tipos tumorales a partir de datos de expresión génica mediante aprendizaje automático

Trabajo Final de Máster - Desarrollo de Programas y Aplicaciones - UOC

**Autora:** Ona Sánchez Núñez · **Tutor:** Giuseppe Tardiolo

**Palabras clave:** Gene expression, RNA-seq, Cancer classification, Machine learning, Explainable AI (SHAP), Nested cross-validation, Streamlit, Reproducible software

## Descripción

El TFM consiste en diseñar, implementar y validar un prototipo software demostrativo y reproducible, desarrollado en Python con una interfaz en Streamlit, para clasificar cinco tipos tumorales a partir de datos de expresión génica.

El trabajo combina dos partes:

- Una de **análisis de datos**, en la que se construyen, validan y comparan varios modelos de clasificación y se estudian los genes que más contribuyen a sus predicciones.
- Una de **desarrollo de software**, en la que todo este proceso se integra en una aplicación interactiva y reproducible que permite explorar los datos, comparar los modelos e interpretar sus resultados.

> El prototipo no es una herramienta diagnóstica ni clínica. Su finalidad es demostrativa, metodológica y docente.

## Objetivos

### Objetivos generales

- **OG1.** Desarrollar y validar un pipeline reproducible de aprendizaje automático para clasificar cinco tipos tumorales a partir de datos de expresión génica.
- **OG2.** Identificar e interpretar los genes que más contribuyen a la clasificación mediante técnicas de explicabilidad.
- **OG3.** Implementar un prototipo software interactivo y reproducible que integre la exploración de los datos, el entrenamiento y la comparación de modelos, su interpretación y (opcionalmente) la generación de informes.

## Datos

Se utiliza el dataset *Gene expression cancer RNA-Seq* del UCI Machine Learning Repository ([Fiorini, 2016](https://doi.org/10.24432/C5R88H)). Procede del proyecto TCGA Pan-Cancer y contiene 801 muestras y 20.531 genes.

| Tipo tumoral | Abreviatura | Muestras |
|---|---|---|
| Carcinoma invasivo de mama | BRCA | 300 |
| Carcinoma renal de células claras | KIRC | 146 |
| Adenocarcinoma de pulmón | LUAD | 141 |
| Adenocarcinoma de próstata | PRAD | 136 |
| Adenocarcinoma de colon | COAD | 78 |

## Enfoque y método

| Decisión | Estrategia |
|---|---|
| Datos | Dataset procesado UCI (TCGA PANCAN) |
| Modelos | Modelos clásicos (LR penalizada, RF, SVM, XGBoost) |
| Validación | CV anidada estratificada |
| Explicabilidad | Permutation importance + SHAP |
| Interfaz | Streamlit |
| Visualización | PCA (principal) + UMAP (exploratoria) |


## Aplicación web

Primera prueba del prototipo, publicada en Streamlit Community Cloud:

**https://tfm-tipos-tumorales.streamlit.app**

Permite subir `data.csv` y `labels.csv` o descargar el dataset directamente de UCI, y muestra el número de muestras y genes y el porcentaje de cada tipo tumoral.

## Estructura del repositorio

```
├── TFM_código.ipynb   # T2: obtención de los datos, QC y análisis exploratorio (PCA/UMAP)
├── TFM_streamlit.ipynb  # Primera prueba con Streamlit: carga del dataset
├── app/               # App de Streamlit publicada (streamlit_app.py y sus dependencias)
├── .streamlit/        # Configuración de Streamlit (límite de subida de 300 MB)
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

Para ejecutar la app en local:

```bash
streamlit run app/streamlit_app.py
```

También se puede abrir con VS Code (extensión de Jupyter). Las figuras se muestran en el notebook y se guardan en `figuras/`.


## Licencia de los datos

Los datos pertenecen a sus autores y se distribuyen bajo licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Si se reutilizan, deben citarse Fiorini (2016) y Weinstein et al. (2013).
