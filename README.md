# TFM – Clasificación interpretable de tipos tumorales a partir de datos de expresión génica

Trabajo Final de Máster (M0.211 – TFM Desarrollo de Programas y Aplicaciones, UOC).

**Autora:** Ona Sánchez Núñez · **Tutor:** Giuseppe Tardiolo

## Objetivo

Desarrollar un prototipo software reproducible e interpretable que clasifica cinco tipos tumorales a partir de datos de expresión génica (RNA-seq) mediante aprendizaje automático. El proyecto combina:

- una **pipeline de ML** validada con validación cruzada anidada estratificada, para evitar el *data leakage*;
- **técnicas de explicabilidad** (permutation importance y SHAP) para identificar los genes que más contribuyen a la clasificación;
- una **aplicación interactiva en Streamlit** para explorar los datos, comparar modelos e interpretar los resultados.

> El prototipo tiene una finalidad demostrativa, metodológica y docente. **No es una herramienta diagnóstica ni clínica.**

## Datos

Se utiliza el dataset *Gene expression cancer RNA-Seq* del UCI Machine Learning Repository ([Fiorini, 2016](https://doi.org/10.24432/C5R88H)), con licencia CC BY 4.0. Es una extracción aleatoria del proyecto TCGA Pan-Cancer ([Weinstein et al., 2013](https://doi.org/10.1038/ng.2764)) y contiene 801 muestras y 20.531 genes.

| Tipo tumoral | Abreviatura | Muestras |
|---|---|---|
| Carcinoma invasivo de mama | BRCA | 300 |
| Carcinoma renal de células claras | KIRC | 146 |
| Adenocarcinoma de pulmón | LUAD | 141 |
| Adenocarcinoma de próstata | PRAD | 136 |
| Adenocarcinoma de colon | COAD | 78 |

Los datos **no se incluyen en el repositorio**, porque superan el límite de tamaño de GitHub. El notebook los descarga automáticamente de UCI la primera vez que se ejecuta y los guarda en `Dataset/UCI/`.

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

## Estado del proyecto

| Tarea | Descripción | Estado |
|---|---|---|
| T0 | Definición y plan de trabajo (PEC1) | En curso |
| T1 | Estado del arte (PEC2) | Pendiente |
| T2 | Obtención de los datos, QC y análisis exploratorio | Hecho |
| T3 | Pipeline, validación cruzada anidada y comparación de modelos | Pendiente |
| T4 | Explicabilidad (permutation importance y SHAP) | Pendiente |
| T5 | Prototipo en Streamlit | Pendiente |
| T6 | Pruebas, documentación y reproducibilidad | Pendiente |

## Licencia de los datos

Los datos pertenecen a sus autores y se distribuyen bajo licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Si se reutilizan, deben citarse Fiorini (2016) y Weinstein et al. (2013).
