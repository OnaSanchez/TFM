# Clasificación de tipos tumorales a partir de la expresión génica con aprendizaje automático explicable: un prototipo de software interactivo y reproducible

Trabajo Final de Máster (TFM) realizado en el marco del Máster universitario en Bioinformática y Bioestadística, interuniversitario entre la Universitat Oberta de Catalunya (UOC) y la Universitat de Barcelona (UB).

- **Área:** Área 5: Desarrollo de Programas y Aplicaciones
- **Autora:** Ona Sánchez Núñez
- **Tutor:** Giuseppe Tardiolo
- **Palabras clave:** expresión génica, secuenciación de ARN (RNA-seq), clasificación del cáncer, aprendizaje automático, inteligencia artificial explicable, validación cruzada anidada, Streamlit, software reproducible

> **Aviso:** el prototipo no es una herramienta diagnóstica ni clínica. Su finalidad es exclusivamente metodológica y docente.

## Objetivo

El TFM consiste en diseñar, implementar y validar un prototipo de software demostrativo y reproducible, desarrollado en Python con una interfaz en Streamlit, para clasificar cinco tipos tumorales a partir de datos de expresión génica.

El trabajo combina dos partes:

- Una de **análisis de datos**, en la que se construyen, validan y comparan varios modelos de clasificación y se estudian las variables (genes) que más contribuyen a sus predicciones.
- Una de **desarrollo de software**, en la que todo este proceso se integra en una aplicación interactiva y reproducible que permite explorar los datos, comparar los modelos e interpretar sus resultados.

Clasificar estos cinco tipos tumorales no es, por sí solo, una aportación nueva: con este mismo conjunto de datos ya se han publicado resultados cercanos al techo de rendimiento (Akter et al., 2025). Por eso el objetivo no es maximizar la exactitud, sino combinar en un mismo trabajo una validación sin fuga de información, una comparación controlada de modelos, el estudio de la estabilidad de las variables seleccionadas, la explicabilidad, la reproducibilidad y una interfaz de software utilizable.

### Objetivos Generales (OG)

- **OG1.** Desarrollar y validar un flujo de análisis reproducible de aprendizaje automático para clasificar cinco tipos tumorales a partir de datos de expresión génica.
- **OG2.** Identificar las variables que más contribuyen a la clasificación mediante técnicas de explicabilidad y, cuando sea posible establecer una correspondencia fiable con identificadores génicos, explorar su interpretación biológica.
- **OG3.** Implementar un prototipo de software interactivo y reproducible que integre la exploración de los datos, el entrenamiento y la comparación de modelos, su interpretación y (opcionalmente) la generación de informes.

## Datos

Se utiliza el conjunto de datos *Gene expression cancer RNA-Seq* del UCI Machine Learning Repository ([Fiorini, 2016](https://doi.org/10.24432/C5R88H)). Procede del proyecto The Cancer Genome Atlas (TCGA) Pan-Cancer, abreviado PANCAN (Weinstein et al., 2013), y contiene 801 muestras y 20.531 genes.

| Tipo tumoral | Abreviatura | Muestras |
|---|---|---|
| Carcinoma invasivo de mama | BRCA | 300 |
| Carcinoma renal de células claras | KIRC | 146 |
| Adenocarcinoma de pulmón | LUAD | 141 |
| Adenocarcinoma de próstata | PRAD | 136 |
| Adenocarcinoma de colon | COAD | 78 |

Los genes aparecen con identificadores genéricos (`gene_0`, `gene_1`…). Según la ficha de UCI, las variables conservan el orden de la fuente original PANCAN ([Synapse syn4301332](https://www.synapse.org/#!Synapse:syn4301332)). La correspondencia con los genes de TCGA se ha reconstruido y verificado con datos públicos de la misma fuente (TCGA RNASeqV2, Broad GDAC Firehose): con el mismo orden de genes, las 78 muestras COAD de UCI coinciden exactamente con 78 muestras COAD de TCGA. La tabla `gene_X` → símbolo del gen e identificador Entrez está en `correspondencia_genes/resultados/correspondencia_genes.csv` y se genera con:

```bash
python correspondencia_genes/correspondencia_genes.py
```

Los datos humanos son de acceso público y no contienen identificadores personales directos. El proyecto no intenta reidentificar a los participantes y los utiliza solo con fines metodológicos y académicos.

## Enfoque y método

| Decisión | Estrategia |
|---|---|
| Datos | Conjunto de datos procesado de UCI (TCGA PANCAN) |
| Modelos | Regresión logística multinomial penalizada (modelo de referencia), bosque aleatorio (Random Forest, RF), máquina de vectores de soporte (Support Vector Machine, SVM) lineal y Extreme Gradient Boosting (XGBoost) |
| Validación | Validación cruzada (cross-validation, CV) anidada y estratificada: 5 particiones externas y 3 internas, las mismas particiones externas para todos los modelos |
| Métricas | F1 macro (media no ponderada del F1 de cada clase) como métrica principal; exactitud equilibrada, exactitud, precisión, sensibilidad y F1 por clase, y matriz de confusión |
| Explicabilidad | Importancia por permutación para todos los modelos, estabilidad de la selección de genes (índice de Nogueira) y SHapley Additive exPlanations (SHAP) para los modelos finalistas, con interpretación biológica de los genes |
| Selección del modelo | Candidatos: F1 macro a menos de 0,01 del mejor; entre ellos, menos genes, más estabilidad y menos tiempo de ajuste |
| Interfaz | Streamlit |
| Visualización | Análisis de componentes principales (Principal Component Analysis, PCA) como representación principal y Uniform Manifold Approximation and Projection (UMAP) solo como exploración |

Todo lo que aprende de los datos (filtrado, selección de variables, escalado y ajuste de hiperparámetros) se ejecuta dentro de cada partición de la validación, para evitar la fuga de información (data leakage). La validación cruzada anidada estima el rendimiento de la estrategia de modelado, pero no sustituye una validación externa en una cohorte independiente, que queda como limitación del trabajo y posible extensión.

## Aplicación web

Prototipo publicado en Streamlit Community Cloud: **https://tfm-tipos-tumorales.streamlit.app**

**Requisitos del prototipo**

| Aspecto | Definición |
|---|---|
| Usuario previsto | Estudiantes y personal docente o investigador que quieran explorar el problema y los modelos. No es para uso clínico. |
| Entrada | `data.csv` (muestras × genes, valores numéricos) y `labels.csv` (columna `Class`), con el formato de UCI. También puede usarse directamente el conjunto de datos de UCI. |
| Comprobaciones | Columna `Class` presente, tipos tumorales conocidos, mismas muestras y en el mismo orden en los dos ficheros, valores numéricos, no negativos y sin faltantes (`tfm/datos.py`). |
| Resultados | Datos: número de muestras y genes y distribución de los tipos tumorales. Exploración: PCA y UMAP, coloreados por tipo tumoral, muestras atípicas o muestras mal clasificadas. Comparación de modelos: métricas con intervalos de confianza, F1 por clase, matriz de confusión, hiperparámetros y muestras mal clasificadas. Explicabilidad: importancia de los genes, SHAP de los finalistas, interpretación biológica, estabilidad y análisis del sexo. Predicción: tipo predicho y puntuación de cada clase, genes que más contribuyen (modelos lineales) e informe descargable. |
| Funciones obligatorias | El prototipo se considerará terminado cuando integre la exploración de los datos, la comparación de modelos, la explicabilidad y la predicción de nuevas muestras (objetivo específico 5 del plan de trabajo). La generación de informes con un modelo de lenguaje es una extensión opcional. |
| Cálculo | La validación cruzada anidada no se ejecuta en la aplicación: el análisis se ejecuta en los cuadernos de Jupyter, que guardan sus resultados en tablas (`qc_eda/resultados/` y `modelado/resultados/`); la aplicación solo las lee, con caché. |
| Predicción | Validación estricta (`tfm.datos.validar_nuevas_muestras`): las nuevas muestras tienen que tener exactamente los mismos 20.531 identificadores `gene_X`, sin faltantes ni negativos y en escala log2(x + 1); se les aplica el mismo flujo (selección de genes y escalado) que en el entrenamiento. |

Estado actual: las cinco secciones están implementadas.

| Sección | Qué muestra |
|---|---|
| Datos | Carga de `data.csv` y `labels.csv` (o descarga de UCI), validación del formato y distribución de los tipos tumorales |
| Exploración | PCA y UMAP coloreados por tipo tumoral, muestras atípicas o muestras mal clasificadas |
| Comparación de modelos | Métricas con IC 95 %, intervalo exacto de la exactitud, F1 por clase, regla de elección del modelo final, matriz de confusión, hiperparámetros y muestras mal clasificadas |
| Explicabilidad | Importancia propia y por permutación, SHAP de los finalistas por tipo tumoral, interpretación biológica, estabilidad (Nogueira y Jaccard) y sensibilidad al sexo |
| Predicción de nuevas muestras | Validación estricta, predicción con el modelo final (o con cualquiera de los cuatro), contribución de cada gen (modelos lineales) y **módulo de informes**: informe HTML descargable con las predicciones, la validación, el rendimiento estimado, los genes que más contribuyen, las versiones y el aviso de uso |

Siguiendo el formato acordado con el tutor, los títulos, tablas, gráficos y resultados aparecen centrados.

## Resultados

Resultados del cuaderno `modelado/02_modelado.ipynb` (CV anidada con 5 particiones externas y 3 internas; predicciones out-of-fold de las 801 muestras; IC 95 % por bootstrap estratificado):

| Modelo | F1 macro [IC 95 %] | Errores | Genes (mediana) | Estabilidad (Nogueira, 50 genes) | Tiempo por partición |
|---|---|---|---|---|---|
| Regresión logística multinomial penalizada | 0,999 [0,997–1,000] | 1 | 200 | 0,58 | 28 s |
| **SVM lineal (modelo final)** | 1,000 [1,000–1,000] | 0 | 200 | 0,82 | 7 s |
| Bosque aleatorio (RF) | 0,996 [0,992–0,999] | 4 | 1.000 | 0,50 | 13 s |
| XGBoost | 0,996 [0,990–1,000] | 3 | 200 | 0,52 | 20 s |

- Los cuatro modelos rinden casi igual y sus intervalos se solapan. Como el rendimiento está en el techo, el modelo final se elige con una regla fijada de antemano (F1 macro a menos de 0,01 del mejor; después, menos genes, más estabilidad y menos tiempo): la **SVM lineal con 200 genes**, con la regresión logística como segundo finalista. Su exactitud es ≥ 0,995 con un 95 % de confianza (intervalo exacto de Clopper-Pearson).
- **Interpretación biológica:** los genes con más SHAP son marcadores conocidos del tejido de origen: surfactantes, NKX2-1 y NAPSA en pulmón; PCA3, ACPP, KLK2, KLK3 y NKX3-1 en próstata; CDX1, CDX2 y CDH17 en colon; TRPS1 y GATA3 en mama, y transportadores del túbulo proximal y PAX2 en riñón.
- La importancia por permutación de genes individuales es casi nula y las listas de genes cambian entre modelos (Jaccard 0,22–0,61) por la redundancia entre genes, sin que cambie el rendimiento.
- Solo 6 muestras se clasifican mal con algún modelo; las dos que fallan más son muestras LUAD situadas, en el UMAP, junto a un pequeño grupo de BRCA de perfil basal (registradas en TCGA como adenocarcinoma de pulmón).
- **Sexo:** el proxy basado en XIST y genes del cromosoma Y coincide con el sexo registrado en TCGA en el 99,9 % de las muestras, y sin los 877 genes de los cromosomas X e Y la SVM no cambia ninguna predicción.
- **Limitación principal:** no hay validación externa en una cohorte independiente.

## Instalación y uso

Con conda (recomendado):

```bash
conda env create -f environment.yml
conda activate tfm
```

O con pip (Python 3.12):

```bash
pip install -r requirements.txt
```

Se ha comprobado que `requirements.txt` se instala sin errores con Python 3.12.16 y que pasan las 28 pruebas. En Windows, si la instalación falla con un error de ruta demasiado larga (paquete `notebook`), hay que activar las rutas largas de Windows o crear el entorno en una carpeta con una ruta corta.

Después, desde la raíz del repositorio, el análisis se ejecuta en tres pasos, en este orden (cada carpeta tiene un README con lo que hace cada apartado y qué contiene cada fichero de resultados):

| Paso | Carpeta | Qué hace | Bloques del flujo |
|---|---|---|---|
| 1 | `qc_eda/` | Obtención de los datos, control de calidad (Quality Control, QC) y análisis exploratorio | 1 a 3 |
| 2 | `correspondencia_genes/` | Correspondencia `gene_X` → genes de TCGA | — |
| 3 | `modelado/` | Particiones, preprocesamiento, selección de genes, CV anidada de los cuatro modelos, evaluación, explicabilidad, elección del modelo final y exportación para la aplicación | 4 a 12 |

```bash
jupyter notebook qc_eda/01_qc_eda.ipynb                # paso 1: descarga los datos en Dataset/ la primera vez
python correspondencia_genes/correspondencia_genes.py   # paso 2: descarga unos 240 MB de TCGA (Firehose)
jupyter notebook modelado/02_modelado.ipynb             # paso 3: unos 30 minutos con 12 núcleos
streamlit run app/streamlit_app.py                      # aplicación en local (lee los resultados y los modelos de los pasos 1 y 3)
pytest                                                  # pruebas de la carga, la validación, el QC, la correspondencia de genes, los modelos y las métricas
```

Los cuadernos también pueden abrirse en Google Colab: si no encuentran el paquete `tfm`, descargan este repositorio automáticamente.

## Estructura del repositorio

```
├── qc_eda/                     # Paso 1 (T2): obtención de los datos, QC y análisis exploratorio
│   ├── README.md               #   apartados del cuaderno y contenido de cada resultado
│   ├── 01_qc_eda.ipynb         #   cuaderno del análisis
│   ├── figuras/                #   figuras generadas (PNG a 600 ppp, PDF y SVG)
│   └── resultados/             #   QC por muestra (qc_muestras.csv)
├── correspondencia_genes/      # Paso 2: correspondencia gene_X → genes de TCGA (riesgo R3 de la PEC1)
│   ├── README.md
│   ├── correspondencia_genes.py  # descarga TCGA de Firehose, verifica con las 801 muestras y genera la tabla
│   └── resultados/             #   tabla gene_X → símbolo y Entrez, y muestra de TCGA de cada muestra de UCI
├── modelado/                   # Paso 3 (T3 y T4): modelado, evaluación, explicabilidad y exportación
│   ├── README.md               #   apartados del cuaderno y contenido de cada resultado
│   ├── 02_modelado.ipynb       #   cuaderno del modelado (bloques 4 a 12)
│   ├── figuras/                #   figuras generadas (PNG a 600 ppp, PDF y SVG)
│   ├── resultados/             #   métricas, predicciones, errores, importancias, estabilidad, SHAP y sexo
│   └── modelos/                #   modelos finales (joblib) y metadatos que carga la aplicación
├── app/                        # Aplicación de Streamlit (lee los resultados de qc_eda/ y modelado/)
│   ├── streamlit_app.py        #   aplicación publicada en Streamlit Community Cloud
│   └── requirements.txt        #   dependencias que instala Streamlit Community Cloud
├── .streamlit/config.toml      # Configuración de Streamlit (límite de subida de 300 MB); tiene que estar en la raíz
├── tfm/                        # Funciones compartidas por los cuadernos, el script y la aplicación
│   ├── datos.py                #   carga y validación de los datos y de las nuevas muestras
│   ├── qc.py                   #   QC
│   ├── genes.py                #   correspondencia de genes, datos de TCGA (Firehose) y cromosomas (NCBI)
│   ├── modelos.py              #   flujos de los modelos, CV anidada, importancias y estabilidad de Nogueira
│   └── metricas.py             #   métricas e intervalos de confianza (bootstrap estratificado y exacto)
├── tests/                      # Pruebas con pytest (datos, QC, genes, modelos y métricas)
├── requirements.txt            # Dependencias (pip)
└── environment.yml             # Entorno reproducible (conda)
```

Los datos de UCI y de TCGA se descargan en `Dataset/`, en la raíz del repositorio, y no se suben a GitHub.

## Reproducibilidad

Las versiones exactas de las dependencias están fijadas en `requirements.txt` y `environment.yml`, todos los procesos aleatorios usan una semilla fija y cada cuaderno muestra al principio las versiones del entorno con las que se ha ejecutado.

## Licencia de los datos

Los datos pertenecen a sus autores y se distribuyen bajo licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Si se reutilizan, deben citarse Fiorini (2016) y Weinstein et al. (2013).

## Referencias

- Akter, S., Adesola, R. O., & Basnet, S. (2025). Machine learning approach to identify significant genes and classify cancer types from RNA-seq data. *Global Medical Genetics, 12*(4), 100079. https://doi.org/10.1016/j.gmg.2025.100079
- Broad Institute TCGA Genome Data Analysis Center. (2016). *Analysis-ready standardized TCGA data from Broad GDAC Firehose 2016_01_28 run* [Conjunto de datos]. Broad Institute of MIT and Harvard. https://doi.org/10.7908/C11G0KM9
- Fiorini, S. (2016). *Gene expression cancer RNA-Seq* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5R88H
- Weinstein, J. N., Collisson, E. A., Mills, G. B., Shaw, K. R. M., Ozenberger, B. A., Ellrott, K., Shmulevich, I., Sander, C., Stuart, J. M., & The Cancer Genome Atlas Research Network. (2013). The Cancer Genome Atlas Pan-Cancer analysis project. *Nature Genetics, 45*(10), 1113–1120. https://doi.org/10.1038/ng.2764
