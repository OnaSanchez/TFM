import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# El paquete tfm (carga y validación de los datos) está en la raíz del repositorio
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from tfm.datos import CLASES, descargar_uci, validar_datos  # noqa: E402

# Los resultados los generan los notebooks; la aplicación solo los lee (no repite la validación cruzada)
RES_QC = RAIZ / "qc_eda" / "resultados"
RES_MODELADO = RAIZ / "modelado" / "resultados"
MODELOS = {"LR": "Regresión logística", "SVM": "SVM lineal", "RF": "Bosque aleatorio (RF)", "XGBoost": "XGBoost"}


# Con @st.cache_data el conjunto de datos de UCI solo se descarga la primera vez
@st.cache_data
def descargar_uci_cache():
    return descargar_uci()


# Con @st.cache_data cada tabla de resultados solo se lee una vez
@st.cache_data
def leer(ruta):
    return pd.read_csv(ruta)


st.sidebar.title("Tipos tumorales")
pagina = st.sidebar.radio("Sección", ["Datos", "Exploración", "Comparación de modelos"])

# ---------------------------------------------------------------------------------------------------
if pagina == "Datos":
    st.title("Clasificación de tipos tumorales a partir de la expresión génica")
    st.write("Conjunto de datos *Gene expression cancer RNA-Seq* del "
             "[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq): "
             "`data.csv` (expresión de los genes) y `labels.csv` (tipo tumoral de cada muestra).")

    opcion = st.radio("¿Qué datos quieres usar?", ["Subir mis ficheros", "Usar el conjunto de datos de UCI"])

    datos = None
    etiquetas = None

    if opcion == "Subir mis ficheros":
        archivo_datos = st.file_uploader("Fichero de expresión (data.csv)", type="csv")
        archivo_etiquetas = st.file_uploader("Fichero de etiquetas (labels.csv)", type="csv")

        # Solo se hace algo cuando se han subido los dos ficheros
        if archivo_datos is not None and archivo_etiquetas is not None:
            try:
                with st.spinner("Leyendo los datos..."):
                    datos = pd.read_csv(archivo_datos, index_col=0)
                    etiquetas = pd.read_csv(archivo_etiquetas, index_col=0)
            except Exception:
                st.error("No se han podido leer los ficheros. Comprueba que son CSV separados por comas.")
                st.stop()
    else:
        try:
            with st.spinner("Descargando el conjunto de datos de UCI (unos 70 MB)..."):
                datos, etiquetas = descargar_uci_cache()
        except Exception:
            st.error("No se ha podido descargar el conjunto de datos de UCI. Prueba más tarde o sube los ficheros.")
            st.stop()

    if datos is not None:
        # Mismas comprobaciones que en el notebook (tfm/datos.py)
        problemas = validar_datos(datos, etiquetas)
        if problemas:
            for problema in problemas:
                st.error(problema)
        else:
            st.success("¡El conjunto de datos se ha leído correctamente!")

            st.write("Número de muestras:", datos.shape[0])
            st.write("Número de genes:", datos.shape[1])

            st.subheader("Porcentaje de cada tipo tumoral")
            conteo = etiquetas["Class"].value_counts()
            porcentaje = (conteo / conteo.sum() * 100).round(1)
            # astype(str) para que la tabla muestre 37.5 y no 37.5000
            tabla = pd.DataFrame({"Muestras": conteo, "Porcentaje (%)": porcentaje.astype(str)})
            st.table(tabla)
            st.bar_chart(porcentaje)

# ---------------------------------------------------------------------------------------------------
elif pagina == "Exploración":
    st.title("Exploración de los datos")
    st.write("Cada punto es una muestra. La PCA (análisis de componentes principales) es la representación "
             "principal; el UMAP solo sirve para explorar: puede sugerir grupos, pero no los demuestra.")

    coords = leer(RES_MODELADO / "coordenadas_pca_umap.csv").set_index("muestra")
    qc = leer(RES_QC / "qc_muestras.csv").set_index("Unnamed: 0")
    errores = leer(RES_MODELADO / "analisis_errores.csv").set_index("Unnamed: 0")

    representacion = st.radio("Representación", ["PCA", "UMAP"], horizontal=True)
    colorear = st.selectbox("Colorear por", ["Tipo tumoral", "Muestra atípica en el control de calidad",
                                             "Mal clasificada por algún modelo"])

    tabla = coords.copy()
    if colorear == "Tipo tumoral":
        tabla["Grupo"] = tabla["Tipo"]
    elif colorear == "Muestra atípica en el control de calidad":
        tabla["Grupo"] = qc["Atípica"].map({True: "Atípica", False: "No atípica"})
    else:
        tabla["Grupo"] = (errores["modelos_que_fallan"] > 0).map({True: "Mal clasificada", False: "Bien clasificada"})

    x, y = ("PC1", "PC2") if representacion == "PCA" else ("UMAP1", "UMAP2")
    st.scatter_chart(tabla, x=x, y=y, color="Grupo", height=500)
    st.write("Número de muestras de cada grupo:")
    st.table(tabla["Grupo"].value_counts().rename("Muestras"))

# ---------------------------------------------------------------------------------------------------
else:
    st.title("Comparación de modelos")
    st.write("Rendimiento de los cuatro modelos en la validación cruzada anidada (5 particiones externas y 3 "
             "internas). Cada muestra se predice una sola vez, con un modelo que no la ha visto al entrenar "
             "(predicciones *out-of-fold*). Entre corchetes, el intervalo de confianza del 95 % por bootstrap "
             "estratificado.")

    metricas = leer(RES_MODELADO / "metricas_globales_ic95.csv")
    metricas["valor"] = [f"{f.estimacion:.3f} [{f.ic_inferior:.3f}–{f.ic_superior:.3f}]" for f in metricas.itertuples()]
    tabla = metricas.pivot(index="modelo", columns="metrica", values="valor")
    tabla = tabla.reindex(index=list(MODELOS), columns=metricas["metrica"].unique()).rename(index=MODELOS)
    st.subheader("Métricas globales")
    st.table(tabla)

    st.subheader("Errores e intervalo exacto de la exactitud")
    st.write("Cuando un modelo no comete errores, el bootstrap da un intervalo [1–1]; el intervalo exacto de "
             "Clopper-Pearson indica que la exactitud real podría ser algo menor.")
    st.table(leer(RES_MODELADO / "exactitud_ic_exacto.csv").set_index("modelo").round(4))

    st.subheader("F1 de cada tipo tumoral")
    por_clase = leer(RES_MODELADO / "metricas_por_clase.csv")
    st.table(por_clase.pivot(index="clase", columns="modelo", values="f1")[list(MODELOS)]
             .rename(columns=MODELOS).round(3))

    st.subheader("Detalle de un modelo")
    modelo = st.selectbox("Modelo", list(MODELOS), format_func=lambda m: MODELOS[m])

    predicciones = leer(RES_MODELADO / "predicciones_oof.csv")
    predicciones = predicciones[predicciones["modelo"] == modelo]
    st.write("Matriz de confusión (filas: clase real; columnas: clase predicha):")
    matriz = pd.crosstab(predicciones["real"], predicciones["pred"]).reindex(index=CLASES, columns=CLASES, fill_value=0)
    st.table(matriz)

    st.write("Número de genes e hiperparámetros elegidos en cada partición externa:")
    hiper = leer(RES_MODELADO / "hiperparametros_por_particion.csv")
    hiper = hiper[hiper["modelo"] == modelo].dropna(axis=1).drop(columns="modelo")
    st.table(hiper.set_index("particion").round(4))

    st.subheader("Muestras mal clasificadas por algún modelo")
    errores = leer(RES_MODELADO / "analisis_errores.csv").set_index("Unnamed: 0").rename_axis("Muestra")
    columnas = ["Tipo", "Atípica QC", "Grupo UMAP", "modelos_que_fallan"] + [f"pred_{m}" for m in MODELOS]
    mal = errores.loc[errores["modelos_que_fallan"] > 0, columnas].sort_values("modelos_que_fallan", ascending=False)
    st.table(mal.rename(columns={"modelos_que_fallan": "Modelos que fallan", **{f"pred_{m}": m for m in MODELOS}}))
