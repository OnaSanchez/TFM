import io
import tarfile
import urllib.request
import zipfile

import streamlit as st
import pandas as pd

URL_UCI = "https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip"


# Descarga el dataset de la web de UCI. Con @st.cache_data solo se descarga la primera vez.
@st.cache_data
def descargar_uci():
    with urllib.request.urlopen(URL_UCI) as respuesta:
        zip_uci = zipfile.ZipFile(io.BytesIO(respuesta.read()))
    # El zip contiene un .tar.gz con los dos CSV
    tar_gz = io.BytesIO(zip_uci.read("TCGA-PANCAN-HiSeq-801x20531.tar.gz"))
    with tarfile.open(fileobj=tar_gz) as tar:
        datos = pd.read_csv(tar.extractfile("TCGA-PANCAN-HiSeq-801x20531/data.csv"), index_col=0)
        etiquetas = pd.read_csv(tar.extractfile("TCGA-PANCAN-HiSeq-801x20531/labels.csv"), index_col=0)
    return datos, etiquetas


st.title("Clasificación de tipos tumorales")
st.write("Dataset *Gene expression cancer RNA-Seq* del "
         "[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq): "
         "`data.csv` (expresión de los genes) y `labels.csv` (tipo de cáncer de cada muestra).")
st.caption("Prototipo del TFM de Ona Sánchez Núñez (UOC). No es una herramienta diagnóstica ni clínica.")

opcion = st.radio("¿Qué datos quieres usar?", ["Subir mis ficheros", "Usar el dataset de UCI"])

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
        with st.spinner("Descargando el dataset de UCI (unos 70 MB)..."):
            datos, etiquetas = descargar_uci()
    except Exception:
        st.error("No se ha podido descargar el dataset de UCI. Prueba más tarde o sube los ficheros.")
        st.stop()

if datos is not None:
    # Comprobaciones básicas
    if "Class" not in etiquetas.columns:
        st.error("El fichero de etiquetas no tiene la columna 'Class'.")
    elif not datos.index.equals(etiquetas.index):
        st.error("Las muestras de los dos ficheros no coinciden.")
    else:
        st.success("¡El dataset se ha leído correctamente!")

        st.write("Número de muestras:", datos.shape[0])
        st.write("Número de genes:", datos.shape[1])

        st.subheader("Porcentaje de cada tipo de cáncer")
        conteo = etiquetas["Class"].value_counts()
        porcentaje = (conteo / conteo.sum() * 100).round(1)
        # astype(str) para que la tabla muestre 37.5 y no 37.5000
        tabla = pd.DataFrame({"Muestras": conteo, "Porcentaje (%)": porcentaje.astype(str)})
        st.table(tabla)
        st.bar_chart(porcentaje)
