import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# El paquete tfm (carga y validación de los datos) está en la raíz del repositorio
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tfm.datos import descargar_uci, validar_datos  # noqa: E402


# Con @st.cache_data el conjunto de datos de UCI solo se descarga la primera vez
@st.cache_data
def descargar_uci_cache():
    return descargar_uci()


st.title("Clasificación de tipos tumorales a partir de la expresión génica")
st.write("Conjunto de datos *Gene expression cancer RNA-Seq* del "
         "[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq): "
         "`data.csv` (expresión de los genes) y `labels.csv` (tipo tumoral de cada muestra).")
st.caption("Prototipo del Trabajo Final de Máster de Ona Sánchez Núñez, Máster universitario en "
           "Bioinformática y Bioestadística (UOC, UB). Uso exclusivamente metodológico y docente: "
           "no es una herramienta diagnóstica ni clínica.")

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
