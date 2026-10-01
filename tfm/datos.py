"""Carga y validación del conjunto de datos «Gene expression cancer RNA-Seq» (UCI).

Lo usan el notebook de análisis (01_qc_eda.ipynb) y la aplicación de Streamlit,
para que los dos lean y comprueben los datos exactamente igual.
"""
import io
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

URL_UCI = "https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip"
CARPETA_TAR = "TCGA-PANCAN-HiSeq-801x20531"  # carpeta de los CSV dentro del .tar.gz
CLASES = ["BRCA", "COAD", "KIRC", "LUAD", "PRAD"]


def _descargar_tar():
    """Descarga el zip de UCI y devuelve el .tar.gz que contiene, abierto."""
    with urllib.request.urlopen(URL_UCI) as respuesta:
        zip_uci = zipfile.ZipFile(io.BytesIO(respuesta.read()))
    return tarfile.open(fileobj=io.BytesIO(zip_uci.read(f"{CARPETA_TAR}.tar.gz")))


def descargar_uci():
    """Descarga el conjunto de datos de UCI y devuelve (datos, etiquetas) sin guardarlo en disco."""
    with _descargar_tar() as tar:
        datos = pd.read_csv(tar.extractfile(f"{CARPETA_TAR}/data.csv"), index_col=0)
        etiquetas = pd.read_csv(tar.extractfile(f"{CARPETA_TAR}/labels.csv"), index_col=0)
    return datos, etiquetas


def cargar_datos(carpeta="Dataset"):
    """Lee data.csv y labels.csv de `carpeta` y devuelve (datos, etiquetas).

    Si los ficheros no están, los descarga de UCI y los guarda en `carpeta` tal cual,
    sin modificarlos, para que las siguientes ejecuciones no tengan que descargarlos.
    """
    carpeta = Path(carpeta)
    if not (carpeta / "data.csv").exists() or not (carpeta / "labels.csv").exists():
        print("Descargando el conjunto de datos de UCI...")
        carpeta.mkdir(parents=True, exist_ok=True)
        with _descargar_tar() as tar:
            for nombre in ("data.csv", "labels.csv"):
                (carpeta / nombre).write_bytes(tar.extractfile(f"{CARPETA_TAR}/{nombre}").read())
    datos = pd.read_csv(carpeta / "data.csv", index_col=0)
    etiquetas = pd.read_csv(carpeta / "labels.csv", index_col=0)
    return datos, etiquetas


def validar_datos(datos, etiquetas):
    """Comprueba que los datos tienen el formato esperado.

    Devuelve una lista con la descripción de cada problema encontrado (vacía si todo es correcto).
    """
    problemas = []
    if datos.empty:
        return ["El fichero de expresión está vacío."]
    if "Class" not in etiquetas.columns:
        problemas.append("El fichero de etiquetas no tiene la columna 'Class'.")
    elif not set(etiquetas["Class"].dropna()) <= set(CLASES):
        desconocidas = sorted(set(etiquetas["Class"].dropna()) - set(CLASES))
        problemas.append(f"Hay tipos tumorales desconocidos: {', '.join(map(str, desconocidas))}.")
    if not datos.index.equals(etiquetas.index):
        problemas.append("Las muestras de los dos ficheros no coinciden o no están en el mismo orden.")
    no_numericas = [c for c in datos.columns if not is_numeric_dtype(datos[c])]
    if no_numericas:
        problemas.append(f"Hay {len(no_numericas)} columnas de expresión con valores no numéricos.")
    elif (datos < 0).any().any():
        problemas.append("Hay valores de expresión negativos.")
    if datos.isna().any().any():
        problemas.append("Hay valores de expresión faltantes.")
    return problemas
