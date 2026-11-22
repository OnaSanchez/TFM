"""Correspondencia entre los identificadores genéricos de UCI (gene_X) y los genes de TCGA.

La ficha de UCI indica que las variables conservan el orden de la fuente original PANCAN
(Synapse syn4301332, fichero unc.edu_PANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv). Ese fichero
requiere registro, pero los mismos datos de TCGA (RNASeqV2, RSEM normalizado, procesado por UNC)
son públicos en Broad GDAC Firehose. Si gene_i es el gen i del fichero de TCGA, cada muestra de UCI
debe coincidir exactamente con una muestra de TCGA del mismo tipo tumoral.
"""
import io
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

_BASE_FIREHOSE = "https://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/{c}/20160128/gdac.broadinstitute.org_{c}."
URL_FIREHOSE = (_BASE_FIREHOSE + "Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__"
                "RSEM_genes_normalized__data.Level_3.2016012800.0.0.tar.gz")
URL_FIREHOSE_CLINICO = _BASE_FIREHOSE + "Clinical_Pick_Tier1.Level_4.2016012800.0.0.tar.gz"
URL_FIREHOSE_COAD = URL_FIREHOSE.format(c="COAD")
URL_NCBI_GENE_INFO = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"


def _extraer_de_tar(url, terminacion, ruta):
    """Descarga un .tar.gz y guarda en `ruta` el único fichero cuyo nombre acaba en `terminacion`."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as respuesta:
        tar = tarfile.open(fileobj=io.BytesIO(respuesta.read()))
    miembro = next(m for m in tar.getmembers() if m.name.endswith(terminacion))
    ruta.write_bytes(tar.extractfile(miembro).read())


def cargar_firehose(cohorte, carpeta="Dataset"):
    """Matriz genes x muestras (RSEM normalizado) de una cohorte de TCGA. La descarga la primera vez."""
    ruta = Path(carpeta) / f"firehose_{cohorte}_RSEM_genes_normalized.txt"
    if not ruta.exists():
        print(f"Descargando TCGA {cohorte} de Broad GDAC Firehose...")
        _extraer_de_tar(URL_FIREHOSE.format(c=cohorte), ".data.txt", ruta)
    # La segunda fila de cabecera («gene_id / normalized_count») no contiene datos
    return pd.read_csv(ruta, sep="\t", index_col=0, skiprows=[1]).apply(pd.to_numeric)


def cargar_firehose_coad(carpeta="Dataset"):
    """Matriz genes x muestras de TCGA COAD (la cohorte con la que se reconstruyó la correspondencia)."""
    return cargar_firehose("COAD", carpeta)


def cargar_clinico_firehose(cohorte, carpeta="Dataset"):
    """Variables clínicas básicas (Clinical_Pick_Tier1) de una cohorte de TCGA, una fila por paciente.

    El índice es el código del paciente en mayúsculas (TCGA-XX-XXXX), que son los 12 primeros
    caracteres del código de cada muestra.
    """
    ruta = Path(carpeta) / f"firehose_{cohorte}_clinico.txt"
    if not ruta.exists():
        print(f"Descargando los datos clínicos de TCGA {cohorte}...")
        _extraer_de_tar(URL_FIREHOSE_CLINICO.format(c=cohorte), ".clin.merged.picked.txt", ruta)
    clinico = pd.read_csv(ruta, sep="\t", index_col=0).T  # el fichero tiene una columna por paciente
    clinico.index = clinico.index.str.upper()
    return clinico


def cargar_cromosomas(carpeta="Dataset"):
    """Cromosoma de cada gen humano según NCBI Gene (Entrez ID → cromosoma). Lo descarga la primera vez."""
    ruta = Path(carpeta) / "Homo_sapiens.gene_info.gz"
    if not ruta.exists():
        print("Descargando la anotación de genes de NCBI...")
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL_NCBI_GENE_INFO) as respuesta:
            ruta.write_bytes(respuesta.read())
    info = pd.read_csv(ruta, sep="\t", usecols=["GeneID", "chromosome"], dtype=str)
    return info.set_index("GeneID")["chromosome"]


def tabla_correspondencia(genes_uci, gene_ids_tcga):
    """Une cada gene_X con el identificador de TCGA de la misma posición («SÍMBOLO|Entrez»)."""
    if len(genes_uci) != len(gene_ids_tcga):
        raise ValueError("El número de genes no coincide.")
    partes = pd.Series(gene_ids_tcga).str.split("|", n=1, expand=True)
    return pd.DataFrame({
        "gene_uci": list(genes_uci),
        "gene_id_tcga": list(gene_ids_tcga),
        "simbolo": partes[0].replace("?", np.nan).values,  # «?» = gen sin símbolo oficial en la anotación
        "entrez_id": partes[1].values,
    })


def emparejar_muestras(uci, tcga_log, tolerancia=1e-3):
    """Para cada muestra de UCI busca la muestra de TCGA más parecida (diferencia absoluta máxima).

    `uci` y `tcga_log` son muestras x genes con las columnas en el mismo orden.
    Devuelve, por muestra de UCI, la muestra de TCGA más parecida y la diferencia con ella y con la
    segunda más parecida, que mide cuánto se distingue la coincidencia del resto de muestras.
    """
    tcga = tcga_log.to_numpy(dtype=np.float64)
    filas = []
    for nombre, u in zip(uci.index, uci.to_numpy(dtype=np.float64)):
        dif = np.abs(tcga - u).max(axis=1)
        orden = np.argsort(dif)
        filas.append((nombre, tcga_log.index[orden[0]], dif[orden[0]], tcga_log.index[orden[1]], dif[orden[1]]))
    res = pd.DataFrame(filas, columns=["muestra_uci", "muestra_tcga", "dif_max_mejor",
                                       "muestra_tcga_segunda", "dif_max_segunda"])
    res["coincide"] = res["dif_max_mejor"] < tolerancia
    return res
