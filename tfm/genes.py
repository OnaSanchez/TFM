"""Correspondencia entre los identificadores genéricos de UCI (gene_X) y los genes de TCGA.

La ficha de UCI indica que las variables conservan el orden de la fuente original PANCAN
(Synapse syn4301332, fichero unc.edu_PANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv). Ese fichero
requiere registro, pero los mismos datos de TCGA (RNASeqV2, RSEM normalizado, procesado por UNC)
son públicos en Broad GDAC Firehose. Aquí se usa la cohorte COAD para comprobar la correspondencia:
si gene_i es el gen i del fichero de TCGA, cada muestra COAD de UCI debe coincidir exactamente
con una muestra COAD de TCGA.
"""
import io
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL_FIREHOSE_COAD = (
    "https://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/COAD/20160128/"
    "gdac.broadinstitute.org_COAD.Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__"
    "RSEM_genes_normalized__data.Level_3.2016012800.0.0.tar.gz"
)


def cargar_firehose_coad(carpeta="Dataset"):
    """Matriz genes x muestras (RSEM normalizado) de TCGA COAD. La descarga (26 MB) la primera vez."""
    ruta = Path(carpeta) / "firehose_COAD_RSEM_genes_normalized.txt"
    if not ruta.exists():
        print("Descargando TCGA COAD de Broad GDAC Firehose...")
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL_FIREHOSE_COAD) as respuesta:
            tar = tarfile.open(fileobj=io.BytesIO(respuesta.read()))
        miembro = next(m for m in tar.getmembers() if m.name.endswith(".data.txt"))
        ruta.write_bytes(tar.extractfile(miembro).read())
    # La segunda fila de cabecera («gene_id / normalized_count») no contiene datos
    return pd.read_csv(ruta, sep="\t", index_col=0, skiprows=[1]).apply(pd.to_numeric)


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
    Devuelve, por muestra de UCI, la diferencia con la mejor y con la segunda mejor muestra de TCGA.
    """
    filas = []
    for nombre, u in zip(uci.index, uci.values):
        dif = np.abs(tcga_log.values - u).max(axis=1)
        orden = np.argsort(dif)
        filas.append((nombre, dif[orden[0]], dif[orden[1]]))
    res = pd.DataFrame(filas, columns=["muestra_uci", "dif_max_mejor", "dif_max_segunda"])
    res["coincide"] = res["dif_max_mejor"] < tolerancia
    return res
