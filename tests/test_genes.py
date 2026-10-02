"""Pruebas de la correspondencia gene_X → genes de TCGA (tfm/genes.py)."""
import pandas as pd

from tfm.genes import emparejar_muestras, tabla_correspondencia


def test_tabla_correspondencia_genes():
    tabla = tabla_correspondencia(["gene_0", "gene_1"], ["?|100130426", "A1BG|1"])
    assert tabla["simbolo"].isna().tolist() == [True, False]
    assert tabla.loc[1, "simbolo"] == "A1BG" and tabla.loc[1, "entrez_id"] == "1"


def test_emparejar_muestras_detecta_coincidencia_exacta():
    tcga = pd.DataFrame([[0.0, 1.0, 2.0], [5.0, 5.0, 5.0]], index=["T1", "T2"])
    uci = pd.DataFrame([[5.0, 5.0, 5.0000001], [9.0, 9.0, 9.0]], index=["u1", "u2"])
    res = emparejar_muestras(uci, tcga)
    assert res["coincide"].tolist() == [True, False]
