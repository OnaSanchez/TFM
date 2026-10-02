"""Datos de ejemplo compartidos por las pruebas (pytest carga este fichero automáticamente)."""
import pandas as pd
import pytest


@pytest.fixture
def ejemplo():
    """Conjunto de datos pequeño con el mismo formato que el de UCI."""
    muestras = [f"sample_{i}" for i in range(4)]
    datos = pd.DataFrame({"gene_0": [0.0, 1.5, 2.0, 0.0],
                          "gene_1": [3.2, 0.0, 4.1, 5.0],
                          "gene_2": [0.0, 0.0, 0.0, 0.0]}, index=muestras)
    etiquetas = pd.DataFrame({"Class": ["BRCA", "COAD", "KIRC", "LUAD"]}, index=muestras)
    return datos, etiquetas
