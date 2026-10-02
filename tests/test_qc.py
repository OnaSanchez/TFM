"""Pruebas del control de calidad (tfm/qc.py)."""
import pandas as pd

from tfm.qc import muestras_atipicas, porcentaje_ceros_por_muestra, resumen_qc


def test_resumen_qc(ejemplo):
    resumen = resumen_qc(ejemplo[0])
    assert resumen["Valores faltantes"] == 0
    assert resumen["Genes con varianza 0"] == 1
    assert resumen["Porcentaje de ceros (%)"] == 58.3  # 7 ceros de 12 valores


def test_porcentaje_ceros_por_muestra(ejemplo):
    ceros = porcentaje_ceros_por_muestra(ejemplo[0])
    assert ceros.round(1).tolist() == [66.7, 66.7, 33.3, 66.7]


def test_muestras_atipicas_por_clase():
    valores = pd.Series([1.0, 1.1, 0.9, 1.0, 5.0, 10.0, 10.2, 9.9])
    clases = pd.Series(["A"] * 5 + ["B"] * 3)
    atipicas = muestras_atipicas(valores, clases)
    assert atipicas.tolist() == [False, False, False, False, True, False, False, False]
