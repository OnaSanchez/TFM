"""Pruebas de la carga, la validación y el QC de los datos (se ejecutan con `pytest`)."""
import pandas as pd
import pytest

from tfm.datos import cargar_datos, validar_datos
from tfm.qc import muestras_atipicas, porcentaje_ceros_por_muestra, resumen_qc


@pytest.fixture
def ejemplo():
    """Conjunto de datos pequeño con el mismo formato que el de UCI."""
    muestras = [f"sample_{i}" for i in range(4)]
    datos = pd.DataFrame({"gene_0": [0.0, 1.5, 2.0, 0.0],
                          "gene_1": [3.2, 0.0, 4.1, 5.0],
                          "gene_2": [0.0, 0.0, 0.0, 0.0]}, index=muestras)
    etiquetas = pd.DataFrame({"Class": ["BRCA", "COAD", "KIRC", "LUAD"]}, index=muestras)
    return datos, etiquetas


def test_datos_correctos_sin_problemas(ejemplo):
    assert validar_datos(*ejemplo) == []


def test_detecta_columna_class_ausente(ejemplo):
    datos, etiquetas = ejemplo
    problemas = validar_datos(datos, etiquetas.rename(columns={"Class": "Tipo"}))
    assert any("'Class'" in p for p in problemas)


def test_detecta_muestras_desordenadas(ejemplo):
    datos, etiquetas = ejemplo
    problemas = validar_datos(datos, etiquetas.iloc[::-1])
    assert any("mismo orden" in p for p in problemas)


def test_detecta_faltantes_negativos_y_clases_desconocidas(ejemplo):
    datos, etiquetas = ejemplo
    datos.iloc[0, 0] = None
    datos.iloc[1, 1] = -1.0
    etiquetas.iloc[2, 0] = "OV"
    problemas = " ".join(validar_datos(datos, etiquetas))
    assert "faltantes" in problemas and "negativos" in problemas and "OV" in problemas


def test_detecta_valores_no_numericos(ejemplo):
    datos, etiquetas = ejemplo
    datos["gene_1"] = ["a", "b", "c", "d"]
    assert any("no numéricos" in p for p in validar_datos(datos, etiquetas))


def test_cargar_datos_lee_ficheros_locales(ejemplo, tmp_path):
    datos, etiquetas = ejemplo
    datos.to_csv(tmp_path / "data.csv")
    etiquetas.to_csv(tmp_path / "labels.csv")
    leidos, etiquetas_leidas = cargar_datos(tmp_path)  # no descarga nada: los ficheros ya existen
    pd.testing.assert_frame_equal(leidos, datos)
    pd.testing.assert_frame_equal(etiquetas_leidas, etiquetas)


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
