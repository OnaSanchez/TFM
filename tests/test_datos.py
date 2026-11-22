"""Pruebas de la carga y la validación de los datos (tfm/datos.py)."""
import pandas as pd

from tfm.datos import cargar_datos, validar_datos, validar_nuevas_muestras


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



def test_validar_nuevas_muestras_reordena_columnas(ejemplo):
    datos, _ = ejemplo
    ordenados, problemas = validar_nuevas_muestras(datos[["gene_2", "gene_0", "gene_1"]], datos.columns)
    assert problemas == [] and list(ordenados.columns) == list(datos.columns)


def test_validar_nuevas_muestras_detecta_genes_que_faltan_o_sobran(ejemplo):
    datos, _ = ejemplo
    _, problemas = validar_nuevas_muestras(datos.drop(columns="gene_1").assign(otro=1.0), datos.columns)
    texto = " ".join(problemas)
    assert "Faltan 1 genes" in texto and "otro" in texto


def test_validar_nuevas_muestras_detecta_escala_incorrecta(ejemplo):
    datos, _ = ejemplo
    _, problemas = validar_nuevas_muestras(datos * 1000, datos.columns)
    assert any("log2" in p for p in problemas)
