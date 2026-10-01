"""Funciones del control de calidad (QC) del conjunto de datos."""
import pandas as pd


def resumen_qc(datos):
    """Indicadores básicos de calidad de la matriz de expresión (muestras x genes)."""
    return {
        "Valores faltantes": int(datos.isna().sum().sum()),
        "Muestras duplicadas": int(datos.duplicated().sum()),
        "Valor mínimo": float(datos.min().min()),
        "Valor máximo": round(float(datos.max().max()), 2),
        "Porcentaje de ceros (%)": round(float((datos == 0).mean().mean() * 100), 1),
        "Genes con varianza 0": int((datos.var() == 0).sum()),
    }


def porcentaje_ceros_por_muestra(datos):
    """Porcentaje de genes con expresión 0 en cada muestra."""
    return (datos == 0).mean(axis=1) * 100


def muestras_atipicas(valores, clases, k=1.5):
    """Marca como atípicos los valores fuera de [Q1 - k·IQR, Q3 + k·IQR] dentro de su clase.

    Es el mismo criterio que usan los diagramas de caja. Se aplica por tipo tumoral
    porque cada tipo puede tener una distribución distinta.
    """
    atipica = pd.Series(False, index=valores.index)
    for _, v in valores.groupby(clases):
        q1, q3 = v.quantile([0.25, 0.75])
        iqr = q3 - q1
        atipica[v.index] = (v < q1 - k * iqr) | (v > q3 + k * iqr)
    return atipica
