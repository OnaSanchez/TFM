"""Métricas de clasificación e intervalos de confianza por bootstrap estratificado."""
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score,
                             recall_score)

METRICAS = {
    "F1 macro": lambda r, p: f1_score(r, p, average="macro"),
    "Exactitud equilibrada": balanced_accuracy_score,
    "Exactitud": accuracy_score,
    "Precisión macro": lambda r, p: precision_score(r, p, average="macro", zero_division=0),
    "Sensibilidad macro": lambda r, p: recall_score(r, p, average="macro", zero_division=0),
}


def calcular_metricas(real, pred):
    """Métricas globales (F1 macro como métrica principal)."""
    return {nombre: funcion(real, pred) for nombre, funcion in METRICAS.items()}


def ic_exactitud_exacto(aciertos, n, nivel=0.95):
    """Intervalo de confianza exacto (Clopper-Pearson) de la exactitud.

    Complementa al bootstrap cuando un modelo no comete ningún error: entonces todas las réplicas
    bootstrap dan 1 y el intervalo [1, 1] no refleja la incertidumbre, mientras que el exacto sí
    (con 0 errores en n muestras, el límite inferior es aproximadamente 1 - 3,7/n).
    """
    return tuple(binomtest(int(aciertos), int(n)).proportion_ci(confidence_level=nivel, method="exact"))


def bootstrap_estratificado(real, pred, n=2000, semilla=1234, nivel=0.95):
    """Intervalo de confianza percentil de cada métrica por bootstrap estratificado.

    Se remuestrean con reemplazo las predicciones out-of-fold dentro de cada clase real, de modo que
    cada réplica mantiene el número de muestras de cada tipo tumoral. El intervalo refleja la
    incertidumbre debida a la muestra evaluada (las 801 predicciones); no incluye la variabilidad
    que tendría reentrenar los modelos con otros datos.
    """
    real, pred = np.asarray(real), np.asarray(pred)
    rng = np.random.default_rng(semilla)
    grupos = [np.where(real == c)[0] for c in np.unique(real)]
    replicas = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in grupos])
        replicas.append(calcular_metricas(real[idx], pred[idx]))
    replicas = pd.DataFrame(replicas)
    alfa = (1 - nivel) / 2
    return pd.DataFrame({"estimacion": pd.Series(calcular_metricas(real, pred)),
                         "ic_inferior": replicas.quantile(alfa),
                         "ic_superior": replicas.quantile(1 - alfa)})
