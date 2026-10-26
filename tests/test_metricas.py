"""Pruebas de las métricas y del bootstrap estratificado (tfm/metricas.py)."""
import numpy as np
import pytest

from tfm.metricas import bootstrap_estratificado, calcular_metricas, ic_exactitud_exacto


def test_metricas_prediccion_perfecta():
    real = [0, 0, 1, 1, 2]
    assert all(v == pytest.approx(1.0) for v in calcular_metricas(real, real).values())


def test_f1_macro_da_el_mismo_peso_a_cada_clase():
    # 9 aciertos en la clase 0 y 0 de 1 en la clase 1: exactitud 0,9 pero F1 macro mucho menor
    real = [0] * 9 + [1]
    pred = [0] * 10
    m = calcular_metricas(real, pred)
    assert m["Exactitud"] == pytest.approx(0.9)
    assert m["F1 macro"] < 0.5


def test_bootstrap_intervalo_contiene_la_estimacion():
    rng = np.random.default_rng(0)
    real = np.repeat([0, 1, 2], 30)
    pred = real.copy()
    pred[rng.choice(len(real), 9, replace=False)] = 0
    ic = bootstrap_estratificado(real, pred, n=300, semilla=1)
    assert (ic["ic_inferior"] <= ic["estimacion"]).all()
    assert (ic["estimacion"] <= ic["ic_superior"]).all()


def test_bootstrap_sin_errores_da_intervalo_degenerado():
    real = np.repeat([0, 1], 10)
    ic = bootstrap_estratificado(real, real, n=50)
    assert (ic["ic_inferior"] == 1).all() and (ic["ic_superior"] == 1).all()


def test_ic_exacto_con_cero_errores_no_es_degenerado():
    inferior, superior = ic_exactitud_exacto(801, 801)
    assert superior == 1 and 0.99 < inferior < 1
