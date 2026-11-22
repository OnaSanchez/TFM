"""Pruebas de los flujos de los modelos, la CV anidada y la estabilidad (tfm/modelos.py)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from tfm import modelos
from tfm.modelos import estabilidad_nogueira, matriz_seleccion


@pytest.fixture
def simulados():
    """Cinco clases separables: cada clase expresa más un bloque de 4 genes; el resto es ruido."""
    rng = np.random.default_rng(0)
    n_por_clase, n_genes = 20, 60
    X = rng.normal(5, 1, size=(5 * n_por_clase, n_genes)).clip(0)
    y = np.repeat(np.arange(5), n_por_clase)
    for c in range(5):
        X[y == c, c * 4:(c + 1) * 4] += 6
    X = pd.DataFrame(X, columns=[f"gene_{i}" for i in range(n_genes)])
    X["gene_constante"] = 0.0  # lo tiene que quitar VarianceThreshold
    return X, y


def test_estabilidad_selecciones_identicas_vale_1():
    Z = np.array([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]])
    assert estabilidad_nogueira(Z) == pytest.approx(1.0)


def test_estabilidad_selecciones_disjuntas_es_negativa():
    Z = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert estabilidad_nogueira(Z) < 0


def test_matriz_seleccion():
    Z = matriz_seleccion([["b", "c"], ["a"]], ["a", "b", "c"])
    assert Z.tolist() == [[0, 1, 1], [1, 0, 0]]


@pytest.mark.parametrize("nombre", ["LR", "SVM", "RF", "XGBoost"])
def test_pipeline_quita_genes_constantes_y_selecciona_k(simulados, nombre):
    X, y = simulados
    flujo = modelos.crear_pipeline(nombre).set_params(anova__k=8)
    flujo.fit(X, y)
    genes = modelos.genes_seleccionados(flujo)
    assert len(genes) == 8 and "gene_constante" not in genes
    # Los genes informativos (bloques de las cinco clases) son los primeros 20
    assert sum(int(g.split("_")[1]) < 20 for g in genes) == 8
    assert len(modelos.importancia_modelo(flujo)) == 8


def test_cv_anidada_usa_las_mismas_particiones_y_predice_todas_las_muestras(simulados, monkeypatch):
    X, y = simulados
    monkeypatch.setitem(modelos.REJILLAS, "SVM", {"anova__k": [8, 20], "modelo__C": [0.1]})
    externa = StratifiedKFold(3, shuffle=True, random_state=1)
    interna = StratifiedKFold(2, shuffle=True, random_state=1)
    res = modelos.cv_anidada("SVM", X, y, externa, interna, n_jobs=1, permutaciones=2, verbose=False)
    oof = res["oof"]
    assert (oof["pred"] >= 0).all()  # cada muestra se ha predicho una vez
    esperado = np.empty(len(y), dtype=int)
    for i, (_, prueba) in enumerate(externa.split(X, y)):
        esperado[prueba] = i
    assert (oof["particion"].values == esperado).all()
    assert (oof["pred"] == oof["real"]).mean() > 0.9
    assert len(res["genes"]) == 3 and len(res["particiones"]) == 3
