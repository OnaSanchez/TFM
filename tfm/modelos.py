"""Modelos y validación cruzada (CV) anidada.

Todo lo que aprende de los datos (quitar los genes sin variación, elegir los genes, escalar y
entrenar) va dentro de un Pipeline de scikit-learn. Así se ajusta solo con la parte de
entrenamiento de cada partición y no hay fuga de información (data leakage).
"""
import time

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

NOMBRES = {
    "LR": "Regresión logística multinomial penalizada",
    "SVM": "SVM lineal",
    "RF": "Bosque aleatorio (RF)",
    "XGBoost": "XGBoost",
}

# Rejillas de hiperparámetros. GridSearchCV recorre la rejilla con «anova__k» como parámetro más
# externo (orden alfabético) y, en caso de empate, se queda con la primera combinación: así, si
# varias opciones rinden igual, se elige el menor número de genes (criterio de parsimonia).
K_GENES = [50, 200, 1000]
REJILLAS = {
    "LR": {"anova__k": K_GENES, "modelo__C": [0.05, 0.5], "modelo__l1_ratio": [0.2, 0.8]},
    "SVM": {"anova__k": K_GENES, "modelo__C": [0.001, 0.01, 0.1]},
    "RF": {"anova__k": K_GENES, "modelo__max_features": ["sqrt", "log2"]},
    "XGBoost": {"anova__k": K_GENES, "modelo__max_depth": [2, 4]},
}


def crear_modelo(nombre, semilla=1234):
    """Clasificador sin ajustar. Los modelos usan un solo núcleo: la paralelización se hace en la CV."""
    if nombre == "LR":
        # Elastic Net combina L1 (lleva coeficientes a cero) y L2 (los reduce sin eliminarlos)
        return LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=5000, tol=1e-3,
                                  class_weight="balanced", random_state=semilla)
    if nombre == "SVM":
        return LinearSVC(dual="auto", max_iter=20000, class_weight="balanced", random_state=semilla)
    if nombre == "RF":
        return RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=1,
                                      random_state=semilla)
    if nombre == "XGBoost":
        return XGBClassifier(n_estimators=300, learning_rate=0.1, subsample=0.8, colsample_bytree=0.5,
                             tree_method="hist", n_jobs=1, random_state=semilla)
    raise ValueError(f"Modelo desconocido: {nombre}")


def crear_pipeline(nombre, semilla=1234):
    """Flujo completo: genes sin variación → selección ANOVA (k genes) → [escalado] → modelo.

    El escalado solo se aplica a los modelos lineales; a los árboles no les afecta.
    """
    pasos = [("varianza", VarianceThreshold(0.0)), ("anova", SelectKBest(f_classif))]
    if nombre in ("LR", "SVM"):
        pasos.append(("escalado", StandardScaler()))
    pasos.append(("modelo", crear_modelo(nombre, semilla)))
    return Pipeline(pasos)


def genes_seleccionados(pipeline):
    """Genes que llegan al modelo final del flujo, en el mismo orden que sus coeficientes."""
    return np.asarray(pipeline[:-1].get_feature_names_out())


def importancia_modelo(pipeline):
    """Importancia propia del modelo para cada gen seleccionado.

    Modelos lineales: media del valor absoluto de los coeficientes de las cinco clases (los genes
    están estandarizados, así que son comparables). Árboles: importancia por impureza (RF) o por
    ganancia (XGBoost).
    """
    modelo = pipeline[-1]
    if hasattr(modelo, "coef_"):
        valores = np.abs(modelo.coef_).mean(axis=0)
    else:
        valores = modelo.feature_importances_
    return pd.Series(valores, index=genes_seleccionados(pipeline))


def puntuaciones(pipeline, X):
    """Probabilidades de cada clase (o, en la SVM, la función de decisión)."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)
    return pipeline.decision_function(X)


def ajustar_con_cv_interna(nombre, X, y, cv_interna, n_jobs=-1, semilla=1234):
    """Ajusta los hiperparámetros con CV interna estratificada (F1 macro) y reentrena con todo X."""
    busqueda = GridSearchCV(crear_pipeline(nombre, semilla), REJILLAS[nombre], cv=cv_interna,
                            scoring="f1_macro", n_jobs=n_jobs, refit=True)
    return busqueda.fit(X, y)


def cv_anidada(nombre, X, y, cv_externa, cv_interna, n_jobs=-1, semilla=1234,
               permutaciones=10, verbose=True):
    """CV anidada de un modelo.

    `y` son las clases codificadas como enteros (0..4). Las mismas particiones externas se usan para
    todos los modelos. Para cada partición externa: CV interna para elegir hiperparámetros y número
    de genes → mejor flujo reentrenado con la parte de entrenamiento → evaluación en la parte de prueba,
    que no se ha usado en ningún momento de la optimización.

    Devuelve un diccionario con:
    - `oof`: predicción out-of-fold de cada muestra (clase y puntuaciones de cada clase).
    - `particiones`: hiperparámetros elegidos, F1 macro interno y tiempo de cada partición externa.
    - `genes`: genes seleccionados (ANOVA) en cada partición externa.
    - `importancia_modelo` e `importancia_permutacion`: importancia de cada gen en cada partición
      (la de permutación se calcula en la parte de prueba de la partición externa, con F1 macro).
    """
    n_clases = len(np.unique(y))
    pred = np.full(len(y), -1)
    punt = np.full((len(y), n_clases), np.nan)
    particion = np.full(len(y), -1)
    filas, genes, imp_modelo, imp_perm = [], [], [], []

    for i, (entrena, prueba) in enumerate(cv_externa.split(X, y)):
        inicio = time.perf_counter()
        busqueda = ajustar_con_cv_interna(nombre, X.iloc[entrena], y[entrena], cv_interna, n_jobs, semilla)
        tiempo = time.perf_counter() - inicio
        mejor = busqueda.best_estimator_

        X_prueba = X.iloc[prueba]
        pred[prueba] = mejor.predict(X_prueba)
        punt[prueba] = puntuaciones(mejor, X_prueba)
        particion[prueba] = i

        seleccion = genes_seleccionados(mejor)
        genes.append(seleccion)
        imp_modelo.append(importancia_modelo(mejor))

        # Importancia por permutación sobre los genes que llegan al modelo, en la parte de prueba
        X_t = mejor[:-1].transform(X_prueba)
        perm = permutation_importance(mejor[-1], X_t, y[prueba], scoring="f1_macro",
                                      n_repeats=permutaciones, random_state=semilla, n_jobs=n_jobs)
        imp_perm.append(pd.Series(perm.importances_mean, index=seleccion))

        filas.append({"modelo": nombre, "particion": i, **busqueda.best_params_,
                      "f1_macro_interno": busqueda.best_score_, "tiempo_s": tiempo})
        if verbose:
            print(f"{nombre} · partición {i + 1}: {busqueda.best_params_} · "
                  f"F1 macro interno {busqueda.best_score_:.4f} · {tiempo:.0f} s")

    oof = pd.DataFrame(punt, index=X.index, columns=[f"punt_{c}" for c in range(n_clases)])
    oof.insert(0, "particion", particion)
    oof.insert(0, "pred", pred)
    oof.insert(0, "real", y)
    return {"oof": oof, "particiones": pd.DataFrame(filas), "genes": genes,
            "importancia_modelo": imp_modelo, "importancia_permutacion": imp_perm}


def matriz_seleccion(listas_genes, todos_los_genes):
    """Matriz binaria (particiones × genes): 1 si el gen se ha seleccionado en esa partición."""
    todos = pd.Index(todos_los_genes)
    Z = np.zeros((len(listas_genes), len(todos)), dtype=int)
    for i, lista in enumerate(listas_genes):
        Z[i, todos.get_indexer(lista)] = 1
    return Z


def estabilidad_nogueira(Z):
    """Estabilidad de la selección de variables de Nogueira, Sechidis y Brown (2018).

    `Z` es una matriz binaria M × d (M selecciones, d variables). Vale 1 si todas las selecciones son
    idénticas y alrededor de 0 si coinciden lo mismo que selecciones al azar del mismo tamaño. Admite
    selecciones de tamaño distinto.
    """
    Z = np.asarray(Z, dtype=float)
    M, d = Z.shape
    p = Z.mean(axis=0)              # frecuencia con la que se selecciona cada variable
    k_medio = p.sum()               # número medio de variables seleccionadas
    var_f = M / (M - 1) * p * (1 - p)
    return 1 - var_f.mean() / ((k_medio / d) * (1 - k_medio / d))


def top_genes(importancias, n=50):
    """Los n genes más importantes de cada partición (lista de series de importancia)."""
    return [imp.sort_values(ascending=False).index[:n] for imp in importancias]
