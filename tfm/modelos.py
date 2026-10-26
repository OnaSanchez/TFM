"""Modelos y validación cruzada (CV) anidada.

Todo lo que aprende de los datos (quitar los genes sin variación, elegir los genes, escalar y
entrenar) va dentro de un Pipeline de scikit-learn. Así se ajusta solo con la parte de
entrenamiento de cada partición y no hay fuga de información (data leakage).
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
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

# Rejillas de hiperparámetros. El número de genes (anova__k) va de menos a más: si dos opciones
# empatan, GridSearchCV se queda con la primera, es decir, con la de menos genes (parsimonia).
K_GENES = [50, 200, 1000]
REJILLAS = {
    "LR": {"anova__k": K_GENES, "modelo__C": [0.05, 0.5], "modelo__l1_ratio": [0.2, 0.8]},
    "SVM": {"anova__k": K_GENES, "modelo__C": [0.001, 0.01, 0.1]},
    "RF": {"anova__k": K_GENES, "modelo__max_features": ["sqrt", "log2"]},
    "XGBoost": {"anova__k": K_GENES, "modelo__max_depth": [2, 4]},
}


def crear_pipeline(nombre, semilla=1234):
    """Genes sin variación → selección ANOVA de k genes → escalado (solo modelos lineales) → modelo.

    Los modelos usan un solo núcleo: lo que se reparte entre los núcleos es la CV.
    """
    if nombre == "LR":
        # Elastic Net combina L1 (lleva coeficientes a cero) y L2 (los reduce sin eliminarlos)
        modelo = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=5000, tol=1e-3,
                                    class_weight="balanced", random_state=semilla)
    elif nombre == "SVM":
        modelo = LinearSVC(dual="auto", max_iter=20000, class_weight="balanced", random_state=semilla)
    elif nombre == "RF":
        modelo = RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=1, random_state=semilla)
    elif nombre == "XGBoost":
        modelo = XGBClassifier(n_estimators=300, learning_rate=0.1, subsample=0.8, colsample_bytree=0.5,
                               tree_method="hist", n_jobs=1, random_state=semilla)
    else:
        raise ValueError(f"Modelo desconocido: {nombre}")

    pasos = [("varianza", VarianceThreshold(0.0)), ("anova", SelectKBest(f_classif))]
    if nombre in ("LR", "SVM"):
        pasos.append(("escalado", StandardScaler()))  # a los árboles no les afecta la escala
    pasos.append(("modelo", modelo))
    return Pipeline(pasos)


def cv_anidada(nombre, X, y, cv_externa, cv_interna, n_jobs=-1, semilla=1234):
    """CV anidada de un modelo. `y` son las clases como enteros (0 a 4).

    En cada partición externa, la CV interna elige el número de genes y los hiperparámetros (con el
    F1 macro), el mejor flujo se reentrena con toda la parte de entrenamiento y se predice la parte
    de prueba, que no se ha usado para nada antes.

    Devuelve dos tablas:
    - oof: para cada muestra, la clase real, la predicha (out-of-fold), la partición externa en la
      que se predijo y las puntuaciones de cada clase (probabilidades o, en la SVM, la función de
      decisión).
    - particiones: hiperparámetros elegidos, F1 macro interno y tiempo de cada partición externa.
    """
    oof = pd.DataFrame(index=X.index)
    oof["real"] = y
    oof["pred"] = -1
    oof["particion"] = -1
    puntuaciones = np.full((len(y), len(np.unique(y))), np.nan)
    filas = []

    for i, (entrena, prueba) in enumerate(cv_externa.split(X, y)):
        inicio = time.perf_counter()
        busqueda = GridSearchCV(crear_pipeline(nombre, semilla), REJILLAS[nombre], cv=cv_interna,
                                scoring="f1_macro", n_jobs=n_jobs)
        busqueda.fit(X.iloc[entrena], y[entrena])
        tiempo = time.perf_counter() - inicio

        mejor = busqueda.best_estimator_
        X_prueba = X.iloc[prueba]
        oof.iloc[prueba, oof.columns.get_loc("pred")] = mejor.predict(X_prueba)
        oof.iloc[prueba, oof.columns.get_loc("particion")] = i
        if hasattr(mejor, "predict_proba"):
            puntuaciones[prueba] = mejor.predict_proba(X_prueba)
        else:
            puntuaciones[prueba] = mejor.decision_function(X_prueba)

        filas.append({"modelo": nombre, "particion": i, **busqueda.best_params_,
                      "f1_macro_interno": busqueda.best_score_, "tiempo_s": tiempo})
        print(f"{nombre} · partición {i + 1}: {busqueda.best_params_} · "
              f"F1 macro interno {busqueda.best_score_:.4f} · {tiempo:.0f} s")

    for c in range(puntuaciones.shape[1]):
        oof[f"punt_{c}"] = puntuaciones[:, c]
    return oof, pd.DataFrame(filas)
