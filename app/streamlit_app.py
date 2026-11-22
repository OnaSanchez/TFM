import html
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# El paquete tfm (carga y validación de los datos) está en la raíz del repositorio
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from tfm.datos import CLASES, descargar_uci, validar_datos, validar_nuevas_muestras  # noqa: E402

# Los resultados los generan los notebooks; la aplicación solo los lee (no repite la validación cruzada)
RES_QC = RAIZ / "qc_eda" / "resultados"
RES_MODELADO = RAIZ / "modelado" / "resultados"
MOD_DIR = RAIZ / "modelado" / "modelos"
MODELOS = {"LR": "Regresión logística", "SVM": "SVM lineal", "RF": "Bosque aleatorio (RF)", "XGBoost": "XGBoost"}
COLORES = {"BRCA": "#a1c9f4", "COAD": "#ffb482", "KIRC": "#8de5a1", "LUAD": "#ff9f9b", "PRAD": "#d0bbff"}
COLORES_MODELO = {"LR": "#4c72b0", "SVM": "#55a868", "RF": "#c44e52", "XGBoost": "#8172b3"}
AVISO = ("Prototipo metodológico y docente: el resultado no tiene validez diagnóstica ni clínica. "
         "Los modelos no se han validado en una cohorte externa independiente.")

st.set_page_config(page_title="Tipos tumorales y expresión génica", layout="centered")

# Formato acordado con el tutor: títulos, tablas, gráficos y resultados centrados; en las tablas, encabezados
# y números centrados (los textos largos se alinean a la izquierda con estilo propio en cada tabla)
st.markdown("""
<style>
h1, h2, h3 { text-align: center; }
[data-testid="stCaptionContainer"] { text-align: center; }
[data-testid="stTable"] { display: flex; justify-content: center; }
[data-testid="stTable"] table { margin-left: auto; margin-right: auto; }
[data-testid="stTable"] th, [data-testid="stTable"] td { text-align: center !important; }
[data-testid="stTable"] tbody th { white-space: nowrap; }[data-testid="stPlotlyChart"], [data-testid="stVegaLiteChart"], [data-testid="stImage"] {
    display: flex; justify-content: center; }
[data-testid="stMetric"] { text-align: center; }
</style>
""", unsafe_allow_html=True)


# Con @st.cache_data el conjunto de datos de UCI solo se descarga la primera vez
@st.cache_data
def descargar_uci_cache():
    return descargar_uci()


# Con @st.cache_data cada tabla de resultados solo se lee una vez
@st.cache_data
def leer(ruta, **kwargs):
    return pd.read_csv(ruta, **kwargs)


@st.cache_data
def leer_metadatos():
    return json.loads((MOD_DIR / "metadatos.json").read_text(encoding="utf-8"))


# Con @st.cache_resource cada modelo se carga una sola vez y se comparte entre sesiones
@st.cache_resource
def cargar_modelo(nombre):
    return joblib.load(MOD_DIR / f"modelo_final_{nombre}.joblib")


@st.cache_data
def simbolos():
    corr = pd.read_csv(RAIZ / "correspondencia_genes" / "resultados" / "correspondencia_genes.csv").set_index("gene_uci")
    return corr["simbolo"].fillna(corr["gene_id_tcga"])


def nombre_gen(g):
    return f"{simbolos()[g]} ({g})"


def tabla_texto(df, columnas_texto):
    """st.table con las columnas de texto largo alineadas a la izquierda (el resto, centrado)."""
    # !important para que prevalezca sobre el centrado general (la regla de cada celda es más específica)
    st.table(df.style.set_properties(subset=columnas_texto, **{"text-align": "left !important"}))


def barras(datos, x, y, color, titulo_x, altura=None):
    """Gráfico de barras horizontales con las etiquetas en el orden dado (de arriba abajo)."""
    fig = px.bar(datos[::-1], x=x, y=y, orientation="h", color_discrete_sequence=[color],
                 labels={x: titulo_x, y: ""}, height=altura or 28 * len(datos) + 120)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig)


def informe_html(modelo, meta, resultado, contribuciones, metricas_modelo, origen):
    """Informe autocontenido (HTML) de una predicción: se puede abrir en el navegador e imprimir en PDF."""
    estilo = ("body{font-family:Arial,sans-serif;max-width:900px;margin:auto;color:#222}"
              "h1,h2{text-align:center}table{border-collapse:collapse;margin:12px auto}"
              "th,td{border:1px solid #bbb;padding:4px 8px;text-align:center}th{background:#eef}"
              ".aviso{border:1px solid #c44e52;background:#fdeeee;padding:8px;margin:12px 0}"
              "caption{font-weight:bold;margin-bottom:4px}")
    n_genes = f"{len(meta['genes_entrada']):,}".replace(",", ".")
    partes = [f"<html><head><meta charset='utf-8'><title>Informe de predicción</title><style>{estilo}</style></head><body>",
              "<h1>Informe de predicción del tipo tumoral</h1>",
              f"<p><b>Fecha:</b> {datetime.now():%d/%m/%Y %H:%M} · <b>Modelo:</b> {html.escape(MODELOS[modelo])}"
              f"{' (modelo final)' if modelo == meta['modelo_final'] else ''} · <b>Muestras:</b> {len(resultado)}"
              f" · <b>Origen:</b> {html.escape(origen)}</p>",
              f"<div class='aviso'>{html.escape(AVISO)}</div>",
              "<h2>Validación de las muestras</h2>",
              f"<p>Las muestras tienen los {n_genes} genes esperados (identificadores gene_X del "
              f"conjunto de UCI), sin valores faltantes ni negativos, y en la escala esperada: {html.escape(meta['escala'])}.</p>",
              "<h2>Predicciones</h2>",
              resultado.to_html(float_format=lambda v: f"{v:.3f}"),
              "<h2>Rendimiento estimado del modelo</h2>",
              "<p>Validación cruzada anidada (5 particiones externas y 3 internas) con las 801 muestras de TCGA del "
              "conjunto de UCI; intervalos de confianza del 95 % por bootstrap estratificado. No sustituye una "
              "validación externa.</p>",
              metricas_modelo.to_html(index=False)]
    if contribuciones:
        partes.append("<h2>Genes que más contribuyen a cada predicción</h2>"
                      "<p>Valor SHAP de cada gen para la clase predicha (exacto en un modelo lineal: coeficiente × "
                      "valor escalado, con la media de las muestras de entrenamiento como referencia). Positivo: "
                      "empuja hacia la clase predicha; negativo: en contra.</p>")
        for muestra, tabla in contribuciones.items():
            partes.append(tabla.to_html(index=False, float_format=lambda v: f"{v:.3f}")
                          .replace("<table", f"<table><caption>{html.escape(str(muestra))}</caption", 1))
    partes.append("<h2>Reproducibilidad</h2><p>" + ", ".join(
        f"{html.escape(k)} {html.escape(str(v))}" for k, v in meta["versiones"].items())
        + f"; semilla {meta['semilla']}. Código: https://github.com/OnaSanchez/TFM</p></body></html>")
    return "\n".join(partes)


hay_modelos = (MOD_DIR / "metadatos.json").exists()

st.sidebar.title("Tipos tumorales")
pagina = st.sidebar.radio("Sección", ["Datos", "Exploración", "Comparación de modelos", "Explicabilidad",
                                      "Predicción de nuevas muestras"])
st.sidebar.caption("Prototipo del Trabajo Final de Máster de Ona Sánchez Núñez, Máster universitario en "
                   "Bioinformática y Bioestadística (UOC, UB). Uso exclusivamente metodológico y docente: "
                   "no es una herramienta diagnóstica ni clínica.")

if pagina in ("Explicabilidad", "Predicción de nuevas muestras") and not hay_modelos:
    st.warning("Todavía no hay modelos finales. Ejecuta antes `modelado/02_modelado.ipynb`.")
    st.stop()

# ---------------------------------------------------------------------------------------------------
if pagina == "Datos":
    st.title("Clasificación de tipos tumorales a partir de la expresión génica")
    st.write("Conjunto de datos *Gene expression cancer RNA-Seq* del "
             "[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq): "
             "`data.csv` (expresión de los genes) y `labels.csv` (tipo tumoral de cada muestra).")

    opcion = st.radio("¿Qué datos quieres usar?", ["Subir mis ficheros", "Usar el conjunto de datos de UCI"])

    datos = None
    etiquetas = None

    if opcion == "Subir mis ficheros":
        archivo_datos = st.file_uploader("Fichero de expresión (data.csv)", type="csv")
        archivo_etiquetas = st.file_uploader("Fichero de etiquetas (labels.csv)", type="csv")

        # Solo se hace algo cuando se han subido los dos ficheros
        if archivo_datos is not None and archivo_etiquetas is not None:
            try:
                with st.spinner("Leyendo los datos..."):
                    datos = pd.read_csv(archivo_datos, index_col=0)
                    etiquetas = pd.read_csv(archivo_etiquetas, index_col=0)
            except Exception:
                st.error("No se han podido leer los ficheros. Comprueba que son CSV separados por comas.")
                st.stop()
    else:
        try:
            with st.spinner("Descargando el conjunto de datos de UCI (unos 70 MB)..."):
                datos, etiquetas = descargar_uci_cache()
        except Exception:
            st.error("No se ha podido descargar el conjunto de datos de UCI. Prueba más tarde o sube los ficheros.")
            st.stop()

    if datos is not None:
        # Mismas comprobaciones que en el notebook (tfm/datos.py)
        problemas = validar_datos(datos, etiquetas)
        if problemas:
            for problema in problemas:
                st.error(problema)
        else:
            st.success("¡El conjunto de datos se ha leído correctamente!")

            c1, c2 = st.columns(2)
            c1.metric("Número de muestras", f"{datos.shape[0]:,}".replace(",", "."))
            c2.metric("Número de genes", f"{datos.shape[1]:,}".replace(",", "."))

            st.subheader("Porcentaje de cada tipo tumoral")
            conteo = etiquetas["Class"].value_counts()
            porcentaje = (conteo / conteo.sum() * 100).round(1)
            # astype(str) para que la tabla muestre 37.5 y no 37.5000
            tabla = pd.DataFrame({"Muestras": conteo, "Porcentaje (%)": porcentaje.astype(str)})
            st.table(tabla)
            st.bar_chart(porcentaje)

# ---------------------------------------------------------------------------------------------------
elif pagina == "Exploración":
    st.title("Exploración de los datos")
    st.write("Cada punto es una muestra. La PCA (análisis de componentes principales) es la representación "
             "principal; el UMAP solo sirve para explorar: puede sugerir grupos, pero no los demuestra.")

    coords = leer(RES_MODELADO / "coordenadas_pca_umap.csv").set_index("muestra")
    qc = leer(RES_QC / "qc_muestras.csv").set_index("Unnamed: 0")
    errores = leer(RES_MODELADO / "analisis_errores.csv").set_index("Unnamed: 0")

    representacion = st.radio("Representación", ["PCA", "UMAP"], horizontal=True)
    colorear = st.selectbox("Colorear por", ["Tipo tumoral", "Muestra atípica en el control de calidad",
                                             "Mal clasificada por algún modelo"])

    tabla = coords.copy()
    if colorear == "Tipo tumoral":
        tabla["Grupo"] = tabla["Tipo"]
    elif colorear == "Muestra atípica en el control de calidad":
        tabla["Grupo"] = qc["Atípica"].map({True: "Atípica", False: "No atípica"})
    else:
        tabla["Grupo"] = (errores["modelos_que_fallan"] > 0).map({True: "Mal clasificada", False: "Bien clasificada"})

    x, y = ("PC1", "PC2") if representacion == "PCA" else ("UMAP1", "UMAP2")
    st.scatter_chart(tabla, x=x, y=y, color="Grupo", height=500)
    st.write("Número de muestras de cada grupo:")
    st.table(tabla["Grupo"].value_counts().rename("Muestras"))

# ---------------------------------------------------------------------------------------------------
elif pagina == "Comparación de modelos":
    st.title("Comparación de modelos")
    st.write("Rendimiento de los cuatro modelos en la validación cruzada anidada (5 particiones externas y 3 "
             "internas). Cada muestra se predice una sola vez, con un modelo que no la ha visto al entrenar "
             "(predicciones *out-of-fold*). Entre corchetes, el intervalo de confianza del 95 % por bootstrap "
             "estratificado.")

    metricas = leer(RES_MODELADO / "metricas_globales_ic95.csv")
    metricas["valor"] = [f"{f.estimacion:.3f} [{f.ic_inferior:.3f}–{f.ic_superior:.3f}]" for f in metricas.itertuples()]
    tabla = metricas.pivot(index="modelo", columns="metrica", values="valor")
    tabla = tabla.reindex(index=list(MODELOS), columns=metricas["metrica"].unique()).rename(index=MODELOS)
    st.subheader("Métricas globales")
    st.table(tabla)

    st.subheader("Errores e intervalo exacto de la exactitud")
    st.write("Cuando un modelo no comete errores, el bootstrap da un intervalo [1–1]; el intervalo exacto de "
             "Clopper-Pearson indica que la exactitud real podría ser algo menor.")
    st.table(leer(RES_MODELADO / "exactitud_ic_exacto.csv").set_index("modelo").round(4))

    st.subheader("F1 de cada tipo tumoral")
    por_clase = leer(RES_MODELADO / "metricas_por_clase.csv")
    st.table(por_clase.pivot(index="clase", columns="modelo", values="f1")[list(MODELOS)]
             .rename(columns=MODELOS).round(3))

    if (RES_MODELADO / "comparacion_modelos.csv").exists():
        st.subheader("Elección del modelo final")
        st.write("Como los cuatro modelos rinden casi igual, la regla de elección (fijada de antemano) considera "
                 "candidatos a los modelos con un F1 macro a menos de 0,01 del mejor y, entre ellos, prefiere menos "
                 "genes, más estabilidad de los 50 genes principales y menos tiempo de ajuste.")
        comp = leer(RES_MODELADO / "comparacion_modelos.csv").set_index("modelo").sort_values("orden")
        comp = comp.rename(index=MODELOS).rename(columns={"k mediana": "Genes (mediana)",
                                                          "estabilidad top 50": "Estabilidad (50 genes)",
                                                          "tiempo medio (s)": "Tiempo medio (s)",
                                                          "candidato": "Candidato", "orden": "Orden"})
        st.table(comp.round(3).astype(str))

    st.subheader("Detalle de un modelo")
    modelo = st.selectbox("Modelo", list(MODELOS), format_func=lambda m: MODELOS[m])

    predicciones = leer(RES_MODELADO / "predicciones_oof.csv")
    predicciones = predicciones[predicciones["modelo"] == modelo]
    st.write("Matriz de confusión (filas: clase real; columnas: clase predicha):")
    matriz = pd.crosstab(predicciones["real"], predicciones["pred"]).reindex(index=CLASES, columns=CLASES, fill_value=0)
    st.table(matriz)

    st.write("Número de genes e hiperparámetros elegidos en cada partición externa:")
    hiper = leer(RES_MODELADO / "hiperparametros_por_particion.csv")
    hiper = hiper[hiper["modelo"] == modelo].dropna(axis=1).drop(columns="modelo")
    st.table(hiper.set_index("particion").round(4))

    st.subheader("Muestras mal clasificadas por algún modelo")
    errores = leer(RES_MODELADO / "analisis_errores.csv").set_index("Unnamed: 0").rename_axis("Muestra")
    columnas = ["Tipo", "Atípica QC", "Grupo UMAP", "modelos_que_fallan"] + [f"pred_{m}" for m in MODELOS]
    mal = errores.loc[errores["modelos_que_fallan"] > 0, columnas].sort_values("modelos_que_fallan", ascending=False)
    st.table(mal.rename(columns={"modelos_que_fallan": "Modelos que fallan", **{f"pred_{m}": m for m in MODELOS}}))

# ---------------------------------------------------------------------------------------------------
elif pagina == "Explicabilidad":
    st.title("Explicabilidad")
    meta = leer_metadatos()
    st.write("Qué genes usan los modelos para clasificar. Los nombres de los genes proceden de la correspondencia "
             "`gene_X` → genes de TCGA reconstruida y verificada en `correspondencia_genes/`. La interpretación es "
             "orientativa: que un gen sea importante para el modelo no implica que tenga un papel causal en el tumor.")

    st.subheader("Importancia de los genes")
    imp = leer(RES_MODELADO / "importancia_genes.csv")
    c1, c2 = st.columns(2)
    modelo = c1.selectbox("Modelo", list(MODELOS), format_func=MODELOS.get)
    tipo = c2.radio("Importancia", ["modelo", "permutacion"],
                    format_func={"modelo": "Propia del modelo", "permutacion": "Por permutación"}.get)
    n = st.slider("Número de genes", 5, 40, 15)
    sel = imp[(imp["modelo"] == modelo) & (imp["tipo"] == tipo)].nlargest(n, "importancia")
    if sel.empty:
        st.info("Ningún gen tiene importancia por permutación mayor que 0 en este modelo: como hay muchos genes "
                "redundantes, permutar uno solo no cambia las predicciones.")
    else:
        titulo = ("Caída media del F1 macro al permutar el gen" if tipo == "permutacion" else
                  "|Coeficiente| medio" if modelo in ("LR", "SVM") else "Importancia media entre particiones")
        barras(sel.assign(gen=sel["gene_uci"].map(nombre_gen)), "importancia", "gen", COLORES_MODELO[modelo], titulo)

    st.subheader("SHAP de los modelos finalistas")
    st.write("Media del valor absoluto de SHAP de cada gen en las muestras de cada tipo tumoral, para la salida de "
             f"ese tipo. Finalistas: {', '.join(MODELOS[m] for m in meta['finalistas'])}; modelo final: "
             f"{MODELOS[meta['modelo_final']]}.")
    shap_res = leer(RES_MODELADO / "shap_resumen.csv")
    c1, c2 = st.columns(2)
    finalista = c1.selectbox("Modelo finalista", meta["finalistas"], format_func=MODELOS.get)
    clase = c2.selectbox("Tipo tumoral", CLASES)
    s = shap_res[(shap_res["modelo"] == finalista) & (shap_res["clase"] == clase)].nlargest(15, "shap_abs_medio")
    barras(s.assign(gen=s["gene_uci"].map(nombre_gen)), "shap_abs_medio", "gen", COLORES[clase], "|SHAP| medio")

    st.subheader("Interpretación biológica (modelo final)")
    interp = leer(RES_MODELADO / "interpretacion_biologica.csv", index_col=0)
    tabla_texto(interp, list(interp.columns))

    st.subheader("Estabilidad de la selección de genes")
    st.write("Índice de Nogueira, Sechidis y Brown (2018) entre las cinco particiones externas: 1 si se eligen "
             "siempre los mismos genes y alrededor de 0 si coinciden como al azar.")
    est = leer(RES_MODELADO / "estabilidad_nogueira.csv")
    st.table(est.pivot(index="modelo", columns="seleccion", values="estabilidad").reindex(list(MODELOS))
             .rename(index=MODELOS).round(3).astype(str).replace("nan", "—"))
    if (RES_MODELADO / "jaccard_top50_modelos.csv").exists():
        st.write("Coincidencia de los 50 genes más importantes entre modelos (índice de Jaccard):")
        st.table(leer(RES_MODELADO / "jaccard_top50_modelos.csv", index_col=0).round(2))

    st.subheader("Posible confusión por sexo")
    sens = leer(RES_MODELADO / "sensibilidad_sexo.csv").set_index(["modelo", "metrica"])
    st.write("Rendimiento de los finalistas en la misma validación cruzada anidada con y sin los genes de los "
             "cromosomas X e Y:")
    st.table(sens)

# ---------------------------------------------------------------------------------------------------
else:
    st.title("Predicción de nuevas muestras")
    meta = leer_metadatos()
    n_genes = f"{len(meta['genes_entrada']):,}".replace(",", ".")
    st.write("Sube un CSV con una fila por muestra y una columna por gen, con **los mismos "
             f"{n_genes} identificadores `gene_X` que el conjunto de UCI** y la misma escala, "
             "log2(x + 1) del RSEM normalizado de TCGA. La aplicación comprueba estrictamente el formato antes de "
             "predecir y aplica a las muestras el mismo flujo (selección de genes y escalado) con el que se "
             "entrenó el modelo.")
    st.warning(AVISO)

    modelo = st.selectbox("Modelo", list(MODELOS), index=list(MODELOS).index(meta["modelo_final"]),
                          format_func=lambda m: MODELOS[m] + (" (modelo final)" if m == meta["modelo_final"] else ""))
    origen = st.radio("Muestras", ["Subir un CSV", "Ejemplo: una muestra de cada tipo del conjunto de UCI"],
                      horizontal=True)
    nuevas = None
    if origen == "Subir un CSV":
        archivo = st.file_uploader("Muestras (CSV, muestras × genes, con el nombre de la muestra en la primera columna)",
                                   type="csv")
        if archivo is not None:
            try:
                nuevas = pd.read_csv(archivo, index_col=0)
            except Exception:
                st.error("No se ha podido leer el fichero. Comprueba que es un CSV separado por comas.")
                st.stop()
    else:
        st.caption("Estas muestras se usaron para entrenar el modelo final, así que solo sirven para ver cómo "
                   "funciona la aplicación, no para evaluar el modelo.")
        try:
            with st.spinner("Descargando el conjunto de datos de UCI (unos 70 MB)..."):
                datos, etiquetas = descargar_uci_cache()
        except Exception:
            st.error("No se ha podido descargar el conjunto de datos de UCI. Prueba más tarde o sube un CSV.")
            st.stop()
        ejemplo = etiquetas.groupby("Class").head(1).index
        nuevas = datos.loc[ejemplo]
        st.write("Tipo real:", ", ".join(f"{m}: {c}" for m, c in etiquetas.loc[ejemplo, "Class"].items()))

    if nuevas is not None:
        # Validación estricta (tfm/datos.py): mismos genes, sin faltantes ni negativos y escala log2(x + 1)
        nuevas, problemas = validar_nuevas_muestras(nuevas, meta["genes_entrada"])
        if problemas:
            for problema in problemas:
                st.error(problema)
            st.stop()
        st.success(f"Formato correcto: {len(nuevas)} muestras con los genes esperados.")

        flujo = cargar_modelo(modelo)
        pred = flujo.predict(nuevas)
        resultado = pd.DataFrame({"Tipo predicho": [CLASES[p] for p in pred]}, index=nuevas.index)
        if hasattr(flujo, "predict_proba"):
            puntuacion = pd.DataFrame(flujo.predict_proba(nuevas), index=nuevas.index, columns=CLASES)
            st.caption("Probabilidad estimada de cada tipo tumoral.")
        else:
            puntuacion = pd.DataFrame(flujo.decision_function(nuevas), index=nuevas.index, columns=CLASES)
            st.caption("La SVM no da probabilidades: se muestra su función de decisión para cada clase "
                       "(la clase predicha es la de valor más alto).")
        resultado = resultado.join(puntuacion)
        st.table(resultado.style.format(precision=3))

        # Contribución de cada gen a la predicción (modelos lineales): coeficiente × valor escalado, que es el
        # valor SHAP exacto de un modelo lineal con la media de las muestras de entrenamiento como referencia
        contribuciones = {}
        if modelo in ("LR", "SVM"):
            genes = flujo[:-1].get_feature_names_out()
            Xt = flujo[:-1].transform(nuevas)
            for i, muestra in enumerate(nuevas.index):
                k = int(pred[i])
                contrib = pd.Series(flujo[-1].coef_[k] * Xt[i], index=genes)
                contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10)
                contribuciones[muestra] = pd.DataFrame({"Gen": [nombre_gen(g) for g in contrib.index],
                                                        f"SHAP (clase {CLASES[k]})": contrib.values})
            st.subheader("Genes que más contribuyen a la predicción")
            muestra = st.selectbox("Muestra", list(nuevas.index))
            c = contribuciones[muestra]
            col = c.columns[1]
            fig = px.bar(c[::-1], x=col, y="Gen", orientation="h",
                         color=np.where(c[::-1][col] > 0, "A favor", "En contra"),
                         color_discrete_map={"A favor": "#c44e52", "En contra": "#4c72b0"},
                         labels={"Gen": "", "color": ""}, height=420)
            st.plotly_chart(fig)
        else:
            st.caption("La contribución de cada gen a una predicción individual se muestra para los modelos "
                       "lineales (finalistas); para RF y XGBoost, la importancia global está en «Explicabilidad».")

        # Módulo de informes: resumen descargable de la predicción
        st.subheader("Informe")
        metricas = leer(RES_MODELADO / "metricas_globales_ic95.csv")
        metricas = metricas[metricas["modelo"] == modelo]
        metricas_modelo = pd.DataFrame({"Métrica": metricas["metrica"],
                                        "Estimación [IC 95 %]": [f"{f.estimacion:.3f} [{f.ic_inferior:.3f}–{f.ic_superior:.3f}]"
                                                                 for f in metricas.itertuples()]})
        informe = informe_html(modelo, meta, resultado, contribuciones, metricas_modelo,
                               "CSV subido" if origen == "Subir un CSV" else "muestras de ejemplo de UCI")
        st.download_button("Descargar el informe (HTML)", informe, file_name="informe_prediccion.html",
                           mime="text/html")
        st.caption("El informe incluye las predicciones, la validación de las muestras, el rendimiento estimado del "
                   "modelo, los genes que más contribuyen (modelos lineales), las versiones de las librerías y el "
                   "aviso de uso. Se puede imprimir en PDF desde el navegador.")
