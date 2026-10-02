"""Reconstruye y verifica la correspondencia gene_X (UCI) -> gen de TCGA.

Uso, desde la raíz del repositorio:
    python correspondencia_genes/correspondencia_genes.py

Descarga la cohorte COAD de TCGA (Broad GDAC Firehose, datos públicos) y comprueba que, con el mismo
orden de genes y la transformación log2(x + 1), cada muestra COAD de UCI coincide exactamente con una
muestra de TCGA. Si la comprobación es correcta, guarda la tabla en
correspondencia_genes/resultados/correspondencia_genes.csv.
"""
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from tfm.datos import cargar_datos  # noqa: E402
from tfm.genes import cargar_firehose_coad, emparejar_muestras, tabla_correspondencia  # noqa: E402

datos, etiquetas = cargar_datos(RAIZ / "Dataset")
tcga = cargar_firehose_coad(RAIZ / "Dataset")
print(f"UCI: {datos.shape[1]} genes · TCGA COAD: {tcga.shape[0]} genes y {tcga.shape[1]} muestras")

tabla = tabla_correspondencia(datos.columns, tcga.index)

# Verificación con las muestras COAD: mismo orden de genes y escala log2(x + 1)
coad_uci = datos[etiquetas["Class"] == "COAD"]
tcga_log = np.log2(tcga.T + 1)
res = emparejar_muestras(coad_uci, tcga_log)
print(f"Muestras COAD de UCI que coinciden con una muestra de TCGA: {res['coincide'].sum()} de {len(res)}")
print(f"Diferencia máxima con la muestra coincidente: {res['dif_max_mejor'].max():.6f}")
print(f"Diferencia mínima con la segunda muestra más parecida: {res['dif_max_segunda'].min():.2f}")

if not res["coincide"].all():
    sys.exit("No se ha podido verificar la correspondencia: no se guarda la tabla.")

salida = Path(__file__).resolve().parent / "resultados" / "correspondencia_genes.csv"
salida.parent.mkdir(exist_ok=True)
tabla.to_csv(salida, index=False)
print(f"Correspondencia verificada. Tabla guardada en {salida.relative_to(RAIZ)}")
print(f"Genes sin símbolo oficial («?» en la anotación de TCGA): {tabla['simbolo'].isna().sum()}")
