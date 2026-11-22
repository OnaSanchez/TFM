"""Reconstruye y verifica la correspondencia gene_X (UCI) -> gen de TCGA.

Uso, desde la raíz del repositorio:
    python correspondencia_genes/correspondencia_genes.py

1. Descarga la cohorte COAD de TCGA (Broad GDAC Firehose, datos públicos) y comprueba que, con el
   mismo orden de genes y la transformación log2(x + 1), cada muestra COAD de UCI coincide
   exactamente con una muestra de TCGA. Si es así, guarda la tabla de correspondencia.
2. Repite la comprobación con los otros cuatro tipos tumorales (BRCA, KIRC, LUAD y PRAD), es decir,
   con las 801 muestras, y documenta para cada una la diferencia con la segunda muestra de TCGA más
   parecida (lo bien que se distingue la coincidencia del resto).

Resultados en correspondencia_genes/resultados/:
- correspondencia_genes.csv: gene_X -> identificador de TCGA, símbolo e identificador Entrez.
- emparejamiento_muestras.csv: para cada muestra de UCI, la muestra de TCGA con la que coincide
  (código de TCGA), la diferencia máxima con ella y con la segunda más parecida.
- resumen_verificacion.csv: resumen por tipo tumoral.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from tfm.datos import CLASES, cargar_datos  # noqa: E402
from tfm.genes import cargar_firehose, emparejar_muestras, tabla_correspondencia  # noqa: E402

SALIDA = Path(__file__).resolve().parent / "resultados"
SALIDA.mkdir(exist_ok=True)

datos, etiquetas = cargar_datos(RAIZ / "Dataset")

# 1. Reconstrucción con COAD (la cohorte más pequeña)
ORDEN = ["COAD"] + [c for c in CLASES if c != "COAD"]
emparejamientos, resumen, genes_tcga = [], [], None
for cohorte in ORDEN:
    tcga = cargar_firehose(cohorte, RAIZ / "Dataset")
    if genes_tcga is None:
        genes_tcga = tcga.index
        tabla = tabla_correspondencia(datos.columns, genes_tcga)
    # Los ficheros de las cinco cohortes tienen los mismos genes en el mismo orden
    mismo_orden = tcga.index.equals(genes_tcga)

    uci = datos[etiquetas["Class"] == cohorte]
    res = emparejar_muestras(uci, np.log2(tcga.T + 1))
    res.insert(1, "clase", cohorte)
    emparejamientos.append(res)

    tipo_muestra = res["muestra_tcga"].str[13:15]  # 01 = tumor primario, 11 = tejido normal...
    segunda_mismo_paciente = res["muestra_tcga"].str[:12] == res["muestra_tcga_segunda"].str[:12]
    resumen.append({
        "clase": cohorte,
        "muestras_uci": len(uci),
        "muestras_tcga": tcga.shape[1],
        "mismo_orden_de_genes": mismo_orden,
        "coinciden": int(res["coincide"].sum()),
        "dif_max_mejor_maxima": res["dif_max_mejor"].max(),
        "dif_max_segunda_minima": res["dif_max_segunda"].min(),
        "dif_max_segunda_mediana": res["dif_max_segunda"].median(),
        "muestras_tcga_distintas": res["muestra_tcga"].nunique(),
        "pacientes_distintos": res["muestra_tcga"].str[:12].nunique(),
        "tumor_primario_01": int((tipo_muestra == "01").sum()),
        "segunda_del_mismo_paciente": int(segunda_mismo_paciente.sum()),
    })
    print(f"{cohorte}: {res['coincide'].sum()} de {len(uci)} muestras de UCI coinciden con una de las "
          f"{tcga.shape[1]} de TCGA · diferencia máxima {res['dif_max_mejor'].max():.2g} · "
          f"segunda más parecida: mínimo {res['dif_max_segunda'].min():.2f}, "
          f"mediana {res['dif_max_segunda'].median():.2f} · mismo orden de genes: {mismo_orden}")

    if cohorte == "COAD":
        if not res["coincide"].all():
            sys.exit("No se ha podido verificar la correspondencia con COAD: no se guarda la tabla.")
        tabla.to_csv(SALIDA / "correspondencia_genes.csv", index=False)
        print(f"Correspondencia verificada con COAD y guardada en resultados/correspondencia_genes.csv "
              f"({tabla['simbolo'].isna().sum()} genes sin símbolo oficial)")

emparejamientos = pd.concat(emparejamientos, ignore_index=True)
emparejamientos.to_csv(SALIDA / "emparejamiento_muestras.csv", index=False)
resumen = pd.DataFrame(resumen)
resumen.to_csv(SALIDA / "resumen_verificacion.csv", index=False)

total = int(emparejamientos["coincide"].sum())
print(f"\nTotal: {total} de {len(emparejamientos)} muestras de UCI coinciden con una muestra de TCGA; "
      f"{emparejamientos['muestra_tcga'].nunique()} muestras de TCGA distintas de "
      f"{emparejamientos['muestra_tcga'].str[:12].nunique()} pacientes.")
