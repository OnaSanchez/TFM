# Paso 2 · Correspondencia `gene_X` → genes de TCGA

En el conjunto de UCI los genes se llaman `gene_0` a `gene_20530`. `correspondencia_genes.py` recupera qué gen es cada uno (riesgo R3 de la PEC1):

```bash
python correspondencia_genes/correspondencia_genes.py
```

1. Descarga la cohorte de adenocarcinoma de colon (COAD) de TCGA desde Broad GDAC Firehose (datos públicos, mismo procesamiento que UCI).
2. Comprueba que, con el mismo orden de genes y la transformación log2(x + 1), cada una de las 78 muestras COAD de UCI coincide con una muestra de TCGA.
3. Solo si coinciden todas, guarda la tabla.

| Fichero | Contenido |
|---|---|
| `resultados/correspondencia_genes.csv` | `gene_uci`, identificador de TCGA (`SÍMBOLO\|Entrez`), símbolo e identificador Entrez |

La confirmación definitiva de esta correspondencia está pendiente de revisar con el tutor, así que los nombres de los genes se usan con prudencia.

Siguiente paso: `modelado/`.
