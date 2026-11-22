# Paso 2 · Correspondencia `gene_X` → genes de TCGA

En el conjunto de UCI los genes se llaman `gene_0` a `gene_20530`. `correspondencia_genes.py` recupera qué gen es cada uno (riesgo R3 de la PEC1):

```bash
python correspondencia_genes/correspondencia_genes.py
```

La primera vez descarga unos 240 MB de Broad GDAC Firehose (versión del 28/01/2016) en `Dataset/`.

1. Descarga la cohorte de adenocarcinoma de colon (COAD) de TCGA desde Broad GDAC Firehose (datos públicos, mismo procesamiento que UCI: RNASeqV2, RSEM normalizado, UNC).
2. Comprueba que, con el mismo orden de genes y la transformación log2(x + 1), cada una de las 78 muestras COAD de UCI coincide con una muestra de TCGA (diferencia absoluta máxima entre los 20.531 genes menor que 10⁻³). Solo si coinciden todas, guarda la tabla de correspondencia.
3. Repite la comprobación con los otros cuatro tipos tumorales (BRCA, KIRC, LUAD y PRAD) y, para cada muestra, registra la diferencia con la **segunda** muestra de TCGA más parecida, que mide lo bien que se distingue la coincidencia del resto.

## Resultados

| Tipo | Muestras UCI | Muestras TCGA | Coinciden | Diferencia máxima con la coincidente | Segunda más parecida: mínimo (mediana) |
|---|---|---|---|---|---|
| COAD | 78 | 328 | 78 | 7,2·10⁻⁶ | 7,48 (8,98) |
| BRCA | 300 | 1.212 | 299 | 7,2·10⁻⁶ | 7,74 (9,57) |
| KIRC | 146 | 606 | 146 | 7,2·10⁻⁶ | 6,99 (8,50) |
| LUAD | 141 | 576 | 141 | 7,2·10⁻⁶ | 7,58 (9,63) |
| PRAD | 136 | 550 | 136 | 7,2·10⁻⁶ | 5,75 (8,01) |
| **Total** | **801** | **3.272** | **800** | | |

- Los cinco ficheros de Firehose tienen los mismos 20.531 genes en el mismo orden.
- **800 de las 801 muestras** coinciden con una muestra de TCGA con una diferencia máxima de 7,2·10⁻⁶ (redondeo de los decimales), mientras que la segunda muestra más parecida difiere siempre en al menos 5,75 unidades de log2: la coincidencia es inequívoca. Las 800 muestras proceden de 800 pacientes distintos y todas son de tumor primario (código de tipo de muestra `01`).
- En dos muestras (`sample_19`, PRAD, y `sample_393`, BRCA) la segunda más parecida es tejido normal del mismo paciente (código `11`), lo que es esperable; la de PRAD es la que tiene la diferencia más pequeña con la segunda muestra (5,75).
- La muestra BRCA `sample_247` no tiene ninguna muestra igual en Firehose (diferencia mínima 8,76, igual que con cualquier otra muestra): probablemente es una alícuota incluida en PANCAN que no está en la versión de 2016 de Firehose. Su ausencia no afecta a la correspondencia de los genes, que ya está verificada con las otras 800 muestras.

| Fichero | Contenido |
|---|---|
| `resultados/correspondencia_genes.csv` | `gene_uci`, identificador de TCGA (`SÍMBOLO\|Entrez`), símbolo e identificador Entrez (29 genes sin símbolo oficial) |
| `resultados/emparejamiento_muestras.csv` | Para cada muestra de UCI: tipo, muestra de TCGA más parecida (código de barras), diferencia máxima con ella, segunda muestra más parecida y diferencia con ella, y si coincide |
| `resultados/resumen_verificacion.csv` | Resumen por tipo tumoral (tabla anterior) |

El tutor ha revisado el procedimiento y lo considera una evidencia convincente, así que los nombres de los genes se usan en la interpretación. La identificación de la muestra de TCGA de cada muestra de UCI permite, además, consultar los datos clínicos básicos de TCGA (sexo y tipo histológico), que se usan en el notebook 02 para contrastar el proxy de sexo y el análisis de errores (nunca como variables de los modelos).

Siguiente paso: `modelado/`.
