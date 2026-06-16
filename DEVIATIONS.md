# DEVIATIONS.md — Desviaciones respecto del paper

Este archivo registra **toda** diferencia entre lo que hacemos nosotros y la receta
del paper de referencia:

> Li, G.; Erickson, G.E.; Xiong, Y. (2022). *Individual Beef Cattle Identification
> Using Muzzle Images and Deep Learning Techniques.* Animals 12(11):1453.

La replicación se evalúa por **fidelidad**, no solo por el número. Separar claramente
"lo que dice el paper" de "lo que decidimos nosotros".

---

## Estado: el dataset coincide con el paper

Verificado en Fase 0 (`scripts/00_inspect_data.py`, conteo por archivos sobre las
4923 imágenes reales):

| | Paper | Real (medido) |
|---|---|---|
| Nº de clases | 268 | 268 ✓ |
| Total imágenes | 4923 | 4923 ✓ |
| Min imágenes/clase | 4 | 4 ✓ |
| Max imágenes/clase | 70 | 70 ✓ |
| Media | — | 18.4 |

**No hay desviación en los datos.** La receta del paper (`N_max=70`, distribución
4–70) aplica tal cual.

---

## D1 — Aclaración: hay 8 clases con 4 imágenes (no 4)

**Paper:** menciona "las 4 vacas con solo 4 imágenes (IDs 2100, 4549, 5355, 5925)".

**Real:** hay **8** clases con exactamente 4 imágenes:
`2100, 3420, 4549, 5208, 5355, 5630, 5925, 8050`. Las 4 que nombra el paper son un
subconjunto de estas.

**Impacto:** ninguno sobre la receta. Solo importa al reportar accuracy por clase:
revisar las **8**, no 4, como las candidatas a tirar el promedio. No es una
desviación, es una corrección factual de la nota del paper.

---

## D2 — Valores no especificados por el paper (decididos por nosotros)

No son desviaciones (el paper no los fija), pero se documentan para reproducibilidad:

- **Batch size:** 32 (bajar si hay OOM con 300×300 + VGG16_BN).
- **Semillas de réplica:** plan original `(0, 1, 2, 3, 4)` = 5 corridas. **En el run de
  Kaggle usamos 3 semillas `(0, 1, 2)`** por el presupuesto de tiempo de la T4 (ver D3):
  VGG16_BN a 300×300 toma ~43 min/corrida medido → 3 var × 5 sem ≈ 10.7 h, no entra junto
  con ResNet en el límite de 12 h de "Save & Run All". Con 3 var × 3 sem ≈ 6.5 h sí entra.
  Sigue habiendo media ± std (3 muestras); el nº de réplicas fue decisión nuestra, no de
  la receta del paper. `config.REPLICATE_SEEDS` queda en 5; el notebook pasa `--seeds 0 1 2`.
- **Semilla de split:** 42, fija; los splits se guardan a disco y se reusan.
- **num_workers** del DataLoader: 4.
- **Weighted CE `N_max`:** los pesos se calculan desde el split de **train** (sin
  fuga): `N_i` = conteo en train, `N_max` = máximo de esos conteos (**≈46**, no 70,
  porque train ≈65% de la clase de 70 imágenes). `N_max` es solo un factor de escala
  global de la loss; el peso relativo entre clases (`N_max/N_i`) se preserva, que es
  lo que importa. El paper usa el literal 70 (máximo del dataset completo). Override
  disponible en `config.py` (`WCE_NMAX_OVERRIDE = 70`) para reproducir el valor exacto
  del paper y comparar.

---

## D3 — Hardware: T4 en vez de P100 (no afecta la receta)

**Paper / plan:** GPU **P100** (la que usó el paper; `plan.md` §5 la prefería).

**Real:** la imagen actual de Kaggle trae un PyTorch compilado **sin soporte Pascal
(sm_60)**, así que la P100 tira `CUDA error: no kernel image is available for execution`
en el primer forward. Entrenamos en **GPU T4 ×2** (Turing, sm_75), que sí está en las
capabilities soportadas por ese PyTorch (sm_70–sm_120). Se usa una sola T4 (`cuda:0`).

**Impacto en resultados:** ninguno. Es solo el dispositivo de cómputo; la receta
(resolución, normalización, optimizador, losses, épocas, seeds) no cambia. Puede variar
levemente el tiempo por época y el redondeo numérico de bajo nivel, no la metodología.

---

## D4 — Data augmentation ADITIVA (corrección hacia el método del paper)

**Brillo: NO hay desviación.** El paper dice textual: *"the brightness factor was set from
0.2 to 0.5 (minimum = 0, maximum = 1)"*. O sea oscurece al 20–50% a propósito. Nuestro
`ColorJitter(brightness=(0.2,0.5))` lo hace igual. El valor es fiel.

**Bug que tuvimos (y corregimos):** el paper describe la augmentation como *"a technique to
create synthesized images and increase limited datasets"* → **crea imágenes sintéticas y
AGRANDA el dataset, manteniendo los originales** (sobre todo para las clases con pocas
imágenes). Nuestra primera versión usaba el augmentation online típico de PyTorch, que
**REEMPLAZA** cada imagen por una versión aumentada en cada época. Con el brillo 0.2–0.5,
eso dejaba TODO el train oscuro (medido: ~3.5× más oscuro que val/test) y el modelo nunca
veía las imágenes a brillo real → mismatch sistemático train/test → la variante `ce_aug`
**colapsó** (test ~0.20 vs ~0.96 de `ce`; train 0.83 / val 0.17).

**Fix (faithful):** augmentation **aditiva**. Se mantienen los originales con transform
limpio (resize + ToTensor) y se **agregan** copias sintéticas aumentadas. Por clase se
expande hasta `min(AUG_TARGET_CAP, AUG_FACTOR·N_i)` (= `min(20, 2·N_i)`): las clases con
pocas imágenes se duplican o suben hasta 20, las de ≥20 no se tocan. Resultado: train
**3187 → 4675 imgs (1.47×)**, 234/268 clases expandidas. Implementado en
`dataset.make_train_loader` / `build_augmented_entries`.

Las copias sintéticas se **precomputan a disco** una vez (`scripts/04_precompute_aug.py`
→ `outputs/aug_cache/`), igual que el paper la describe como *"preprocessing step"*: el
conjunto sintético queda FIJO (no varía por época), reproducible e inspeccionable. Si el
cache no existe (p.ej. smoke), se cae a augmentation online equivalente.

**Qué es nuestro vs del paper:** el *método* (aditivo, expandir clases chicas) es del
paper; el *multiplicador exacto* `min(20, 2·N_i)` es decisión nuestra (el paper no da el
número de imágenes sintéticas por clase) — elegido para acotar el costo de cómputo en la
T4. Override en `config.py` (`AUG_TARGET_CAP`, `AUG_FACTOR`).

---

## D5 — (reservado)

Anotar acá cualquier desviación futura (resolución, normalización, optimizador,
descongelar backbone, etc.) con su justificación **antes** de aplicarla.

---

> Nota histórica: una primera inspección por `unzip -l | grep cattle_[0-9]+` reportó
> erróneamente 8–140 (media 36.7). Causa: cada ruta contiene el ID dos veces
> (`.../cattle_0100/cattle_0100_x.jpg`), duplicando el conteo. El conteo por archivos
> es el válido y coincide con el paper.
