# Stage 2 — Re-identificación cross-dominio y domain adaptation (resultados)

Resumen de evidencia del Stage 2: ¿la biometría de hocico aprendida en un dataset
**transfiere** a otro dominio (otro dataset de hocico, o morros recortados de fotos de
cara)? Protocolo de re-ID: gallery/probe, split **por sesión** + **single-shot**
(cross-sesión, evita matchear fotos gemelas de la misma ráfaga). Métrica: Rank-1 (CMC).

- **Source (etiquetado):** CMPD300 (300 vacas, hocico close-up dedicado).
- **Targets (sin etiquetas de entrenamiento):** Zenodo Muzzle DB (268 vacas) y
  **morros recortados de caras** (Cows Frontal Face Dataset, 349 vacas usables).
- **Baseline de control:** ResNet-50 preentrenado en ImageNet **sin ningún entrenamiento
  en vacas** — mide cuánta performance es "gratis" por similitud visual genérica.

## 0. Recorte de hocicos desde caras (paso previo)

`scripts/crop_muzzles.py` — detección zero-shot con **GroundingDINO** (prompt de texto,
sin anotar). 2847 caras → **2844 recortes** válidos (0 no-detección, 3 descartados por
tamaño). Lado menor del recorte: **mediana ~1300 px**.

> **Hallazgo:** la resolución **NO** es el cuello de botella. Desde caras 4K, el morro
> recortado tiene alta resolución (mediana 1300 px, muy por encima del umbral de ~150 px).
> Lo que sigue no se explica por falta de píxeles.

## 1. Diagnóstico: ¿el encoder tiene señal de hocico en su PROPIO dominio?

CMPD300 test (imágenes held-out de las vacas que el encoder entrenó), single-shot:

| Encoder | Rank-1 | ImageNet | Ventaja |
|---|---|---|---|
| Clasificación (CE, `cmpd300_source`) | 0.916 | 0.768 | **+0.148** |
| ArcFace (metric learning) | 0.989 | 0.768 | **+0.221** |

> **La señal de hocico existe y es fuerte en casa.** El encoder le gana claramente a
> ImageNet en su propio dominio. La data de morro tiene detalle discriminativo usable.

## 2. Transferencia cross-dominio (el resultado central)

Todos los encoders vs. ImageNet, sobre targets con **vacas nuevas** (identidades disjuntas).

### Target: morros de cara (349 vacas)

| Encoder | Rank-1 | Ventaja vs ImageNet |
|---|---|---|
| **ImageNet** (genérico, sin vacas) | **0.724** | — (techo) |
| Clasificación (CE) | 0.724 | 0.000 |
| DANN v1 (feat256, sin warmup) | 0.665 | −0.059 |
| DANN v2 — mejor época (ep5) | 0.680 | −0.044 |
| DANN v2 — época final (ep40) | 0.637 | −0.087 |
| DANN warm-start desde `cmpd300_source` — mejor (ep5) | 0.701 | −0.023 |
| ArcFace (metric learning) | 0.596 | −0.128 |

### Target: Zenodo, cross-dataset hocico→hocico (268 vacas)

| Encoder | Rank-1 | Ventaja vs ImageNet |
|---|---|---|
| **ImageNet** | **0.894** | — |
| Clasificación (CE) | 0.864 | −0.030 |
| ArcFace | 0.834 | −0.060 |

## 3. DANN: curva de transferencia por época (target = caras)

`scripts/08_train_dann.py` (source=CMPD300, target=caras sin etiquetas, GRL + schedule
de Ganin, warmup 5, lam_max 0.5). Eval del re-ID cada 5 épocas:

| Época | `task_acc` (source) | `dom_acc` | Rank-1 (target) |
|---|---|---|---|
| 5 (warmup, λ=0) | 0.22 | 0.87 | **0.680** ← mejor |
| 10 | 0.71 | 0.66 | 0.656 |
| 15 | 0.97 | 0.44 | 0.651 |
| 20 | 0.99 | 0.40 | 0.636 |
| 25 | 0.99 | 0.42 | 0.639 |
| 30 | 0.998 | 0.48 | 0.645 |
| 40 | 0.999 | 0.51 | 0.637 |

> La transferencia **peakea temprano (ep5, encoder apenas especializado) y decae** a
> medida que `task_acc→0.99`. La alineación de dominio (`dom_acc→0.4` en ep15-20)
> **coincide con la caída**, no la revierte.

## Conclusiones

1. **La señal de hocico existe** (sección 1: +0.148 / +0.221 en el dominio propio), pero
   **no transfiere** a vacas nuevas / otro dominio (sección 2: ningún encoder supera a
   ImageNet).

2. **Transferencia negativa por sobreajuste al source.** Cuanto más se especializa el
   encoder en los hocicos del source (CE < DANN < ArcFace; y dentro de DANN, ep5 < ep40),
   **peor** transfiere. La cercanía a las features genéricas de ImageNet predice la
   transferencia; lo genérico es el techo.

3. **La domain adaptation adversarial (DANN) no rescata nada.** Alinea la distribución
   marginal de features (`dom_acc→0.5`, comprobado) pero eso **no** implica transferencia
   de la estructura discriminativa por identidad cuando las identidades son disjuntas —
   límite teórico de DANN para re-ID, mostrado empíricamente (sección 3).

**En estos datasets, la biometría de hocico no transfiere cross-dominio.** No es problema
de resolución (mediana 1300 px), ni del objetivo de entrenamiento (probamos CE, ArcFace,
DANN), ni de falta de adaptación (DANN con selección oráculo de época tampoco supera a
ImageNet). El límite es la data/dominio.

### Notas metodológicas y limitaciones

- Elegir la mejor época de DANN mirando labels del target es **oracle model selection**
  (techo, no lo disponible en producción). Aun así no supera a ImageNet → conclusión
  reforzada.
- Se probó también **warm-start del DANN desde el encoder source** (`--init-from
  cmpd300_source.pt`): es el mejor DANN (mejor época 0.701) pero **sigue sin superar a
  ImageNet** (−0.023). Confirma que ni la mejor inicialización + adaptación gana a lo
  genérico; la adaptación adversarial solo resta.
- No se probó **self-training / pseudo-labels en el target** (el complemento que el plan
  emparejaba con DANN); dada la consistencia de la evidencia, queda como trabajo futuro.
- Encoders: ResNet-50. `cmpd300_source.pt` = ft + aug (clasificación); ArcFace s=30 m=0.5;
  DANN feat_dim=512.