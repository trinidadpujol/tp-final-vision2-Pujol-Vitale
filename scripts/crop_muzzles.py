"""crop_muzzles.py — Recorte zero-shot del hocico desde fotos de cara (Stage 2).

Usa GroundingDINO (open-vocabulary) vía HuggingFace transformers: detecta el morro
por PROMPT de texto, sin entrenar ni anotar nada. Para cada imagen elige la caja de
mayor score CUYO TAMAÑO sea razonable (filtro anti-error: descarta cajas gigantes
que agarran toda la cara, o minúsculas que agarran un fosa nasal/ruido), le agrega un
margen, recorta y guarda preservando <individuo>/<imagen> (así el harness de re-ID
`entries_from_folders` lo lee igual que los otros datasets).

Pensado para correr DESATENDIDO (de noche):
  - REANUDABLE: si se corta, volvé a correr el mismo comando; saltea lo ya recortado.
  - RESILIENTE: si una imagen falla, la loguea y sigue.
  - Escribe `_crop_report.json` en --out-dir con conteos y estadísticas al terminar.
  - Grilla de QC con la caja dibujada (verde) para revisar a ojo.

Uso:
    pip install "transformers==4.44.2"   # compatible con torch 2.2.2
    # prueba:
    python scripts/crop_muzzles.py --faces-dir "$HOME/data/cows_face/INDIVIDUAL SUBJECTS Data" --limit 60
    # run completo con el modelo grande (correr en tmux):
    python scripts/crop_muzzles.py --faces-dir "$HOME/data/cows_face/INDIVIDUAL SUBJECTS Data" \\
        --out-dir ~/data/cows_face_muzzle --model IDEA-Research/grounding-dino-base
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root: Path, limit: int | None) -> list[Path]:
    imgs = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)
    return imgs[:limit] if limit else imgs


def detect(processor, model, image: Image.Image, prompt: str, device: str,
           box_thr: float, text_thr: float):
    """Devuelve el dict de resultados de GroundingDINO (scores, boxes) para la imagen."""
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    try:  # el nombre del kwarg cambió entre versiones de transformers
        return processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_thr, text_threshold=text_thr,
            target_sizes=[image.size[::-1]])[0]
    except TypeError:
        return processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=box_thr, text_threshold=text_thr,
            target_sizes=[image.size[::-1]])[0]


def pick_box(res, w: int, h: int, min_frac: float, max_frac: float):
    """Mejor caja por score DENTRO del rango de tamaño. Devuelve (box, frac) o (None, motivo)."""
    if len(res["scores"]) == 0:
        return None, "no_detection"
    area = float(w * h)
    cand = []
    for i in range(len(res["scores"])):
        x0, y0, x1, y1 = (float(v) for v in res["boxes"][i])
        frac = max(0.0, x1 - x0) * max(0.0, y1 - y0) / area
        if min_frac <= frac <= max_frac:
            cand.append((float(res["scores"][i]), (x0, y0, x1, y1), frac))
    if not cand:
        return None, "bad_size"
    cand.sort(key=lambda t: t[0], reverse=True)
    return cand[0][1], cand[0][2]


def add_margin(box, w: int, h: int, margin: float):
    x0, y0, x1, y1 = box
    mw, mh = (x1 - x0) * margin, (y1 - y0) * margin
    return (max(0, int(x0 - mw)), max(0, int(y0 - mh)),
            min(w, int(x1 + mw)), min(h, int(y1 + mh)))


def save_qc_grid(previews, path: Path, cell: int = 256) -> None:
    if not previews:
        return
    cols = int(len(previews) ** 0.5 + 0.999) or 1
    rows = (len(previews) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell, rows * cell), (20, 20, 20))
    for k, im in enumerate(previews):
        im = im.copy(); im.thumbnail((cell, cell))
        grid.paste(im, ((k % cols) * cell, (k // cols) * cell))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--faces-dir", required=True)
    ap.add_argument("--out-dir", default="~/data/cows_face_muzzle")
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--prompt", default="cow muzzle. cow nose. cow snout.")
    ap.add_argument("--box-threshold", type=float, default=0.25)
    ap.add_argument("--text-threshold", type=float, default=0.20)
    ap.add_argument("--margin", type=float, default=0.12)
    ap.add_argument("--min-box-frac", type=float, default=0.005,
                    help="Descartar cajas MÁS CHICAS que esta fracción del área (ruido/fosa nasal).")
    ap.add_argument("--max-box-frac", type=float, default=0.60,
                    help="Descartar cajas MÁS GRANDES que esta fracción del área (toda la cara).")
    ap.add_argument("--overwrite", action="store_true", help="Rehacer recortes ya existentes.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--qc-n", type=int, default=30)
    ap.add_argument("--qc-out", default="~/data/cows_face_muzzle_qc.jpg")
    args = ap.parse_args()

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    faces_dir = Path(args.faces_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    if not faces_dir.is_dir():
        print(f"[ERROR] no existe {faces_dir}", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | model={args.model} | prompt='{args.prompt}'")
    print(f"filtro de tamaño: {args.min_box_frac:.3f} <= box/area <= {args.max_box_frac:.2f}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()

    images = list_images(faces_dir, args.limit)
    print(f"{len(images)} imágenes → {out_dir}\n")

    previews, sizes, fracs = [], [], []
    n_ok = n_skip = n_nodet = n_badsize = n_error = 0
    for idx, img_path in enumerate(images):
        rel = img_path.relative_to(faces_dir)
        dst = out_dir / rel
        if dst.is_file() and not args.overwrite:
            n_skip += 1
            continue
        try:
            image = Image.open(img_path).convert("RGB")
            res = detect(processor, model, image, args.prompt, device,
                         args.box_threshold, args.text_threshold)
            box, info = pick_box(res, image.width, image.height,
                                 args.min_box_frac, args.max_box_frac)
            if box is None:
                if info == "no_detection":
                    n_nodet += 1
                else:
                    n_badsize += 1
                continue
            mbox = add_margin(box, image.width, image.height, args.margin)
            crop = image.crop(mbox)
            dst.parent.mkdir(parents=True, exist_ok=True)
            crop.save(dst)
            n_ok += 1
            sizes.append(min(crop.width, crop.height))
            fracs.append(info)
            if len(previews) < args.qc_n:
                prev = image.copy()
                ImageDraw.Draw(prev).rectangle(mbox, outline=(0, 220, 0), width=6)
                previews.append(prev)
        except Exception as e:  # unattended: nunca cortar por una sola imagen
            n_error += 1
            print(f"[err] {rel}: {e}", file=sys.stderr)
            continue
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(images)} | ok={n_ok} skip={n_skip} "
                  f"nodet={n_nodet} badsize={n_badsize} err={n_error}", flush=True)

    save_qc_grid(previews, Path(args.qc_out).expanduser())

    total = len(images)
    report = {
        "faces_dir": str(faces_dir), "out_dir": str(out_dir), "model": args.model,
        "prompt": args.prompt, "total": total, "ok": n_ok, "skipped_existing": n_skip,
        "no_detection": n_nodet, "bad_size": n_badsize, "errors": n_error,
        "crop_min_side_px": {"min": min(sizes), "median": int(statistics.median(sizes)),
                             "max": max(sizes)} if sizes else None,
        "box_frac": {"min": round(min(fracs), 4), "median": round(statistics.median(fracs), 4),
                     "max": round(max(fracs), 4)} if fracs else None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_crop_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    for k in ("ok", "skipped_existing", "no_detection", "bad_size", "errors"):
        print(f"  {k:18s}: {report[k]}")
    if sizes:
        s = report["crop_min_side_px"]
        print(f"  lado menor recorte : min={s['min']} mediana={s['median']} max={s['max']} px")
        print("    → si la mediana es chica (<~150 px), poca resolución de morro (¡hallazgo!).")
    print(f"  reporte            : {out_dir / '_crop_report.json'}")
    print(f"  grilla QC          : {Path(args.qc_out).expanduser()}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())