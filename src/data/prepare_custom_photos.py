"""Normaliza las fotos propias del grupo y genera su archivo de anotaciones.

Lee la carpeta descargada de Drive (estructura ``<Calidad>_Quality_Fruits/<fruta>_<calidad>/``)
y guarda cada foto en ``data/custom_group_photos/`` como JPEG RGB. Las variantes aumentadas
(``*_aug_*``: rotación, zoom y relleno de bordes) no son fotos independientes: si la foto
original está en la carpeta se usa solo la original; si no (limón malo y regular) se conserva
una única variante por captura y se marca en ``notes`` porque su zoom altera ``d_norm``.
Los archivos idénticos con etiquetas distintas se descartan por ambigüedad. Cada imagen se
guarda con:

* orientación corregida según EXIF y metadatos EXIF eliminados (privacidad, indicador PI1);
* lado mayor reducido a ``MAX_SIDE`` px (el extractor trabaja a 128x128 y así el repositorio
  no carga 1.4 GB de fotos HEIC/JPG originales);
* nombre uniforme ``<fruta>_<calidad>_<nnn>.jpg``.

Las etiquetas de fruta y calidad salen de la carpeta de origen. La talla no viene en Drive y
el enunciado solo exige 30-50 fotos anotadas a mano, así que el script marca con
``size_sample = 1`` una muestra estratificada de ``SIZE_SAMPLE_PER_FRUIT`` fotos por fruta
del catálogo (sin variantes aumentadas, repartida entre calidades) y genera
``size_sample_sheet.png`` con esas fotos para que el grupo llene ``size_label`` solo en esas
filas. Si el CSV ya existe, se conservan los valores escritos a mano (``size_label``,
``capture_distance_cm``, ``annotator``) de cada foto de origen.

Uso:
    python src/data/prepare_custom_photos.py <carpeta_proyectoAPO>
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "custom_group_photos"
ANNOTATIONS = OUT_DIR / "annotations_template.csv"
SIZE_SHEET = OUT_DIR / "size_sample_sheet.png"
MAX_SIDE = 512
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic"}
CATALOG_FRUITS = ["Apple", "Banana", "Guava", "Lime", "Orange", "Pomegranate"]
SIZE_SAMPLE_PER_FRUIT = 12
SIZE_SAMPLE_SEED = 42
SIZE_SAMPLE_MIN_SIDE = 200
SIZE_SAMPLE_EXCLUDE = {  # varias frutas en la misma foto: no hay una sola talla que anotar
    "Regular_Quality_Fruits/orange_regular/1794858712542.jpg",
    "Regular_Quality_Fruits/orange_regular/1794858712584.jpg",
}
# Fotogramas recortados (banana_001_*, banana_002_*): no son fotos tomadas con el protocolo.
SIZE_SAMPLE_EXCLUDE_PATTERN = r"/banana_00\d_\d{3}\.jpg$"
MANUAL_COLUMNS = ["size_label", "capture_distance_cm", "annotator"]

QUALITY_BY_FOLDER = {"bad": "Bad", "good": "Good", "regular": "Medium"}
FRUIT_ALIASES = {
    "apple": "Apple", "manzana": "Apple",
    "banana": "Banana", "banano": "Banana",
    "guava": "Guava", "guayaba": "Guava",
    "lime": "Lime", "limon": "Lime", "lemon": "Lime",
    "orange": "Orange", "orage": "Orange", "naranja": "Orange",
    "pomegranate": "Pomegranate", "granada": "Pomegranate",
    "granadilla": "Granadilla",  # fuera de las 6 frutas del dataset de referencia
}
COLUMNS = ["filename", "fruit_type", "quality_label", "size_label",
           "capture_distance_cm", "annotator", "size_sample", "notes"]


def select_size_sample(df: pd.DataFrame) -> pd.Series:
    """Marca SIZE_SAMPLE_PER_FRUIT fotos por fruta del catálogo, repartidas entre calidades.

    Se excluyen las fotos en las que la talla no se puede juzgar: variantes aumentadas (su zoom
    cambia el tamaño aparente), recortes de menos de SIZE_SAMPLE_MIN_SIDE px o fotogramas que
    no se tomaron con el protocolo (SIZE_SAMPLE_EXCLUDE_PATTERN) y las fotos con varias frutas
    listadas en SIZE_SAMPLE_EXCLUDE.
    """
    min_side = [min(Image.open(OUT_DIR / name).size) for name in df["filename"]]
    origin = df["notes"].str.split(" | ", regex=False).str[0].str.removeprefix("origen: ")
    eligible = df[df["fruit_type"].isin(CATALOG_FRUITS)
                  & ~df["notes"].str.contains("variante aumentada")
                  & (pd.Series(min_side, index=df.index) >= SIZE_SAMPLE_MIN_SIDE)
                  & ~origin.isin(SIZE_SAMPLE_EXCLUDE)
                  & ~origin.str.contains(SIZE_SAMPLE_EXCLUDE_PATTERN, regex=True)]
    chosen = []
    for _, fruit_rows in eligible.groupby("fruit_type"):
        groups = {q: g.sample(frac=1.0, random_state=SIZE_SAMPLE_SEED).index.tolist()
                  for q, g in fruit_rows.groupby("quality_label")}
        picked = []
        while len(picked) < SIZE_SAMPLE_PER_FRUIT and any(groups.values()):
            for quality in sorted(groups):  # turno rotativo entre calidades
                if groups[quality] and len(picked) < SIZE_SAMPLE_PER_FRUIT:
                    picked.append(groups[quality].pop(0))
        chosen.extend(picked)
    return df.index.isin(chosen).astype(int)


def write_size_sheet(df: pd.DataFrame, tile: int = 220, cols: int = 6) -> None:
    """Hoja de contactos con las fotos de la muestra de talla y su nombre de archivo."""
    sample = df[df["size_sample"] == 1].sort_values("filename")
    rows = -(-len(sample) // cols)
    label_h = 22
    sheet = Image.new("RGB", (cols * tile, rows * (tile + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=15)
    for i, name in enumerate(sample["filename"]):
        x, y = (i % cols) * tile, (i // cols) * (tile + label_h)
        with Image.open(OUT_DIR / name) as img:
            img.thumbnail((tile - 8, tile - 8))
            sheet.paste(img, (x + (tile - img.width) // 2, y + (tile - img.height) // 2))
        draw.text((x + 6, y + tile), name, fill="black", font=font)
    sheet.save(SIZE_SHEET)


def labels_for(path: Path, root: Path) -> tuple[str, str]:
    """Deduce (fruta, calidad) a partir de la carpeta o, si no hay subcarpeta, del nombre."""
    rel = path.relative_to(root)
    quality_dir = rel.parts[0].split("_")[0].lower()
    quality = QUALITY_BY_FOLDER[quality_dir]
    hint = rel.parts[1] if len(rel.parts) > 2 else path.stem
    for token in re.split(r"[_\-\s]+", hint.lower()):
        if token in FRUIT_ALIASES:
            return FRUIT_ALIASES[token], quality
    raise ValueError(f"No se pudo deducir la fruta de {rel}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="Carpeta proyectoAPO descargada de Drive")
    args = parser.parse_args()
    root = args.source.resolve()

    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXT)
    capture = lambda p: (p.parent, p.stem.lower().split("_aug_")[0])  # noqa: E731
    with_original = {capture(p) for p in files if "_aug_" not in p.stem.lower()}
    selected, augmented_kept, kept_captures = [], set(), set()
    for path in files:
        if "_aug_" not in path.stem.lower():
            selected.append(path)
        elif capture(path) not in with_original and capture(path) not in kept_captures:
            kept_captures.add(capture(path))
            augmented_kept.add(path)
            selected.append(path)

    by_hash: dict[str, list[Path]] = {}
    for path in selected:
        by_hash.setdefault(hashlib.md5(path.read_bytes()).hexdigest(), []).append(path)
    duplicates, conflicts = [], []
    usable = []
    for group in by_hash.values():
        if len({labels_for(p, root) for p in group}) > 1:
            conflicts.extend(group)
        else:
            usable.append(group[0])
            duplicates.extend(group[1:])

    records, counters = [], {}
    for path in sorted(usable):
        fruit, quality = labels_for(path, root)
        key = (fruit, quality)
        counters[key] = counters.get(key, 0) + 1
        name = f"{fruit.lower()}_{quality.lower()}_{counters[key]:03d}.jpg"

        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
            img.save(OUT_DIR / name, "JPEG", quality=92)  # sin exif= -> sin metadatos

        note = f"origen: {path.relative_to(root).as_posix()}"
        if path in augmented_kept:
            note += " | variante aumentada (sin original disponible)"
        records.append({
            "filename": name, "fruit_type": fruit, "quality_label": quality, "size_label": "",
            "capture_distance_cm": "", "annotator": "", "notes": note,
        })

    for old in OUT_DIR.glob("*.jpg"):
        if old.name not in {r["filename"] for r in records}:
            old.unlink()

    df = pd.DataFrame(records, columns=COLUMNS)
    if ANNOTATIONS.exists():  # no perder lo que el grupo ya anotó a mano
        previous = pd.read_csv(ANNOTATIONS, dtype=str).fillna("")
        if "notes" in previous:
            previous = previous.assign(origin=previous["notes"].str.split(" | ", regex=False).str[0])
            manual = previous.drop_duplicates("origin").set_index("origin")
            origin = df["notes"].str.split(" | ", regex=False).str[0]
            for col in MANUAL_COLUMNS:
                if col in manual:
                    df[col] = origin.map(manual[col]).fillna("")
    df["size_sample"] = select_size_sample(df)
    df.to_csv(ANNOTATIONS, index=False, encoding="utf-8")
    write_size_sheet(df)

    summary = df.groupby(["fruit_type", "quality_label"]).size().unstack(fill_value=0)
    print(f"Fotos leídas: {len(files)} | aumentadas descartadas: {len(files) - len(selected)} | "
          f"aumentadas conservadas (única por captura): {len(augmented_kept)} | "
          f"duplicadas: {len(duplicates)} | conflicto de etiqueta: {len(conflicts)} | "
          f"guardadas: {len(records)}")
    for path in conflicts:
        print(f"  conflicto descartado: {path.relative_to(root).as_posix()}")
    print(summary)
    sample = df[df["size_sample"] == 1]
    print(f"\nMuestra para anotar talla: {len(sample)} fotos (size_sample = 1) -> {SIZE_SHEET.name}")
    print(sample.groupby(["fruit_type", "quality_label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
