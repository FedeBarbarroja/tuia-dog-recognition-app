"""Genera anotaciones automaticas (YOLOv5 o COCO) para una carpeta de imagenes (Etapa 4).

Requiere haber implementado PipelineService.generate_annotations (y las
funciones de la Etapa 3 que este reutiliza).

Uso:
    python scripts/generate_annotations.py <carpeta_de_imagenes> [--format yolo|coco]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="Carpeta con imagenes a procesar.")
    parser.add_argument(
        "--format",
        dest="output_format",
        default="yolo",
        choices=("yolo", "coco"),
        help="Formato de salida de las anotaciones.",
    )
    args = parser.parse_args()

    from lib.bootstrap import build_classifier, build_detection, build_pipeline
    from lib.config import settings

    classifier = build_classifier(settings)
    detection = build_detection(settings, classifier)
    pipeline = build_pipeline(settings, detection)

    output = pipeline.generate_annotations(args.folder, args.output_format)
    print(f"Anotaciones generadas en: {output}")


if __name__ == "__main__":
    main()
