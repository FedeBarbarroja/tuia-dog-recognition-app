from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from lib.services.detection_service import DetectionService

logger = logging.getLogger(__name__)


class PipelineService:
    """Etapa 4: evaluacion, optimizacion y herramientas de anotacion.

    Funciones a implementar por el estudiante:
      - evaluate_pipeline()
      - optimize_model()
      - generate_annotations(folder_path, output_format)
    """

    def __init__(
        self,
        detection: DetectionService,
        eval_path: Path,
        output_path: Path,
    ) -> None:
        self.detection = detection
        self.eval_path = eval_path
        self.output_path = output_path

    # ------------------------------------------------------------------
    # Etapa 4: funciones a implementar
    # ------------------------------------------------------------------

    def evaluate_pipeline(self) -> dict[str, float]:
        """
        Evalua el pipeline completo (deteccion + clasificacion) sobre el
        conjunto de prueba anotado manualmente en self.eval_path
        (al menos 10 imagenes complejas con bounding boxes y raza).

        Debe calcular: mAP, IoU, precision, recall y F1-Score.
        Helpers disponibles en lib.evaluation.metrics (iou, match_detections,
        average_precision, precision_recall_f1, ndcg_at_k).

        Retorna un dict con las metricas, ej:
          {"map": 0.78, "mean_iou": 0.82, "precision": 0.85,
           "recall": 0.80, "f1": 0.82}
        """
        raise NotImplementedError("Etapa 4: implementar evaluate_pipeline")

    def optimize_model(self) -> dict[str, Any]:
        """
        Optimiza el modelo de clasificacion para inferencia eligiendo UNA
        estrategia:
          - Opcion 1: cuantizacion (FP32 -> INT8).
          - Opcion 2: exportacion optimizada (ONNX / TensorRT).

        Debe comparar contra el modelo original: tiempo de inferencia, uso de
        memoria y precision. Guardar el modelo optimizado en models/.

        Retorna un dict con la comparacion, ej:
          {"strategy": "onnx", "latency_ms": {"fp32": 41.2, "optimized": 18.7},
           "accuracy": {"fp32": 0.91, "optimized": 0.90}, ...}
        """
        raise NotImplementedError("Etapa 4: implementar optimize_model")

    def generate_annotations(self, folder_path: str, output_format: str) -> str:
        """
        Procesa una carpeta de imagenes, detecta perros, clasifica razas y
        genera anotaciones automaticas en el formato indicado.

        output_format:
          - "yolo": un .txt por imagen con lineas
            `class x_center y_center width height` (coordenadas normalizadas).
          - "coco": un unico .json con claves images / annotations / categories.

        Sugerencia: reutilizar self.detection.detect_dogs y
        self.detection.classify_detected_dog.

        Retorna la ruta de la carpeta o archivo generado (dentro de
        self.output_path).
        """
        raise NotImplementedError("Etapa 4: implementar generate_annotations")
