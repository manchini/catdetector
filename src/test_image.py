"""
Script de teste — roda detecção em uma imagem estática.
Útil para validar que o YOLO funciona antes de conectar as câmeras.

Uso:
    python -m src.test_image caminho/da/imagem.jpg
    python -m src.test_image caminho/da/imagem.jpg --model models/catdetector.pt
"""

import argparse
import sys
import cv2
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test")


def main():
    parser = argparse.ArgumentParser(description="Testa detecção em imagem estática")
    parser.add_argument("image", help="Caminho da imagem")
    parser.add_argument("--model", default=None, help="Caminho do modelo .pt (default: yolov8s.pt)")
    parser.add_argument("--confidence", type=float, default=0.3, help="Threshold de confiança")
    args = parser.parse_args()

    image_path = args.image
    if not Path(image_path).exists():
        print(f"Arquivo não encontrado: {image_path}")
        sys.exit(1)

    # Carrega imagem
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Não foi possível carregar: {image_path}")
        sys.exit(1)

    print(f"Imagem carregada: {frame.shape[1]}x{frame.shape[0]}")

    # Inicializa detector
    model_name = Path(args.model).name if args.model else "YOLOv8s"
    print(f"Carregando modelo {model_name}...")
    from .detection.detector import AnimalDetector

    detector = AnimalDetector(model_path=args.model, confidence=args.confidence)

    # Roda detecção
    print("Rodando detecção...")
    detections = detector.detect(frame)

    if not detections:
        print("\n❌ Nenhum animal detectado.")
        print("Dicas:")
        print("  - Tente reduzir confidence para 0.2")
        print("  - O animal pode estar muito pequeno no frame")
        print("  - Imagens IR (P&B) podem ter precisão menor")
    else:
        print(f"\n✅ {len(detections)} detecção(ões):\n")
        for i, det in enumerate(detections):
            print(f"  [{i+1}] {det['class_name']}")
            print(f"      Confiança: {det['confidence']:.1%}")
            print(f"      BBox: {det['bbox']}")
            print()

    # Salva imagem anotada
    annotated = detector.draw_detections(frame, detections)
    output_path = str(Path(image_path).stem) + "_detected.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"Imagem anotada salva em: {output_path}")

    # Teste de movimento (bonus)
    print("\n--- Teste do Detector de Movimento ---")
    from .camera.motion import MotionDetector

    md = MotionDetector(threshold=3000)
    # Precisa de pelo menos 2 frames, simula com a mesma imagem
    md.detect(frame)  # frame 1 (calibração)
    # Cria um frame levemente diferente para simular movimento
    frame2 = frame.copy()
    frame2[100:200, 100:200] = 255  # bloco branco
    result = md.detect(frame2)
    print(f"  Movimento detectado: {result['motion']}")
    print(f"  Score: {result['score']}")


if __name__ == "__main__":
    main()
