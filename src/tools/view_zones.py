"""
Visualiza as zonas de detecção configuradas em uma câmera.
"""

import cv2
import argparse
import yaml
from pathlib import Path


def view_zones(camera_id: str, config_path: str = "config/cameras.yaml"):
    """Visualiza as zonas definidas para uma câmera."""

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Procura câmera
    camera_config = None
    for cam in config.get("cameras", []):
        if cam["id"] == camera_id:
            camera_config = cam
            break

    if not camera_config:
        print(f"Câmera não encontrada: {camera_id}")
        return

    dvr_id = camera_config.get("dvr")
    channel = camera_config.get("channel")
    dvr_config = config.get("dvrs", {}).get(dvr_id)

    if not dvr_config:
        print(f"DVR não configurado: {dvr_id}")
        return

    # Conecta ao RTSP
    rtsp_url = (
        f"rtsp://{dvr_config['username']}:{dvr_config['password']}"
        f"@{dvr_config['host']}:{dvr_config['port']}"
        f"/cam/realmonitor?channel={channel}&subtype={dvr_config.get('subtype', 1)}"
    )

    print(f"Conectando a {camera_config['name']}...")
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Falha ao conectar")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Falha ao capturar frame")
        return

    # Desenha zonas
    zones = camera_config.get("detection_zone", [])
    h, w = frame.shape[:2]

    for i, zone in enumerate(zones):
        x1, y1, x2, y2 = zone
        pt1 = (int(x1*w), int(y1*h))
        pt2 = (int(x2*w), int(y2*h))
        cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
        # Label
        cv2.putText(frame, f"Zone {i+1}", pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    print(f"\n{camera_config['name']}: {len(zones)} zona(s) definida(s)")
    for i, zone in enumerate(zones):
        print(f"  Zone {i+1}: {[f'{v:.3f}' for v in zone]}")

    cv2.imshow(f"{camera_config['name']} - Zones", frame)
    print("\nPressione qualquer tecla para fechar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Visualiza zonas de detecção")
    parser.add_argument("--camera", "-c", required=True, help="ID da câmera")
    parser.add_argument("--config", default="config/cameras.yaml", help="Arquivo de config")

    args = parser.parse_args()
    view_zones(args.camera, args.config)


if __name__ == "__main__":
    main()
