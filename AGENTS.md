# AGENTS.md

Guia para o Codex operar neste repositório.

## O que o projeto faz

Monitora câmeras IP de um DVR Intelbras/Dahua, usa o motion detection nativo do DVR
como sinal primário, baixa o segmento `.dav` do evento, pré-filtra frames por zonas
calibradas, roda YOLOv8s + SAHI para detectar animais e manda alertas no Telegram com
botões inline que movem a imagem para `detections/labeled/<label>/`.

## Entry point

`python -m src.main` — carrega `config/cameras.yaml`, inicializa o bot do Telegram e
dispara `DetectionPipeline.setup() → start()`.

## Arquitetura

Pipeline em `src/pipeline.py:DetectionPipeline`, com dois modos controlados por
`motion_source` no YAML:

- `motion_source: dvr` (produção) — um `DVREventPoller` por DVR consulta
  `mediaFileFind.cgi` a cada `poll_interval` segundos, recebe `RecordedFile`, baixa via
  `/RPC_Loadfile`, extrai frames com `frames_from_video()`, filtra com
  `filter_frames_with_motion()` (restrito às zonas) e enfileira para o
  `_inference_worker`.
- `motion_source: mog2` (fallback) — streams RTSP diretos + `MotionDetector` MOG2 em
  cada frame. Menos estável em IR; mantido para emergência.

A mesma `inference_queue` e o mesmo `_inference_worker` servem ambos os modos — o
contrato do item enfileirado é idêntico.

## Módulos ativos

| Caminho | Função |
|---|---|
| `src/pipeline.py` | Orquestrador; threads, fila, callbacks DVR, cooldowns |
| `src/camera/stream.py` | Captura RTSP (modo mog2) |
| `src/camera/motion.py` | `MotionDetector` (live MOG2) + `filter_frames_with_motion` (offline, sem warmup, contra mediana) |
| `src/detection/detector.py` | `AnimalDetector` — YOLOv8s + SAHI (tiles 480×480, overlap 15%) |
| `src/dvr/dahua_api.py` | `DahuaHTTPClient`: digest auth, `list_motion_events`, `download_file` (HEAD + Range + backoff) |
| `src/dvr/event_poller.py` | Thread daemon de polling por DVR |
| `src/dvr/offline_processor.py` | `frames_from_video(path, frame_interval_sec)` — gerador de `(ts_sec, frame)` |
| `src/dvr/state.py` | Cache persistente em `detections/dvr_state.json` (último timestamp consultado por canal) |
| `src/dvr/rtsp_opencv_extractor.py` | Extração RTSP via OpenCV (usado só pelo offline processor) |
| `src/bot/notifications.py` | `send_detection_alert(bot, chat_id, ...)` |
| `src/bot/handler.py` | Callback dos botões inline: move imagem para `labeled/<label>/` e atualiza DB |
| `src/bot/commands.py` | `/start /stats /cameras /pause /resume /snapshot /dataset` |
| `src/storage/database.py` | SQLite `detections/catdetector.db` |
| `src/storage/dataset.py` | Organização `detections/{unlabeled,labeled/*}` |
| `src/tools/zone_calibrator.py` | CLI: desenhar zonas (GUI OpenCV) |
| `src/tools/view_zones.py` | CLI: overlay das zonas sobre frame ao vivo |
| `src/tools/analyze_event.py` | CLI: baixa segmento + roda YOLO em todos os frames (debug pontual) |
| `src/tools/motion_sensitivity.py` | CLI: leitura da sensibilidade de motion do DVR |

## Decisões registradas

- **Por que não SIM Next puro (protocolo binário 37777):** o `.dav` não carrega motion
  grid e os RPC methods que retornam grids históricos são permission-blocked neste
  firmware (MHDX 3008-C, Dahua 4.001.00IB000). HTTP CGI entrega o mesmo evento com
  latência ~30s, suficiente para dataset.
- **Por que pré-filtro offline além do motion do DVR:** um segmento de 30 min @ 2s =
  900 frames. YOLO+SAHI custa ~7s por frame em CPU. O pré-filtro por zonas derruba
  para ~1% (≈10 frames), preservando o alvo. Validado em
  `scripts/validate_full_pipeline.py`.
- **Por que detectamos `bird, cat, dog, horse, sheep, cow` e não só cat:** YOLOv8s
  COCO erra cat visto de cima em IR como bird ou dog. Classificação final é do
  usuário via botão inline.
- **Sensibilidade de motion do DVR:** escrita via HTTP CGI retorna 400 (firmware
  bloqueia). Ajuste pela interface web (`http://<host>/` → Detection → Motion).

## Config (cameras.yaml)

Seções: `motion_source` (`dvr` | `mog2`), `dvrs` (dict), `cameras` (list),
`detection`, `notification`.

Campos relevantes:

- `dvrs.<name>`: `host`, `http_port`, `username`, `password`, `rpc_username`,
  `rpc_password` (usados no `/RPC_Loadfile`), `poll_interval`.
- `cameras[]`: `id`, `name`, `channel`, `dvr`, `enabled`, `motion_threshold`,
  `detection_zone` (lista de `[x1,y1,x2,y2]` 0-1).
- `detection`: `confidence_threshold`, `cooldown_seconds`, `frame_interval`,
  `force_reprocess_since` (opcional: `"YYYY-MM-DD"`, `today`, `yesterday`).

`config/cameras.example.yaml` tem o schema completo comentado.

## Validação end-to-end

```bash
python scripts/validate_full_pipeline.py
```

Usa um segmento `.dav` fixo em `detections/analysis/20260414_000329_cam_frente/` para
exercer: extração de frames → pré-filtro → YOLO → envio real ao Telegram. Smoke test
sem depender do DVR estar online.

## Deploy

```bash
sudo cp cat-detector.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now cat-detector
sudo journalctl -u cat-detector -f
```

## Secrets

`.env` precisa de `TELEGRAM_BOT_TOKEN` e `TELEGRAM_CHAT_ID`. Senhas de DVR e credenciais
RPC ficam em `config/cameras.yaml` (gitignorado — só `cameras.example.yaml` vai ao
repo).

## Convenções que o Codex deve seguir aqui

- Não recriar os módulos removidos no refactor (proxy MITM, RPC 37777 binário, motion
  metadata via `.dav`). Se precisar mexer em algo relacionado, pergunte antes.
- Não reintroduzir MOG2-sobre-RTSP como caminho principal — é fallback.
- Ao adicionar nova câmera, calibre as zonas com `zone_calibrator` antes de ligar.
- Antes de reprocessar histórico, use `force_reprocess_since` em vez de apagar
  `detections/dvr_state.json` (mais seguro e reversível).
