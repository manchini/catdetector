# Cat Detector

Sistema de detecção de animais em câmeras de segurança usando DVR Intelbras/Dahua,
YOLOv8s + SAHI e alertas no Telegram com classificação por botões.

Licença: [MIT](LICENSE)

## Como funciona

```
DVR Intelbras (motion events via HTTP CGI)
    │
    ▼ a cada 30s
[DVREventPoller] ──► baixa segmento .dav novo
    │
    ▼
[frames_from_video] ──► extrai frames a cada N segundos
    │
    ▼
[filter_frames_with_motion] ──► pré-filtro: só frames com movimento nas zonas calibradas
    │
    ▼
[YOLOv8s + SAHI] ──► sliced inference nos frames sobreviventes
    │
    ▼
[SQLite + filesystem] ──► detecção persistida em detections/unlabeled/
    │
    ▼
[Telegram] ──► foto anotada + botões (Gato / Meu Cachorro / Outro / Falso Alarme)
    │
    ▼
[dataset labeled/] ──► imagem movida para a pasta do label clicado
```

A pipeline usa o motion nativo do DVR (muito mais estável em IR do que MOG2 sobre RTSP).
Segmentos baixados passam por um pré-filtro de movimento **restrito às zonas calibradas**
antes de virar inferência YOLO — tipicamente 1% dos frames sobrevivem.

## Requisitos

- Python 3.10+
- DVR Intelbras/Dahua com HTTP CGI habilitado (porta 80)
- Bot do Telegram + chat_id

O modelo `yolov8s.pt` é baixado automaticamente pelo Ultralytics na primeira execução
(~22 MB). Não versionamos pesos no repositório.

## Instalação

```bash
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
cp config/cameras.example.yaml config/cameras.yaml
# edite .env com TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
# edite config/cameras.yaml com IPs, credenciais e câmeras
```

**Não commite** `.env` nem `config/cameras.yaml` — os dois já estão no `.gitignore`.

## Rodar

```bash
python -m src.main
```

Deploy contínuo via systemd:

```bash
sudo cp cat-detector.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now cat-detector
sudo journalctl -u cat-detector -f
```

## Comandos do bot

`/start` `/stats` `/cameras` `/pause <cam_id>` `/resume <cam_id>` `/snapshot` `/dataset`

## Calibração de zonas de detecção

Zonas restringem a análise de movimento a regiões específicas — evita falso positivo de
árvore, sombra, carro na rua. Armazenadas em `config/cameras.yaml` em coordenadas
normalizadas 0–1.

```bash
# desenhar zonas (clique+arraste; ENTER confirma, ESC pula)
python -m src.tools.zone_calibrator --camera cam_frente

# ver as zonas de uma câmera sobre a imagem ao vivo
python -m src.tools.view_zones --camera cam_frente
```

Formato no YAML:

```yaml
cameras:
  - id: cam_frente
    channel: 1
    dvr: dvr_casa
    enabled: true
    motion_threshold: 3000
    detection_zone:
      - [0.22, 0.51, 0.34, 0.66]   # [x1, y1, x2, y2], frações 0-1
      - [0.37, 0.59, 0.52, 0.77]
```

## Config (`cameras.yaml`)

Seções obrigatórias:

- `motion_source: dvr` — usa polling HTTP do DVR (alternativa: `mog2` para MOG2-RTSP).
- `dvrs:` — dict `dvr_name → {host, http_port, username, password, rpc_username,
  rpc_password, poll_interval}`. `rpc_*` são usados para o endpoint `/RPC_Loadfile`
  (download do `.dav`).
- `cameras:` — lista `{id, name, channel, dvr, enabled, motion_threshold,
  detection_zone}`.
- `detection:` — `confidence_threshold`, `cooldown_seconds`, `frame_interval`.
  Opcional: `force_reprocess_since: "YYYY-MM-DD"` (ou `today`/`yesterday`) para
  reprocessar eventos históricos.

Veja `config/cameras.example.yaml` para o schema completo.

## Ferramentas úteis

- `python -m src.tools.config_gui` — GUI web local para configurar câmeras e revisar eventos.
- `python -m src.tools.analyze_event --dvr dvr_casa --channel 1 --at "2026-04-14 00:03:29" --window-min 5`
  Baixa o segmento que contém o instante indicado, roda YOLO em todos os frames e salva
  resultados em `detections/analysis/`.
- `python -m src.tools.motion_sensitivity --reduce 30 --dry-run`
  Lê as sensibilidades de motion do DVR (leitura apenas — escrita é bloqueada pelo
  firmware; use a interface web).
- `python -m src.tools.dvr_health` — checagem de conectividade e download do DVR.
- `python -m src.test_image caminho/da/imagem.jpg` — smoke test do YOLO numa foto estática.
- `python scripts/validate_full_pipeline.py`
  Roda o pipeline inteiro sobre um `.dav` local (útil para smoke test sem DVR).

## Testes

```bash
python -m unittest discover -s tests -v
```

Os testes unitários usam stubs e não precisam do DVR nem do modelo YOLO.

## Estrutura

```
catdetector/
├── config/
│   └── cameras.example.yaml   # copie para cameras.yaml (não versionado)
├── src/
│   ├── camera/                # stream RTSP + motion (MOG2 + offline)
│   ├── detection/             # YOLOv8s + SAHI
│   ├── dvr/                   # HTTP API Dahua, event poller, download .dav
│   ├── bot/                   # Telegram: notificações, handler, comandos
│   ├── storage/               # SQLite + dataset filesystem
│   ├── tools/                 # CLIs: calibrador, GUI, analyze_event, health
│   ├── pipeline.py            # orquestrador
│   └── main.py                # entry point
├── scripts/
│   ├── validate_full_pipeline.py
│   └── monitor_loop.py
├── tests/
├── detections/                # runtime local (imagens, db, .dav) — gitignored
└── cat-detector.service
```
