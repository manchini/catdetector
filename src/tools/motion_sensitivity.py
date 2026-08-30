#!/usr/bin/env python3
"""
Ferramenta de calibração de sensibilidade de motion no DVR.
Reduz sensibilidade (aumenta detecção) em X%.

Uso:
    python src/tools/motion_sensitivity.py --reduce 30 --dry-run
    python src/tools/motion_sensitivity.py --reduce 30
    python src/tools/motion_sensitivity.py --dvr dvr_casa --reduce 20
"""

import logging
import sys
from pathlib import Path

import yaml

from src.dvr.dahua_api import DahuaHTTPClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def calibrate_motion(
    dvr_id: str | None = None,
    reduction_percent: int = 30,
    dry_run: bool = False,
    config_path: str = "config/cameras.yaml",
) -> bool:
    """
    Reduz sensibilidade (aumenta detecção) em X%.

    Args:
        dvr_id: ID do DVR (ex: 'dvr_casa'). Se None, trata todos.
        reduction_percent: Quanto reduzir em % (30 = reduzir 30%, tornar 30% mais sensível).
        dry_run: Se True, mostra mudanças mas não escreve.
        config_path: Caminho para config/cameras.yaml.

    Returns:
        True se todas as operações completaram com sucesso.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dvrs_config = config.get("dvrs", {})

    if not dvrs_config:
        logger.error("Nenhum DVR configurado em config/cameras.yaml")
        return False

    if dvr_id:
        if dvr_id not in dvrs_config:
            logger.error(f"DVR '{dvr_id}' não encontrado. Disponíveis: {list(dvrs_config.keys())}")
            return False
        dvrs_config = {dvr_id: dvrs_config[dvr_id]}

    all_success = True

    for did, dcfg in dvrs_config.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processando: {did} ({dcfg['host']})")
        logger.info(f"{'=' * 60}")

        client = DahuaHTTPClient(
            dcfg["host"],
            dcfg.get("username", "admin"),
            dcfg.get("password", ""),
            dvr_name=did,
            port=dcfg.get("http_port", 80),
        )

        if not client.check_connection():
            logger.error(f"{did}: falha na conexão")
            all_success = False
            continue

        zones = client.get_motion_zones()
        if not zones:
            logger.warning(f"{did}: nenhuma zona de motion encontrada")
            continue

        logger.info(f"{did}: {len(zones)} zona(s) encontrada(s)\n")

        for zone in zones:
            old_sens = zone.sensitivity
            new_sens = int(old_sens * (1 - reduction_percent / 100))
            new_sens = max(0, min(100, new_sens))

            zone.sensitivity = new_sens

            change = old_sens - new_sens
            change_str = f"{change:+d}" if change != 0 else "0"
            logger.info(
                f"  CH{zone.channel} {zone.name:20s} "
                f"sensitivity: {old_sens:3d} → {new_sens:3d} ({change_str})"
            )

        if dry_run:
            logger.info(f"\n{did}: (dry-run, nenhuma mudança escrita)")
        else:
            if client.set_motion_zones(zones):
                logger.info(f"\n{did}: ✓ configuração escrita com sucesso")
            else:
                logger.error(f"\n{did}: ✗ erro ao escrever configuração")
                all_success = False

    logger.info(f"\n{'=' * 60}")
    if all_success:
        logger.info("✓ Operação concluída com sucesso")
    else:
        logger.error("✗ Alguns DVRs falharam")
    logger.info(f"{'=' * 60}\n")

    return all_success


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ajustar sensibilidade de detecção de motion no DVR"
    )
    parser.add_argument(
        "--dvr",
        help="ID do DVR (ex: dvr_casa). Se omitido, trata todos.",
    )
    parser.add_argument(
        "--reduce",
        type=int,
        default=30,
        help="Reduzir sensibilidade em X%% (padrão 30). "
        "Valores menores = mais sensível.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostrar mudanças propostas sem escrever no DVR",
    )

    args = parser.parse_args()

    success = calibrate_motion(args.dvr, args.reduce, args.dry_run)
    sys.exit(0 if success else 1)
