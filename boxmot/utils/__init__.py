# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import os
import sys
import time
from pathlib import Path

import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
DATA = ROOT / 'data'
BOXMOT = ROOT / "boxmot"
EXAMPLES = ROOT / "tracking"
TRACKER_CONFIGS = ROOT / "boxmot" / "configs"
WEIGHTS = ROOT / "tracking" / "weights"
REQUIREMENTS = ROOT / "requirements.txt"
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of BoxMOT multiprocessing threads

# global logger
from loguru import logger

# 是否保存日志到文件
save_log = False  # 修改为 False 仅输出到控制台

logger.remove()

if save_log:
    # 创建日志目录
    LOG_DIR = ROOT / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 动态生成日志文件名
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = LOG_DIR / f"log_{current_time}.log"

    # 添加文件处理器
    logger.add(log_file, level="DEBUG", encoding="utf-8")

logger.add(sys.stderr, colorize=True, level="DEBUG")
