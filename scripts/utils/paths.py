# -*- coding: utf-8 -*-
"""
Path helpers for locating source/destination media.
"""

from __future__ import annotations

import datetime
import os


def original_video_path_of(output_path: str) -> str:
    base = os.path.dirname(output_path) or os.getcwd()
    name_no_ext = os.path.splitext(os.path.basename(output_path))[0]
    return os.path.join(base, f"{name_no_ext}_original.mp4").replace("\\", "/")


def guess_workdir(output_path: str) -> str:
    base = os.path.dirname(output_path) or os.getcwd()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(base, ts).replace("\\", "/")
    os.makedirs(work_dir, exist_ok=True)
    return work_dir

