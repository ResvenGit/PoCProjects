# -*- coding: utf-8 -*-
"""
Subprocess helpers shared across services.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from typing import Callable, List, Optional, Tuple


def run_command(cmd: str) -> Tuple[bool, str]:
    """Run a command and capture output."""
    try:
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        completed = subprocess.run(
            cmd,
            shell=True,
            check=True,
            creationflags=flags,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        return True, (completed.stdout or "") + (completed.stderr or "")
    except subprocess.CalledProcessError as exc:
        return False, (exc.stdout or "") + (exc.stderr or "")


def run_command_stream(
    cmd: str,
    timeout_sec: Optional[float] = None,
    on_stderr_line: Optional[Callable[[str], None]] = None,
    on_stdout_line: Optional[Callable[[str], None]] = None,
) -> Tuple[int, str, str]:
    """Stream stdout/stderr while running a command."""
    print(f"[run] {cmd}", flush=True)
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    out_lines: List[str] = []
    err_lines: List[str] = []
    out_done = threading.Event()
    err_done = threading.Event()

    def _pump(pipe, collector, cb, done_evt, name):
        try:
            for line in iter(pipe.readline, ""):
                collector.append(line)
                if cb:
                    cb(line)
                if name == "stderr":
                    print(line.rstrip(), file=sys.stderr, flush=True)
                else:
                    print(line.rstrip(), flush=True)
        finally:
            try:
                pipe.close()
            finally:
                done_evt.set()

    t_out = threading.Thread(
        target=_pump, args=(proc.stdout, out_lines, on_stdout_line, out_done, "stdout"), daemon=True
    )
    t_err = threading.Thread(
        target=_pump, args=(proc.stderr, err_lines, on_stderr_line, err_done, "stderr"), daemon=True
    )
    t_out.start()
    t_err.start()

    start = time.time()
    while True:
        if timeout_sec is not None and (time.time() - start) > timeout_sec:
            try:
                proc.kill()
            finally:
                raise TimeoutError(f"Process timed out after {timeout_sec:.1f}s")
        if proc.poll() is not None:
            break
        time.sleep(0.05)

    out_done.wait(2.0)
    err_done.wait(2.0)
    return proc.returncode, "".join(out_lines), "".join(err_lines)
