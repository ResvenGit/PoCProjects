# -*- coding: utf-8 -*-
"""
YouTube download helpers built around yt-dlp.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

from ..utils.process import run_command

try:
    from yt_dlp import YoutubeDL  # type: ignore

    HAS_YTDLP = True
except Exception:  # pragma: no cover - optional dependency
    YoutubeDL = None
    HAS_YTDLP = False


class YoutubeDownloader:
    def __init__(self) -> None:
        self.supports_python_api = HAS_YTDLP and YoutubeDL is not None

    def download_original_and_vtt(
        self,
        youtube_url: str,
        original_path: str,
        progress_cb: Optional[Callable[[dict], None]] = None,
    ) -> Tuple[bool, Optional[str]]:
        if not youtube_url:
            return False, None
        if self.supports_python_api:
            return self._download_with_python(youtube_url, original_path, progress_cb)
        ok, _ = run_command(
            f'yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" '
            f'-o "{original_path}" "{youtube_url}"'
        )
        return ok, self._find_latest_vtt(os.path.dirname(original_path))

    def _download_with_python(
        self,
        youtube_url: str,
        original_path: str,
        progress_cb: Optional[Callable[[dict], None]] = None,
    ) -> Tuple[bool, Optional[str]]:
        def _hook(status: dict):
            if progress_cb and status.get("status") == "downloading":
                progress_cb(status)

        ydl_opts = {
            "outtmpl": original_path,
            "merge_output_format": "mp4",
            "quiet": True,
            "progress_hooks": [_hook] if progress_cb else [],
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["ko", "ko-KR", "en", "en-US"],
            "skip_download": False,
        }
        try:
            assert YoutubeDL is not None  # mypy guard
            with YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(youtube_url, download=True)
            return True, self._find_latest_vtt(os.path.dirname(original_path))
        except Exception:
            return False, None

    def download_vtt_only(self, youtube_url: str, out_dir: str) -> Optional[str]:
        if not youtube_url:
            return None
        os.makedirs(out_dir, exist_ok=True)
        lang_groups = [
            ["ko", "ko-KR"],
            ["en", "en-US"],
        ]
        for langs in lang_groups:
            if self._download_subtitles(youtube_url, out_dir, langs):
                return self._find_latest_vtt(out_dir)
        return None

    def _download_subtitles(self, youtube_url: str, out_dir: str, langs) -> bool:
        if self.supports_python_api:
            ydl_opts = {
                "quiet": True,
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": langs,
                "outtmpl": os.path.join(out_dir, "%(title)s.%(ext)s"),
            }
            try:
                assert YoutubeDL is not None
                with YoutubeDL(ydl_opts) as ydl:
                    ydl.extract_info(youtube_url, download=True)
                return True
            except Exception:
                return False
        lang_str = ",".join(langs)
        ok, _ = run_command(
            'yt-dlp --skip-download --write-auto-sub --write-sub '
            f'--sub-langs "{lang_str}" -o "{os.path.join(out_dir, "%(title)s.%(ext)s")}" '
            f'"{youtube_url}"'
        )
        return bool(ok)

    @staticmethod
    def _find_latest_vtt(folder: str) -> Optional[str]:
        candidates = []
        for root, _, files in os.walk(folder):
            for filename in files:
                if filename.lower().endswith(".vtt"):
                    path = Path(root) / filename
                    candidates.append(path)
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(candidates[0])


_DEFAULT_DOWNLOADER = YoutubeDownloader()


def download_original_and_vtt(
    youtube_url: str,
    original_path: str,
    progress_cb: Optional[Callable[[dict], None]] = None,
):
    return _DEFAULT_DOWNLOADER.download_original_and_vtt(
        youtube_url, original_path, progress_cb=progress_cb
    )


def download_vtt_only(youtube_url: str, out_dir: str) -> Optional[str]:
    return _DEFAULT_DOWNLOADER.download_vtt_only(youtube_url, out_dir)
