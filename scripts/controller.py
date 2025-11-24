# -*- coding: utf-8 -*-
"""
Thin controller that coordinates services for the UI layer.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from .analysis.segments import SegmentAnalyzer
from .config import (
    END_WINDOW,
    MIN_LEN,
    NEXT_END_EXTRA_WINDOW,
    SILENCE_DB,
    SILENCE_DUR,
    START_WINDOW,
)
from .media.ffmpeg_service import FFmpegService
from .services.downloader import YoutubeDownloader


class VideoEditingController:
    def __init__(
        self,
        ffmpeg_service: Optional[FFmpegService] = None,
        segment_analyzer: Optional[SegmentAnalyzer] = None,
        downloader: Optional[YoutubeDownloader] = None,
    ) -> None:
        self.ffmpeg = ffmpeg_service or FFmpegService()
        self.segment_analyzer = segment_analyzer or SegmentAnalyzer(
            silence_db=SILENCE_DB,
            silence_duration=SILENCE_DUR,
            start_window=START_WINDOW,
            end_window=END_WINDOW,
            min_length=MIN_LEN,
            next_end_window=NEXT_END_EXTRA_WINDOW,
        )
        self.downloader = downloader or YoutubeDownloader()

    # ---------- Media helpers ----------
    def get_video_specs(self, path: str):
        return self.ffmpeg.get_video_specs(path)

    def get_media_duration(self, path: str) -> Optional[float]:
        return self.ffmpeg.get_media_duration(path)

    def cut_clip(
        self,
        in_path: str,
        start: float,
        end: float,
        out_path: str,
        fps: str,
        width: int,
        height: int,
        sar: str,
        audio_rate: Optional[str],
        media_duration: Optional[float],
    ) -> bool:
        return self.ffmpeg.cut_clip(in_path, start, end, out_path, fps, width, height, sar, audio_rate, media_duration)

    def concat_all(
        self,
        intro_path: Optional[str],
        clip_paths: Sequence[str],
        outro_path: Optional[str],
        has_audio: bool,
        fps: str,
        width: int,
        height: int,
        sar: str,
        audio_rate: Optional[str],
        output_path: str,
    ):
        return self.ffmpeg.concat_all(intro_path, clip_paths, outro_path, has_audio, fps, width, height, sar, audio_rate, output_path)

    # ---------- Downloading ----------
    def download_original_and_vtt(self, url: str, original_path: str, progress_cb=None):
        return self.downloader.download_original_and_vtt(url, original_path, progress_cb=progress_cb)

    def download_vtt_only(self, url: str, out_dir: str):
        return self.downloader.download_vtt_only(url, out_dir)

    # ---------- Segments ----------
    def detect_silence(
        self,
        video_path: str,
        media_duration: Optional[float],
        progress_cb=None,
        timeout_factor: float = 1.4,
    ):
        return self.segment_analyzer.detect_silence(
            video_path,
            media_duration=media_duration,
            progress_cb=progress_cb,
            timeout_factor=timeout_factor,
        )

    def sentence_bounds(self, vtt_path: str, max_gap: float = 0.4):
        return self.segment_analyzer.sentence_bounds(vtt_path, max_gap=max_gap)

    def refine_segments(
        self,
        segments: Sequence[Tuple[float, float]],
        silence_spans: Sequence[Tuple[float, float]],
        sentence_bounds: Sequence[float],
    ):
        return self.segment_analyzer.refine_segments(segments, silence_spans, sentence_bounds)

    def next_recommended_end(self, current_end: float, silence_spans, sentence_bounds):
        return self.segment_analyzer.next_recommended_end(current_end, silence_spans, sentence_bounds)

    def prev_recommended_start(self, current_start: float, silence_spans, sentence_bounds):
        return self.segment_analyzer.prev_recommended_start(current_start, silence_spans, sentence_bounds)
