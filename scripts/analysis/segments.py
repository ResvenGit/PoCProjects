# -*- coding: utf-8 -*-
"""
Silence detection, VTT parsing, and segment refinement helpers.
"""

from __future__ import annotations

import os
import re
from typing import Callable, List, Optional, Sequence, Tuple

from ..config import (
    END_WINDOW,
    MIN_LEN,
    NEXT_END_EXTRA_WINDOW,
    SILENCE_DB,
    SILENCE_DUR,
    START_WINDOW,
)
from ..utils.process import run_command_stream

try:
    import webvtt  # type: ignore

    HAS_WEBVTT = True
except Exception:  # pragma: no cover - fallback when package missing
    HAS_WEBVTT = False


class SegmentAnalyzer:
    """Wraps ffmpeg/webvtt powered segment utilities."""

    def __init__(
        self,
        silence_db: int = SILENCE_DB,
        silence_duration: float = SILENCE_DUR,
        start_window: Tuple[float, float] = START_WINDOW,
        end_window: Tuple[float, float] = END_WINDOW,
        min_length: float = MIN_LEN,
        next_end_window: Tuple[float, float] = NEXT_END_EXTRA_WINDOW,
        prev_start_window: Tuple[float, float] = (5.0, 5.0),
    ) -> None:
        self.silence_db = silence_db
        self.silence_duration = silence_duration
        self.start_window = start_window
        self.end_window = end_window
        self.min_length = min_length
        self.next_end_window = next_end_window
        self.prev_start_window = prev_start_window

    @staticmethod
    def has_webvtt() -> bool:
        return HAS_WEBVTT

    def detect_silence(
        self,
        video_path: str,
        media_duration: Optional[float] = None,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        timeout_factor: float = 1.4,
    ) -> List[Tuple[float, float]]:
        spans: List[Tuple[float, float]] = []
        starts: List[float] = []
        time_re = re.compile(r"time=([0-9:\.]+)")

        def _hms_to_sec_ff(ts: str) -> Optional[float]:
            parts = ts.split(":")
            if len(parts) != 3:
                return None
            h, m, sec = parts
            try:
                return int(h) * 3600 + int(m) * 60 + float(sec)
            except ValueError:
                return None

        def on_stderr(line: str):
            nonlocal spans, starts
            m0 = re.search(r"silence_start:\s*([0-9\.]+)", line)
            m1 = re.search(r"silence_end:\s*([0-9\.]+)", line)
            if m0:
                try:
                    starts.append(float(m0.group(1)))
                except ValueError:
                    pass
            if m1:
                try:
                    end_time = float(m1.group(1))
                    if starts:
                        start_time = starts.pop(0)
                        spans.append((start_time, end_time))
                except ValueError:
                    pass
            if media_duration and progress_cb:
                mt = time_re.search(line)
                if mt:
                    current = _hms_to_sec_ff(mt.group(1))
                    if current is not None and media_duration > 0:
                        pct = int(min(100, max(0, current * 100.0 / media_duration)))
                        progress_cb(pct, "무음 분석 중...")

        timeout = max(60.0, (media_duration or 600.0) * timeout_factor)
        cmd = (
            f'ffmpeg -hide_banner -nostdin -i "{video_path}" '
            f"-af silencedetect=noise={self.silence_db}dB:d={self.silence_duration} -f null -"
        )
        try:
            rc, _, _ = run_command_stream(cmd, timeout_sec=timeout, on_stderr_line=on_stderr)
            if rc != 0:
                print(f"[silencedetect] returncode={rc}", flush=True)
        except TimeoutError as exc:  # pragma: no cover - only triggered on long ops
            print(f"[silencedetect] TIMEOUT: {exc}", flush=True)
        return spans

    def sentence_bounds(self, vtt_path: str, max_gap: float = 0.4) -> List[float]:
        if not (HAS_WEBVTT and vtt_path and os.path.exists(vtt_path)):
            return []
        print(f"[vtt] parsing: {vtt_path}", flush=True)
        cues = list(webvtt.read(vtt_path))
        print(f"[vtt] cues: {len(cues)}", flush=True)
        bounds: List[float] = []
        buf = ""
        start_ts: Optional[float] = None
        end_ts: Optional[float] = None
        prev_end: Optional[float] = None

        def flush():
            nonlocal buf, end_ts
            if buf and end_ts is not None:
                bounds.append(end_ts)
            buf = ""

        for idx, cue in enumerate(cues):
            if idx % 200 == 0:
                print(f"[vtt] progress cues={idx}/{len(cues)}", flush=True)
            cue_start, cue_end = cue.start_in_seconds, cue.end_in_seconds
            text = cue.text.strip().replace("\n", " ")
            gap = (cue_start - prev_end) if prev_end is not None else None
            if start_ts is None:
                start_ts = cue_start
            if gap is not None and gap > max_gap:
                bounds.append(cue_start)
                flush()
                start_ts = cue_start
            buf = (buf + " " + text).strip() if buf else text
            end_ts = cue_end
            if buf.endswith(("?", "!", ".", "…", "?!", "!!")):
                flush()
            prev_end = cue_end
        flush()
        print(f"[vtt] sentence bounds: {len(bounds)}", flush=True)
        return sorted(set(bounds))

    def _snap_time(
        self,
        ts: float,
        window: Tuple[float, float],
        silence_spans: Sequence[Tuple[float, float]],
        sentence_bounds: Sequence[float],
    ) -> Tuple[float, str]:
        left, right = window
        candidates: List[Tuple[float, int, float]] = []
        for start, end in silence_spans:
            if (ts - left) <= start <= (ts + right):
                candidates.append((abs(start - ts), 3, start))
            if (ts - left) <= end <= (ts + right):
                candidates.append((abs(end - ts), 2, end))
        for bound in sentence_bounds:
            if (ts - left) <= bound <= (ts + right):
                candidates.append((abs(bound - ts), 2, bound))
        if not candidates:
            return ts, "none"
        candidates.sort(key=lambda item: (item[0], -item[1]))
        chosen = candidates[0][2]
        label = "silence" if candidates[0][1] == 3 else "sentence"
        return chosen, label

    def refine_segments(
        self,
        segments: Sequence[Tuple[float, float]],
        silence_spans: Sequence[Tuple[float, float]],
        sentence_bounds: Sequence[float],
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[str, str]]]:
        refined: List[Tuple[float, float]] = []
        reasons: List[Tuple[str, str]] = []
        for start, end in segments:
            r_start, reason_start = self._snap_time(start, self.start_window, silence_spans, sentence_bounds)
            r_end, reason_end = self._snap_time(end, self.end_window, silence_spans, sentence_bounds)
            if r_end - r_start < self.min_length:
                r_end = r_start + self.min_length
                reason_end += "+minlen"
            refined.append((r_start, r_end))
            reasons.append((reason_start, reason_end))
        for idx in range(1, len(refined)):
            prev_start, prev_end = refined[idx - 1]
            cur_start, cur_end = refined[idx]
            if cur_start < prev_end - 0.3:
                mid = (prev_end + cur_start) / 2.0
                refined[idx - 1] = (prev_start, mid)
                refined[idx] = (mid, cur_end)
        return refined, reasons

    def next_recommended_end(
        self,
        current_end: float,
        silence_spans: Sequence[Tuple[float, float]],
        sentence_bounds: Sequence[float],
    ) -> float:
        return self._next_or_prev(current_end, silence_spans, sentence_bounds, forward=True)

    def prev_recommended_start(
        self,
        current_start: float,
        silence_spans: Sequence[Tuple[float, float]],
        sentence_bounds: Sequence[float],
    ) -> float:
        return self._next_or_prev(current_start, silence_spans, sentence_bounds, forward=False)

    def _next_or_prev(
        self,
        ts: float,
        silence_spans: Sequence[Tuple[float, float]],
        sentence_bounds: Sequence[float],
        forward: bool,
    ) -> float:
        window = self.next_end_window if forward else self.prev_start_window
        candidates: List[Tuple[float, int, float]] = []
        for start, end in silence_spans:
            if forward:
                if ts <= start <= ts + window[1]:
                    candidates.append((start, 3))
                if ts <= end <= ts + window[1]:
                    candidates.append((end, 2))
            else:
                if ts - window[0] <= start <= ts:
                    candidates.append((start, 3))
                if ts - window[0] <= end <= ts:
                    candidates.append((end, 2))
        for bound in sentence_bounds:
            if forward:
                if ts <= bound <= ts + window[1]:
                    candidates.append((bound, 2))
            else:
                if ts - window[0] <= bound <= ts:
                    candidates.append((bound, 2))
        if not candidates:
            return ts
        if forward:
            candidates.sort(key=lambda item: (abs(item[0] - ts), -item[1]))
        else:
            candidates.sort(key=lambda item: (abs(ts - item[0]), -item[1]))
        return candidates[0][0]


_DEFAULT_ANALYZER = SegmentAnalyzer()


def get_silence_spans(
    video_path: str,
    noise_db: int = SILENCE_DB,
    dur: float = SILENCE_DUR,
    media_duration: Optional[float] = None,
    progress_cb=None,
    timeout_factor: float = 1.4,
):
    analyzer = SegmentAnalyzer(silence_db=noise_db, silence_duration=dur)
    return analyzer.detect_silence(
        video_path,
        media_duration=media_duration,
        progress_cb=progress_cb,
        timeout_factor=timeout_factor,
    )


def vtt_sentence_bounds(vtt_path: str, max_gap: float = 0.4) -> List[float]:
    return _DEFAULT_ANALYZER.sentence_bounds(vtt_path, max_gap=max_gap)


def refine_segments(
    segments: Sequence[Tuple[float, float]],
    silence_spans: Sequence[Tuple[float, float]],
    sentence_bounds: Sequence[float],
    start_win: Tuple[float, float] = START_WINDOW,
    end_win: Tuple[float, float] = END_WINDOW,
    min_len: float = MIN_LEN,
):
    analyzer = SegmentAnalyzer(
        start_window=start_win, end_window=end_win, min_length=min_len
    )
    return analyzer.refine_segments(segments, silence_spans, sentence_bounds)


def next_recommended_end(
    current_end: float,
    silence_spans: Sequence[Tuple[float, float]],
    sentence_bounds: Sequence[float],
    window: Tuple[float, float] = NEXT_END_EXTRA_WINDOW,
):
    analyzer = SegmentAnalyzer(next_end_window=window)
    return analyzer.next_recommended_end(current_end, silence_spans, sentence_bounds)


def prev_recommended_start(
    current_start: float,
    silence_spans: Sequence[Tuple[float, float]],
    sentence_bounds: Sequence[float],
    window: Tuple[float, float] = (5.0, 5.0),
):
    analyzer = SegmentAnalyzer(prev_start_window=window)
    return analyzer.prev_recommended_start(current_start, silence_spans, sentence_bounds)
