# -*- coding: utf-8 -*-
"""
FFmpeg/FFprobe helpers wrapped in a small service class.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import AFADE_OUT, L_CUT_SEC, VIDEO_CODEC, VIDEO_Q
from ..utils.process import run_command


class FFmpegService:
    def __init__(
        self,
        video_codec: str = VIDEO_CODEC,
        video_quality: str = VIDEO_Q,
        audio_bitrate: str = "192k",
    ) -> None:
        self.video_codec = video_codec
        self.video_quality = video_quality
        self.audio_bitrate = audio_bitrate

    def get_video_specs(self, path: str) -> Optional[Dict[str, str]]:
        normalized = path.replace("\\", "/")
        ok, out = run_command(
            f'ffprobe -v error -select_streams v:0 '
            f'-show_entries stream=width,height,r_frame_rate,sample_aspect_ratio '
            f'-of json "{normalized}"'
        )
        if not ok:
            return None
        try:
            data = json.loads(out)
            video = data["streams"][0]
        except Exception:
            return None
        ok_audio, audio_out = run_command(
            f'ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of json "{normalized}"'
        )
        sample_rate = None
        if ok_audio:
            try:
                audio_data = json.loads(audio_out)
                if audio_data.get("streams"):
                    sample_rate = audio_data["streams"][0].get("sample_rate")
            except Exception:
                sample_rate = None
        return {
            "width": video.get("width", 1920),
            "height": video.get("height", 1080),
            "r_frame_rate": video.get("r_frame_rate", "30/1"),
            "sar": video.get("sample_aspect_ratio", "1:1"),
            "sample_rate": sample_rate,
        }

    def get_media_duration(self, path: str) -> Optional[float]:
        normalized = path.replace("\\", "/")
        ok, out = run_command(
            f'ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "{normalized}"'
        )
        if not ok:
            return None
        try:
            return float(out.strip())
        except ValueError:
            return None

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
        media_duration: Optional[float] = None,
        l_cut: float = L_CUT_SEC,
        afade_out: float = AFADE_OUT,
    ) -> bool:
        if media_duration is not None:
            end = min(end, media_duration)
        base_duration = max(0.01, end - start)
        video_enc = (
            f"-c:v {self.video_codec} -q:v {self.video_quality} "
            f"-r {fps} -s {width}x{height} -sar {sar}"
        )
        if audio_rate:
            total_t = base_duration + max(0.0, l_cut)
            if media_duration is not None:
                total_t = min(total_t, max(0.01, media_duration - start))
            fade_start = max(0.0, base_duration - afade_out)
            audio_enc = f"-c:a aac -b:a {self.audio_bitrate} -ar {audio_rate}"
            cmd = (
                f'ffmpeg -ss {start:.3f} -t {total_t:.3f} -i "{in_path}" '
                f"{video_enc} {audio_enc} "
                f'-af "afade=t=out:st={fade_start:.3f}:d={afade_out:.2f}" '
                f'"{out_path}" -y'
            )
        else:
            cmd = (
                f'ffmpeg -ss {start:.3f} -t {base_duration:.3f} -i "{in_path}" '
                f"{video_enc} -an \"{out_path}\" -y"
            )
        ok, _ = run_command(cmd)
        return ok

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
        inputs: List[str] = []
        chains: List[str] = []
        concat_items: List[str] = []
        idx = 0
        scale_pad = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
        )

        def add_input(path: str, need_scale: bool = False):
            nonlocal idx
            inputs.append(f'-i "{path}"')
            chain = f"[{idx}:v] setpts=PTS-STARTPTS"
            if need_scale:
                chain += f", {scale_pad}, setsar={sar}, fps={fps}"
            chains.append(f"{chain} [v{idx}]")
            if has_audio:
                audio_chain = f"[{idx}:a] asetpts=PTS-STARTPTS"
                if audio_rate:
                    audio_chain += f", aresample={audio_rate}"
                chains.append(f"{audio_chain} [a{idx}]")
                concat_items.append(f"[v{idx}][a{idx}]")
            else:
                concat_items.append(f"[v{idx}]")
            idx += 1

        if intro_path and os.path.exists(intro_path):
            add_input(intro_path, True)
        for clip in clip_paths:
            add_input(clip, False)
        if outro_path and os.path.exists(outro_path):
            add_input(outro_path, True)
        total = len(concat_items)
        if total == 0:
            return False, "no inputs"
        input_str = " ".join(inputs)
        if has_audio:
            filter_complex = (
                f"{'; '.join(chains)}; "
                f"{''.join(concat_items)} concat=n={total}:v=1:a=1 [outv] [outa]"
            )
            maps = (
                f'-map "[outv]" -map "[outa]" -c:a aac -b:a {self.audio_bitrate}'
                + (f" -ar {audio_rate}" if audio_rate else "")
            )
        else:
            filter_complex = (
                f"{'; '.join(chains)}; "
                f"{''.join(concat_items)} concat=n={total}:v=1:a=0 [outv]"
            )
            maps = '-map "[outv]" -an'
        cmd = (
            f'ffmpeg {input_str} -filter_complex "{filter_complex}" {maps} '
            f"-c:v {self.video_codec} -q:v {self.video_quality} -r {fps} "
            f'"{output_path}" -y'
        )
        return run_command(cmd)


_DEFAULT_FFMPEG = FFmpegService()


def get_video_specs(path: str) -> Optional[Dict[str, str]]:
    return _DEFAULT_FFMPEG.get_video_specs(path)


def get_media_duration(path: str) -> Optional[float]:
    return _DEFAULT_FFMPEG.get_media_duration(path)


def cut_clip(
    in_path: str,
    start: float,
    end: float,
    out_path: str,
    fps: str,
    width: int,
    height: int,
    sar: str,
    audio_rate: Optional[str],
    media_duration: Optional[float] = None,
    l_cut: float = L_CUT_SEC,
    afade_out: float = AFADE_OUT,
) -> bool:
    return _DEFAULT_FFMPEG.cut_clip(
        in_path,
        start,
        end,
        out_path,
        fps,
        width,
        height,
        sar,
        audio_rate,
        media_duration=media_duration,
        l_cut=l_cut,
        afade_out=afade_out,
    )


def concat_all(
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
    return _DEFAULT_FFMPEG.concat_all(
        intro_path,
        clip_paths,
        outro_path,
        has_audio,
        fps,
        width,
        height,
        sar,
        audio_rate,
        output_path,
    )
