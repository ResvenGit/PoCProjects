# -*- coding: utf-8 -*-
"""
YouTube Auto Clip Refiner - PySide6 (Total duration label + wider buttons + accurate 3s preview + farther next-end)
"""

import os, sys, re, json, subprocess, datetime, threading, time
from typing import List, Tuple, Optional

from PySide6.QtCore import Qt, QTimer, QObject, Signal, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QListWidget, QListWidgetItem,
    QLabel, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout,
    QTextEdit, QProgressBar, QSizePolicy
)

# -------- Optional deps --------
try:
    from yt_dlp import YoutubeDL
    HAS_YTDLP = True
except:
    HAS_YTDLP = False

try:
    import webvtt
    HAS_WEBVTT = True
except:
    HAS_WEBVTT = False

# =========================
# 설정 / 상수
# =========================
DEFAULT_URL = "https://www.youtube.com/watch?v=********"
DEFAULT_SEGMENTS_JSON = """
[
  { "start": "00:04:06.720", "end": "00:04:43.039" },
  { "start": "00:05:22.880", "end": "00:06:09.830" },
  { "start": "00:16:48.000", "end": "00:17:43.750" },
  { "start": "00:22:56.240", "end": "00:23:47.110" },
  { "start": "00:45:01.480", "end": "00:45:12.950" },
  { "start": "00:46:01.440", "end": "00:46:46.359" },
  { "start": "00:54:37.720", "end": "00:55:31.950" }
]
""".strip()

INTRO_DEFAULT = "C:/MyClips/_Resource/_Intro_251122.mp4"
OUTRO_DEFAULT = "C:/MyClips/_Resource/_Outro_251014.mp4"
OUTPUT_DEFAULT = "C:/MyClips/Target_Clip.mp4"

# 보정 파라미터
SILENCE_DB = -30
SILENCE_DUR = 0.18
START_WINDOW = (1.5, 2.5)
END_WINDOW   = (2.5, 1.0)
MIN_LEN = 8.0

# 미리듣기
TAIL_PREVIEW_SEC = 3.0
HEAD_PREVIEW_SEC = 3.0

# 인코딩
L_CUT_SEC = 0.25
AFADE_OUT = 0.12
VIDEO_CODEC = "mpeg4"
VIDEO_Q     = "3"

# “다음 추천 End” 탐색 범위 확장(+25s 앞으로)
NEXT_END_EXTRA_WINDOW = (5.0, 25.0)

# =========================
# 프롬프트(생략: 이전과 동일)
# =========================
TOPIC_PROMPT = r"""***토픽추출프롬프트***
당신은 베테랑 유튜브 PD이자 전문 편집자입니다.
입력된 이 긴 원본 비디오를 시각적으로 정밀하게 분석하여, **총 길이가 3분에서 6분 사이**인 **독립된 주제**의 영상 기획안을 **여러 개** 작성해야 합니다.
분석 대상 영상 : {}



**[매우 중요]**
하나의 주제(Topic)가 원본 비디오의 **여러 분산된 구간(Non-continuous segments)**에서 다뤄질 수 있습니다.
이 경우, 해당 주제와 관련된 **모든 관련 구간 조각들**을 찾아서 하나의 영상 기획안으로 묶어야 합니다. (예: 'A' 주제가 1분대와 7분대에 나오면, 이 두 구간을 모두 포함)

**[작업 지침]**
식별된 **각각의 영상(기획안)**에 대해, 다음 4가지 항목을 포함해야 합니다.
1.  **topic**: 이 영상의 핵심 주제 (한글로 간략히 설명).
2.  **clip_segments**: 이 주제를 다루는 **모든** 원본 비디오 구간들의 **목록(배열)**. 각 항목은 'start' 및 'end' 타임스탬프를 가져야 합니다. (HH:MM:SS 형식)
3.  **titles**: 추천 제목 5개 (배열).
4.  **thumbnail_prompts**: AI 이미지 생성기용 썸네일 프롬프트 3개 (배열).

**[출력 규칙]**
- 제목과 썸네일 프롬프트는 **한글**을 기본으로 하되, **영어**를 혼용하는 것을 허용합니다.
- 썸네일 프롬프트는 매우 시각적이고 구체적이어야 합니다.
- 결과는 반드시 **JSON 형식**이어야 하며, 모든 기획안은 `video_proposals`라는 이름의 **배열(List)** 안에 포함되어야 합니다.

**[JSON 출력 예시]**
{
  "video_proposals": [
    {
      "topic": "첫 번째 영상 주제 (예: 'A' 개념의 정의와 사례들)",
      "clip_segments": [
        { "start": "00:01:15", "end": "00:02:45" },
        { "start": "00:07:10", "end": "00:09:00" },
        { "start": "00:12:30", "end": "00:13:15" }
      ],
      "titles": [
        "추천 제목 1 (한글)",
        "추천 제목 2 (English/Korean Mix)",
        ...
      ],
      "thumbnail_prompts": [
        "시각적 썸네일 프롬프트 1 (예: 'A' 개념을 상징하는 오브젝트 클로즈업, 생생한 색감)",
        ...
      ]
    },
    {
      "topic": "두 번째 영상 주제 (예: 'B' 개념의 심층 분석)",
      "clip_segments": [
        { "start": "00:03:00", "end": "00:06:30" }
      ],
      "titles": [ ... ],
      "thumbnail_prompts": [ ... ]
    }
  ]
}
"""

VTT_ANALYSIS_PROMPT = r"""***VTT분석프롬프트***
당신은 VTT 트랜스크립트를 분석하여 영상 편집 지점을 추천하는 전문 어시스턴트입니다.

제공된 VTT 트랜스크립트 파일과 {주제}를 기반으로, 해당 주제의 핵심 내용을 가장 잘 담고 있는 복수의 영상 구간을 추천해야 합니다. 아래 복수의 {주제}에 대해 각각의 결과를 생성하되 각 주제의 영상 구간이 최대한 겹치지 않도록 해야 합니다.



**{주제}:**

"AI 버추얼 캐릭터, 뭘로 어떻게 시작할까?"


**엄격한 규칙:**
1.  **주제 관련성:** 추천된 모든 구간은 위에 명시된 {주제}와 직접적으로 관련이 있어야 합니다.
2.  **개별 구간 길이:** 각 영상 구간(segment)의 길이는 **최대 1분을 초과할 수 없습니다.**
3.  **개별 구간 규칙:** 각 영상 구간(segment)의 시작과 끝 위치는 **문장이 중간에 끝나지 않도록 고려해야 합니다.** 다음 문장이 시작되기 전을 끝나는 위치로 해서 문장이 중간에 끝나지 않는 것을 보장해야 합니다.
4.  **총 길이:** 모든 구간의 길이를 합친 총합은 **최소 2분 이상, 최대 6분 이하**여야 합니다.
5.  **타임스탬프 형식:** VTT 파일에 사용된 `HH:MM:SS.mmm` (시:분:초.밀리초) 형식을 정확히 준수해야 합니다.
6.  **출력 형식:** 응답은 **반드시** 아래와 같은 구조로 출력되어야 합니다.
    * **첫째 줄:** 모든 추천 구간을 이었을 때의 **핵심 내용 요약**.
    * **둘째 줄:** 생성될 영상의 흐름이나 완성도에 대한 **간단한 평가**.
    * **셋째 줄:** 생성될 영상의 **총 길이** (예: `총 길이: 3분 45초`).
    * **JSON 블록:** 요약, 평가, 총 길이 텍스트가 끝난 후, 다음 줄부터 JSON 코드 블록으로 타임스탬프를 제공해야 합니다.

**출력 포맷 예시:**
AI 버추얼 캐릭터를 만드는 두 가지 방법(2D, 3D)을 소개하고, AI를 활용한 3D 모델 생성 과정과 필요한 도구(Meshy AI, Unity, VCFace)를 설명하는 영상입니다.
핵심 도구와 제작 흐름을 파악하기에 적합하며, 각 단계가 논리적으로 연결됩니다.
총 길이: 4분 12초
```json
[
  { "start": "00:02:54.959", "end": "00:03:38.990" },
  { "start": "00:05:36.479", "end": "00:06:11.230" },
  { "start": "00:07:06.360", "end": "00:08:02.430" }
]
"""

THUMBNAIL_PROMPT = r"""***썸네일프롬프트***
** 유튜브 썸네일 이미지 생성 요청


** 내용 : 
{  }


** 스타일은 아래의 내용을 반영 : 
전문적인 유튜브 썸네일. 매끈하고, 미래지향적이며, 하이테크한 컨셉 아트 스타일. 캐릭터는 3D 일본 아니메 스타일(Japanese anime style)이며, 맑고 깨끗한(clean rendering) 셀 셰이딩(cell-shading) 또는 반실사(semi-realistic) 렌더링을 특징으로 함. 분위기는 어둡고, 깊은 네이비 블루 또는 블랙을 기본으로 함. 전체 장면은 빛나는 네온 블루와 일렉트릭 사이언(electric cyan) 액센트 컬러가 지배적임. 배경은 홀로그램 UI 화면, 떠다니는 데이터 디스플레이, 빛나는 디지털 회로 경로로 복잡하게 채워져 있으며, 은은한 보케(bokeh) 효과가 있음. 모든 액센트 요소는 강한 '네온 광선' 또는 '발광(emissive)' 효과를 가져야 함.


** 추가적인 반영 필요 내용
글자가 추가되어야 하면 영어로만 쓸 것. 
"""

# =========================
# 공통 유틸
# =========================
def run_command(cmd: str):
    try:
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        p = subprocess.run(cmd, shell=True, check=True, creationflags=flags,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, encoding='utf-8', errors='ignore')
        return True, (p.stdout or "") + (p.stderr or "")
    except subprocess.CalledProcessError as e:
        return False, (e.stdout or "") + (e.stderr or "")

def run_command_stream(cmd: str, timeout_sec: float = None, on_stderr_line=None, on_stdout_line=None):
    print(f"[run] {cmd}", flush=True)
    p = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding='utf-8', errors='ignore'
    )
    out_lines, err_lines = [], []
    out_done = threading.Event()
    err_done = threading.Event()

    def _pump(pipe, collector, cb, done_evt, name):
        try:
            for line in iter(pipe.readline, ''):
                collector.append(line)
                if cb:
                    cb(line)
                if name == 'stderr':
                    print(line.rstrip(), file=sys.stderr, flush=True)
                else:
                    print(line.rstrip(), flush=True)
        finally:
            try: pipe.close()
            except Exception: pass
            done_evt.set()

    t_out = threading.Thread(target=_pump, args=(p.stdout, out_lines, on_stdout_line, out_done, 'stdout'), daemon=True)
    t_err = threading.Thread(target=_pump, args=(p.stderr, err_lines, on_stderr_line, err_done, 'stderr'), daemon=True)
    t_out.start(); t_err.start()

    start = time.time()
    while True:
        if timeout_sec is not None and (time.time() - start) > timeout_sec:
            try: p.kill()
            except Exception: pass
            raise TimeoutError(f"Process timed out after {timeout_sec:.1f}s")
        if p.poll() is not None:
            break
        time.sleep(0.05)

    out_done.wait(2.0); err_done.wait(2.0)
    return p.returncode, ''.join(out_lines), ''.join(err_lines)

def hms_to_sec(ts: str) -> float:
    parts = ts.strip().split(':')
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        return float(ts)
    return int(h)*3600 + int(m)*60 + float(s)

def sec_to_hms(x: float) -> str:
    x = max(0.0, float(x))
    h = int(x // 3600)
    m = int((x % 3600) // 60)
    s = x - h*3600 - m*60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def get_video_specs(path: str):
    path = path.replace("\\", "/")
    ok, out = run_command(f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,sample_aspect_ratio -of json "{path}"')
    if not ok: return None
    try:
        data = json.loads(out); v = data["streams"][0]
    except:
        return None
    ok2, out2 = run_command(f'ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of json "{path}"')
    sr = None
    if ok2:
        try:
            a = json.loads(out2)
            if a.get("streams"):
                sr = a["streams"][0].get("sample_rate")
        except: pass
    return {"width": v.get("width",1920), "height": v.get("height",1080),
            "r_frame_rate": v.get("r_frame_rate","30/1"), "sar": v.get("sample_aspect_ratio","1:1"),
            "sample_rate": sr}

def get_media_duration(path: str) -> Optional[float]:
    path = path.replace("\\", "/")
    ok, out = run_command(f'ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "{path}"')
    if not ok: return None
    try: return float(out.strip())
    except: return None

def original_video_path_of(output_path: str) -> str:
    base = os.path.dirname(output_path) or os.getcwd()
    name_no_ext = os.path.splitext(os.path.basename(output_path))[0]
    return os.path.join(base, f"{name_no_ext}_original.mp4").replace("\\","/")

def guess_workdir(output_path: str) -> str:
    base = os.path.dirname(output_path) or os.getcwd()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wd = os.path.join(base, ts).replace("\\","/")
    os.makedirs(wd, exist_ok=True)
    return wd

def _find_latest_vtt(folder: str) -> Optional[str]:
    candidates = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith(".vtt"):
                candidates.append(os.path.join(root, fn))
    if not candidates: return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

# =========================
# 분석/보정 로직
# =========================
def get_silence_spans(video_path: str, noise_db=SILENCE_DB, dur=SILENCE_DUR,
                      media_duration: float = None, progress_cb=None, timeout_factor: float = 1.4) -> List[Tuple[float,float]]:
    spans, starts = [], []
    time_re = re.compile(r'time=([0-9:\.]+)')

    def _hms_to_sec_ff(s):
        parts = s.split(':')
        if len(parts) != 3: return None
        h, m, sec = parts
        try: return int(h)*3600 + int(m)*60 + float(sec)
        except: return None

    def on_stderr(line: str):
        nonlocal spans, starts
        m0 = re.search(r"silence_start:\s*([0-9\.]+)", line)
        m1 = re.search(r"silence_end:\s*([0-9\.]+)", line)
        if m0:
            try: starts.append(float(m0.group(1)))
            except: pass
        if m1:
            try:
                t = float(m1.group(1))
                if starts:
                    s0 = starts.pop(0)
                    spans.append((s0, t))
            except: pass
        if media_duration and progress_cb:
            mt = time_re.search(line)
            if mt:
                tcur = _hms_to_sec_ff(mt.group(1))
                if tcur is not None and media_duration > 0:
                    pct = int(min(100, max(0, tcur * 100.0 / media_duration)))
                    progress_cb(pct, "무음 분석 중...")

    timeout = max(60.0, (media_duration or 600.0) * timeout_factor)
    cmd = f'ffmpeg -hide_banner -nostdin -i "{video_path}" -af silencedetect=noise={noise_db}dB:d={dur} -f null -'
    try:
        rc, _, _ = run_command_stream(cmd, timeout_sec=timeout, on_stderr_line=on_stderr)
        if rc != 0:
            print(f"[silencedetect] returncode={rc}", flush=True)
    except TimeoutError as te:
        print(f"[silencedetect] TIMEOUT: {te}", flush=True)
    return spans

def vtt_sentence_bounds(vtt_path: str, max_gap=0.4) -> List[float]:
    print(f"[vtt] parsing: {vtt_path}", flush=True)
    if not (HAS_WEBVTT and vtt_path and os.path.exists(vtt_path)):
        return []
    cues = list(webvtt.read(vtt_path))
    print(f"[vtt] cues: {len(cues)}", flush=True)
    bounds, buf, s, e = [], "", None, None
    prev_end = None
    def flush():
        nonlocal buf, s, e
        if buf and e is not None:
            bounds.append(e)
        buf, s, e = "", None, None
    for idx, c in enumerate(cues):
        if idx % 200 == 0:
            print(f"[vtt] progress cues={idx}/{len(cues)}", flush=True)
        cs, ce = c.start_in_seconds, c.end_in_seconds
        t = c.text.strip().replace("\n"," ")
        gap = (cs - prev_end) if prev_end is not None else None
        if s is None: s = cs
        if gap is not None and gap > max_gap:
            bounds.append(cs); flush(); s = cs
        buf = (buf + " " + t).strip() if buf else t
        e = ce
        if buf.endswith(("?", "!", ".", "…", "다.", "요.", "죠.")):
            flush()
        prev_end = ce
    flush()
    print(f"[vtt] sentence bounds: {len(bounds)}", flush=True)
    return sorted(set(bounds))

def snap_time(t: float, window: Tuple[float,float], silence_spans, sentence_bounds):
    wL, wR = window; cands = []
    for s0, s1 in silence_spans:
        if (t - wL) <= s0 <= (t + wR): cands.append((abs(s0-t), 3, s0))
        if (t - wL) <= s1 <= (t + wR): cands.append((abs(s1-t), 2, s1))
    for b in sentence_bounds:
        if (t - wL) <= b <= (t + wR): cands.append((abs(b-t), 2, b))
    if not cands: return t, "none"
    cands.sort(key=lambda x: (x[0], -x[1]))
    chosen = cands[0][2]
    label = "silence" if cands[0][1]==3 else "sentence"
    return chosen, label

def refine_segments(segments: List[Tuple[float,float]], silence_spans, sentence_bounds,
                    start_win=START_WINDOW, end_win=END_WINDOW, min_len=MIN_LEN):
    refined, reasons = [], []
    for (s,e) in segments:
        rs, r1 = snap_time(s, start_win, silence_spans, sentence_bounds)
        re, r2 = snap_time(e, end_win,   silence_spans, sentence_bounds)
        if re - rs < min_len:
            re = rs + min_len; r2 += "+minlen"
        refined.append((rs,re)); reasons.append((r1,r2))
    for i in range(1, len(refined)):
        ps, pe = refined[i-1]; cs, ce = refined[i]
        if cs < pe - 0.3:
            mid = (pe + cs) / 2.0
            refined[i-1] = (ps, mid); refined[i] = (mid, ce)
    return refined, reasons

def next_recommended_end(current_end, silence_spans, sentence_bounds, window=(5.0,25.0)):
    t = current_end; cands = []
    for s0, s1 in silence_spans:
        if t <= s0 <= t+window[1]: cands.append((s0,3))
        if t <= s1 <= t+window[1]: cands.append((s1,2))
    for b in sentence_bounds:
        if t <= b <= t+window[1]: cands.append((b,2))
    if not cands: return current_end
    cands.sort(key=lambda x: (abs(x[0]-t), -x[1])); return cands[0][0]

def prev_recommended_start(current_start, silence_spans, sentence_bounds, window=(5.0,5.0)):
    t = current_start; cands = []
    for s0, s1 in silence_spans:
        if t-window[0] <= s0 <= t: cands.append((s0,3))
        if t-window[0] <= s1 <= t: cands.append((s1,2))
    for b in sentence_bounds:
        if t-window[0] <= b <= t: cands.append((b,2))
    if not cands: return current_start
    cands.sort(key=lambda x: (abs(t-x[0]), -x[1])); return cands[0][0]

# =========================
# 컷/병합
# =========================
def cut_clip(in_path, start, end, out_path, fps, w, h, sar, audio_rate, media_duration=None, l_cut=L_CUT_SEC, afade_out=AFADE_OUT):
    if media_duration is not None:
        end = min(end, media_duration)
    base_dur = max(0.01, end - start)
    venc = f'-c:v {VIDEO_CODEC} -q:v {VIDEO_Q} -r {fps} -s {w}x{h} -sar {sar}'
    if audio_rate:
        total_t = base_dur + max(0.0, l_cut)
        if media_duration is not None:
            total_t = min(total_t, max(0.01, media_duration - start))
        fade_start = max(0.0, base_dur - afade_out)
        aenc = f'-c:a aac -b:a 192k -ar {audio_rate}'
        cmd = (f'ffmpeg -ss {start:.3f} -t {total_t:.3f} -i "{in_path}" '
               f'{venc} {aenc} -af "afade=t=out:st={fade_start:.3f}:d={afade_out:.2f}" '
               f'"{out_path}" -y')
    else:
        aenc = '-an'
        cmd = f'ffmpeg -ss {start:.3f} -t {base_dur:.3f} -i "{in_path}" {venc} {aenc} "{out_path}" -y'
    ok, _ = run_command(cmd)
    return ok

def concat_all(intro_path, clip_paths, outro_path, has_audio, fps, w, h, sar, audio_rate, output_path):
    inputs, chains, concat_items = [], [], []
    idx = 0
    scale_pad = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black"
    def add_input(path, need_scale=False):
        nonlocal idx
        inputs.append(f'-i "{path}"')
        chains.append(f"[{idx}:v] setpts=PTS-STARTPTS{ (', ' + scale_pad + f', setsar={sar}, fps={fps}') if need_scale else '' } [v{idx}]")
        if has_audio:
            chains.append(f"[{idx}:a] asetpts=PTS-STARTPTS{ (f', aresample={audio_rate}') if audio_rate else '' } [a{idx}]")
            concat_items.append(f"[v{idx}][a{idx}]")
        else:
            concat_items.append(f"[v{idx}]")
        idx += 1
    if intro_path and os.path.exists(intro_path): add_input(intro_path, True)
    for p in clip_paths: add_input(p, False)
    if outro_path and os.path.exists(outro_path): add_input(outro_path, True)
    total = len(concat_items)
    if total == 0: return False, "no inputs"
    input_str = " ".join(inputs)
    if has_audio:
        filter_complex = f"{'; '.join(chains)}; {''.join(concat_items)} concat=n={total}:v=1:a=1 [outv] [outa]"
        maps = f'-map "[outv]" -map "[outa]" -c:a aac -b:a 192k' + (f' -ar {audio_rate}' if audio_rate else '')
    else:
        filter_complex = f"{'; '.join(chains)}; {''.join(concat_items)} concat=n={total}:v=1:a=0 [outv]"
        maps = '-map "[outv]" -an'
    cmd = f'ffmpeg {input_str} -filter_complex "{filter_complex}" {maps} -c:v {VIDEO_CODEC} -q:v {VIDEO_Q} -r {fps} "{output_path}" -y'
    return run_command(cmd)

# =========================
# 다운로드
# =========================
def download_original_and_vtt(youtube_url: str, original_path: str, progress_cb=None):
    if not HAS_YTDLP:
        ok, _ = run_command(f'yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" -o "{original_path}" "{youtube_url}"')
        return ok, _find_latest_vtt(os.path.dirname(original_path))
    def _hook(d):
        if progress_cb and d.get('status') == 'downloading':
            progress_cb(d)
    ydl_opts = {
        'outtmpl': original_path,
        'merge_output_format': 'mp4',
        'quiet': True,
        'progress_hooks': [_hook] if progress_cb else [],
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['ko','ko-KR','en','en-US'],
        'skip_download': False
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(youtube_url, download=True)
        return True, _find_latest_vtt(os.path.dirname(original_path))
    except Exception:
        return False, None

def download_vtt_only(youtube_url: str, out_dir: str) -> Optional[str]:
    os.makedirs(out_dir, exist_ok=True)
    if HAS_YTDLP:
        ydl_opts = {
            'quiet': True,
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['ko','ko-KR','en','en-US'],
            'outtmpl': os.path.join(out_dir, '%(title)s.%(ext)s'),
        }
        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(youtube_url, download=True)
        except Exception:
            return None
    else:
        ok, _ = run_command(
            f'yt-dlp --skip-download --write-auto-sub --write-sub '
            f'--sub-langs "ko,ko-KR,en,en-US" -o "{os.path.join(out_dir, "%(title)s.%(ext)s")}" '
            f'"{youtube_url}"'
        )
        if not ok:
            return None
    return _find_latest_vtt(out_dir)

# =========================
# Busy / Invoker
# =========================
class BusyManager(QObject):
    changed = Signal(int)
    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._active = 0
        self._next_token = 1
    def acquire(self) -> int:
        with self._lock:
            tok = self._next_token; self._next_token += 1
            self._active += 1
            print(f"[busy] acquire -> active={self._active} (token={tok})", flush=True)
            self.changed.emit(self._active)
            return tok
    def release(self, token: int):
        with self._lock:
            if self._active > 0:
                self._active -= 1
            else:
                print(f"[busy] WARN release with active=0 (token={token})", flush=True)
            print(f"[busy] release -> active={self._active} (token={token})", flush=True)
            self.changed.emit(self._active)
    @property
    def active(self) -> int:
        with self._lock:
            return self._active

class UiInvoker(QObject):
    sig_call = Signal(object, tuple, dict)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sig_call.connect(self._on_call, Qt.QueuedConnection)
    @Slot(object, tuple, dict)
    def _on_call(self, fn, args, kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as ex:
            print(f"[UiInvoker] exception: {ex}", flush=True)

# =========================
# GUI
# =========================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Auto Clip Refinement & Editor - PySide6")
        self.resize(1180, 780)

        # 상태
        self.segments: List[Tuple[float,float]] = []
        self.silence_spans: List[Tuple[float,float]] = []
        self.sentence_bounds: List[float] = []
        self.media_duration: Optional[float] = None
        self.original_path: Optional[str] = None

        # Busy + Watchdog + Invoker
        self.busy = BusyManager()
        self.busy.changed.connect(self.on_busy_changed)
        self._watchdog = QTimer(self); self._watchdog.setInterval(2000)
        self._watchdog.timeout.connect(self._watchdog_tick); self._watchdog.start()
        self._last_progress_ts = time.time()
        self._invoker = UiInvoker(self)

        # UI
        self._build_widgets()
        self._build_layout()
        self._connect_signals()
        self._load_default_segments()

    # ---------- Widgets ----------
    def _build_widgets(self):
        self.url_edit = QLineEdit(DEFAULT_URL)
        self.btn_download = QPushButton("원본 확인/다운로드")
        self.status_label = QLabel("확인 전")

        self.btn_copy_topic = QPushButton("분석 프롬프트 복사")
        self.btn_copy_vtt   = QPushButton("VTT 분석 프롬프트 복사")
        self.btn_copy_thumb = QPushButton("썸네일 프롬프트 복사")
        for b in (self.btn_copy_topic, self.btn_copy_vtt, self.btn_copy_thumb):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setMinimumHeight(36)

        self.vtt_edit = QLineEdit("")
        self.btn_vtt_browse = QPushButton("찾기")
        self.btn_vtt_download = QPushButton("자막 다운로드")
        self.btn_vtt_apply = QPushButton("VTT 연결")

        # 세그먼트 헤더 라벨(총 길이 포함)
        self.lbl_segments = QLabel("세그먼트 목록")

        self.list_segments = QListWidget()
        self.btn_prev = QPushButton("이전")
        self.btn_next = QPushButton("다음")

        # 폭을 넓혀 자연스럽게
        self.btn_apply_clipboard = QPushButton("클립보드에서 세그먼트 적용")
        self.btn_copy_segments = QPushButton("세그먼트 정보 클립보드에 복사")
        for b in (self.btn_apply_clipboard, self.btn_copy_segments):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setMinimumHeight(28)

        self.start_edit = QLineEdit("")
        self.end_edit = QLineEdit("")
        self.btn_start_m1 = QPushButton("-1s")
        self.btn_start_p1 = QPushButton("+1s")
        self.btn_prev_rec = QPushButton("이전 추천 위치")
        self.btn_end_m1   = QPushButton("-1s")
        self.btn_end_p1   = QPushButton("+1s")
        self.btn_next_rec = QPushButton("다음 추천 위치")
        self.btn_head_preview = QPushButton("처음 3초 미리듣기")
        self.btn_tail_preview = QPushButton("마지막 3초 미리듣기")
        self.btn_save_seg = QPushButton("세그먼트 저장")

        self.btn_extract = QPushButton("무음/문장 경계 추출")
        self.btn_refine  = QPushButton("자동 보정 실행")

        self.intro_edit = QLineEdit(INTRO_DEFAULT)
        self.btn_intro_browse = QPushButton("찾기")
        self.outro_edit = QLineEdit(OUTRO_DEFAULT)
        self.btn_outro_browse = QPushButton("찾기")
        self.output_edit = QLineEdit(OUTPUT_DEFAULT)
        self.btn_output_browse = QPushButton("저장위치")

        self.btn_render = QPushButton("최종 영상 생성")
        self.btn_render.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_render.setMinimumHeight(46)

        self.progress_label = QLabel("")
        self.global_progress = QProgressBar(); self.global_progress.setRange(0, 100); self.global_progress.setValue(0)
        self.global_progress.setFixedHeight(16); self.global_progress.setMaximumWidth(260)

        self.log_view = QTextEdit(); self.log_view.setReadOnly(True)

    def _build_layout(self):
        root = QWidget()
        root_layout = QVBoxLayout(root); root_layout.setContentsMargins(8,8,8,8); root_layout.setSpacing(6)

        row_url = QHBoxLayout(); row_url.setSpacing(6)
        row_url.addWidget(QLabel("YouTube URL")); row_url.addWidget(self.url_edit, 1)
        row_url.addWidget(self.btn_download); row_url.addWidget(QLabel("상태:")); row_url.addWidget(self.status_label)
        root_layout.addLayout(row_url)

        row_prompts = QHBoxLayout(); row_prompts.setSpacing(6)
        row_prompts.addWidget(self.btn_copy_topic); row_prompts.addWidget(self.btn_copy_vtt); row_prompts.addWidget(self.btn_copy_thumb)
        root_layout.addLayout(row_prompts)

        row_vtt = QHBoxLayout(); row_vtt.setSpacing(6)
        row_vtt.addWidget(QLabel("VTT 경로")); row_vtt.addWidget(self.vtt_edit, 1)
        row_vtt.addWidget(self.btn_vtt_browse); row_vtt.addWidget(self.btn_vtt_download); row_vtt.addWidget(self.btn_vtt_apply)
        root_layout.addLayout(row_vtt)

        mid = QHBoxLayout(); mid.setSpacing(10)
        left = QVBoxLayout(); left.setSpacing(6)
        left.addWidget(self.lbl_segments)
        left.addWidget(self.list_segments, 1)
        nav = QHBoxLayout(); nav.addWidget(self.btn_prev); nav.addWidget(self.btn_next)
        left.addLayout(nav)
        cliprow = QHBoxLayout()
        cliprow.addWidget(self.btn_apply_clipboard)
        cliprow.addWidget(self.btn_copy_segments)
        left.addLayout(cliprow)
        mid.addLayout(left, 3)

        right = QVBoxLayout(); right.setSpacing(6)
        grid_edit = QGridLayout(); grid_edit.setHorizontalSpacing(6); grid_edit.setVerticalSpacing(4)
        grid_edit.addWidget(QLabel("Start"), 0, 0); grid_edit.addWidget(self.start_edit, 0, 1)
        grid_edit.addWidget(self.btn_start_m1, 0, 2); grid_edit.addWidget(self.btn_start_p1, 0, 3); grid_edit.addWidget(self.btn_prev_rec, 0, 4)
        grid_edit.addWidget(QLabel("End"), 1, 0); grid_edit.addWidget(self.end_edit, 1, 1)
        grid_edit.addWidget(self.btn_end_m1, 1, 2); grid_edit.addWidget(self.btn_end_p1, 1, 3); grid_edit.addWidget(self.btn_next_rec, 1, 4)
        right.addLayout(grid_edit)

        row_tail = QHBoxLayout()
        row_tail.addWidget(self.btn_head_preview); row_tail.addWidget(self.btn_tail_preview); row_tail.addWidget(self.btn_save_seg); row_tail.addStretch(1)
        right.addLayout(row_tail)

        right.addWidget(self.btn_extract); right.addWidget(self.btn_refine); right.addWidget(QLabel("로그")); right.addWidget(self.log_view, 1)
        mid.addLayout(right, 5)
        root_layout.addLayout(mid, 1)

        grid_io = QGridLayout(); grid_io.setHorizontalSpacing(6); grid_io.setVerticalSpacing(4)
        lbl_w = 60
        lab_intro = QLabel("인트로");  lab_intro.setMinimumWidth(lbl_w)
        lab_outro = QLabel("아웃트로"); lab_outro.setMinimumWidth(lbl_w)
        lab_out   = QLabel("출력 파일"); lab_out.setMinimumWidth(lbl_w)
        grid_io.addWidget(lab_intro, 0, 0, Qt.AlignRight); grid_io.addWidget(self.intro_edit, 0, 1); grid_io.addWidget(self.btn_intro_browse, 0, 2)
        grid_io.addWidget(lab_outro, 1, 0, Qt.AlignRight); grid_io.addWidget(self.outro_edit, 1, 1); grid_io.addWidget(self.btn_outro_browse, 1, 2)
        grid_io.addWidget(lab_out, 2, 0, Qt.AlignRight); grid_io.addWidget(self.output_edit, 2, 1); grid_io.addWidget(self.btn_output_browse, 2, 2)
        root_layout.addLayout(grid_io)

        row_render = QHBoxLayout(); row_render.setSpacing(8)
        row_render.addWidget(self.btn_render, 5); row_render.addWidget(self.progress_label, 0); row_render.addWidget(self.global_progress, 2)
        root_layout.addLayout(row_render)

        self.setCentralWidget(root)

    def _connect_signals(self):
        self.btn_download.clicked.connect(self.on_download)
        self.btn_vtt_browse.clicked.connect(self.on_vtt_browse)
        self.btn_vtt_download.clicked.connect(self.on_vtt_download)
        self.btn_vtt_apply.clicked.connect(self.on_vtt_apply)
        self.list_segments.itemSelectionChanged.connect(self.on_segment_selected)
        self.btn_prev.clicked.connect(lambda: self.navigate_segment(-1))
        self.btn_next.clicked.connect(lambda: self.navigate_segment(+1))
        self.btn_apply_clipboard.clicked.connect(self.on_apply_segments_from_clipboard)
        self.btn_copy_segments.clicked.connect(self.on_copy_segments_to_clipboard)
        self.btn_start_m1.clicked.connect(lambda: self.adjust_start(-1))
        self.btn_start_p1.clicked.connect(lambda: self.adjust_start(+1))
        self.btn_prev_rec.clicked.connect(self.on_prev_recommended)
        self.btn_end_m1.clicked.connect(lambda: self.adjust_end(-1))
        self.btn_end_p1.clicked.connect(lambda: self.adjust_end(+1))
        self.btn_next_rec.clicked.connect(self.on_next_recommended)
        self.btn_head_preview.clicked.connect(self.on_head_preview)
        self.btn_tail_preview.clicked.connect(self.on_tail_preview)
        self.btn_save_seg.clicked.connect(self.on_save_current_segment)
        self.btn_extract.clicked.connect(self.on_extract)
        self.btn_refine.clicked.connect(self.on_refine)
        self.btn_intro_browse.clicked.connect(lambda: self.browse_file(self.intro_edit, "동영상 (*.mp4 *.mov *.mkv)"))
        self.btn_outro_browse.clicked.connect(lambda: self.browse_file(self.outro_edit, "동영상 (*.mp4 *.mov *.mkv)"))
        self.btn_output_browse.clicked.connect(self.on_output_save)
        self.btn_render.clicked.connect(self.on_render)
        self.btn_copy_topic.clicked.connect(self.copy_topic_prompt)
        self.btn_copy_vtt.clicked.connect(lambda: self.copy_to_clipboard(VTT_ANALYSIS_PROMPT, "VTT 분석 프롬프트 복사 완료"))
        self.btn_copy_thumb.clicked.connect(lambda: self.copy_to_clipboard(THUMBNAIL_PROMPT, "썸네일 프롬프트 복사 완료"))

    # ---------- Busy/UI ----------
    def on_busy_changed(self, active: int): print(f"[busy] active={active}", flush=True)
    def ui(self, func, *args, **kwargs): self._invoker.sig_call.emit(func, args, kwargs)
    def log(self, text: str): self.log_view.append(text.rstrip())
    def progress_set(self, pct: int, message: str = ""):
        if self.global_progress.maximum() == 0: self.global_progress.setRange(0, 100)
        self.global_progress.setValue(max(0, min(100, pct))); self.progress_label.setText(message); self._last_progress_ts = time.time()
        print(f"[progress] {pct}% '{message}'", flush=True)
    def progress_reset(self):
        self.global_progress.setRange(0, 100); self.global_progress.setValue(0); self.progress_label.setText(""); print("[progress] RESET", flush=True)
    def progress_acquire(self, message: str = "") -> int:
        token = self.busy.acquire()
        if self.busy.active > 0:
            self.global_progress.setRange(0, 0); self.progress_label.setText(message); self._last_progress_ts = time.time()
            print(f"[progress] BUSY(acquire) -> '{message}'", flush=True)
        return token
    def progress_release(self, token: int, force_reset: bool = False):
        self.busy.release(token)
        if self.busy.active == 0 or force_reset: QTimer.singleShot(600, self.progress_reset)
    def _watchdog_tick(self):
        if self.global_progress.maximum() == 0 and time.time() - self._last_progress_ts > 15:
            print(f"[watchdog] Busy 고착 감지 (active={self.busy.active}) → RESET", flush=True); self.progress_reset()

    # ---------- 슬롯 ----------
    @Slot(list) 
    def set_sentence_bounds(self, bounds: list): 
        self.sentence_bounds = bounds
    @Slot(list)
    def _set_silence_spans(self, spans: list): 
        self.silence_spans = spans

    # ---------- 세그먼트 총 길이 표시 ----------
    def total_segments_duration(self) -> float:
        total = 0.0
        dur = self.media_duration
        for s, e in self.segments:
            ss = max(0.0, s)
            ee = e if dur is None else min(e, dur)
            if ee > ss: total += (ee - ss)
        return total
    def update_segments_header(self):
        tot = self.total_segments_duration()
        hh = int(tot // 3600); mm = int((tot % 3600) // 60); ss = int(round(tot - hh*3600 - mm*60))
        if hh > 0:
            txt = f"세그먼트 목록  (총 {hh:02d}:{mm:02d}:{ss:02d})"
        else:
            txt = f"세그먼트 목록  (총 {mm:02d}:{ss:02d})"
        self.lbl_segments.setText(txt)

    # ---------- 초기 세그먼트 ----------
    def _load_default_segments(self):
        try:
            segs = json.loads(DEFAULT_SEGMENTS_JSON)
            self.segments = [(hms_to_sec(s["start"]), hms_to_sec(s["end"])) for s in segs]
        except:
            self.segments = []
        self.refresh_segment_list()
        if self.segments:
            self.list_segments.setCurrentRow(0)
            self.apply_selected_to_fields(0)

    def refresh_segment_list(self):
        self.list_segments.clear()
        for i, (s, e) in enumerate(self.segments):
            self.list_segments.addItem(QListWidgetItem(f"{i+1}. {sec_to_hms(s)} ~ {sec_to_hms(e)}"))
        self.update_segments_header()

    def current_segment_index(self) -> int:
        idxs = self.list_segments.selectedIndexes()
        return idxs[0].row() if idxs else 0

    def apply_selected_to_fields(self, idx: int):
        if 0 <= idx < len(self.segments):
            s, e = self.segments[idx]
            self.start_edit.setText(sec_to_hms(s))
            self.end_edit.setText(sec_to_hms(e))

    # ---------- 파일 대화상자 ----------
    def browse_file(self, target_edit: QLineEdit, filter_str="모든 파일 (*.*)"):
        start_dir = os.path.dirname(self.output_edit.text().strip() or OUTPUT_DEFAULT) or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(self, "파일 선택", start_dir, filter_str)
        if path: target_edit.setText(path)

    def on_vtt_browse(self):
        start_dir = os.path.dirname(self.output_edit.text().strip() or OUTPUT_DEFAULT) or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(self, "VTT 선택", start_dir, "VTT (*.vtt)")
        if path:
            self.vtt_edit.setText(path)
            self.log(f"VTT 파일 선택: {path}")

    def on_output_save(self):
        path, _ = QFileDialog.getSaveFileName(self, "출력 파일 저장", self.output_edit.text() or OUTPUT_DEFAULT, "MP4 (*.mp4)")
        if path: self.output_edit.setText(path)

    # ---------- 다운로드 ----------
    def on_download(self):
        url = self.url_edit.text().strip()
        out_path = self.output_edit.text().strip() or OUTPUT_DEFAULT
        self.original_path = original_video_path_of(out_path)
        if os.path.exists(self.original_path):
            specs = get_video_specs(self.original_path)
            self.media_duration = get_media_duration(self.original_path)
            status = "존재"
            if specs:
                status += f" / {specs['width']}x{specs['height']} / {specs['r_frame_rate']}fps"
            if self.media_duration:
                status += f" / {self.media_duration:.1f}s"
            self.status_label.setText(status)
            self.log(f"원본 영상 확인 완료: {self.original_path}")
            return
        self.status_label.setText("다운로드 중...")
        tok = self.progress_acquire("다운로드 중...")
        def hook(d):
            if d.get('status') == 'downloading':
                total = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
                done  = d.get('downloaded_bytes') or 0
                pct = int(done*100/total) if total else 0
                self.ui(self.progress_set, pct, "다운로드 중...")
            elif d.get('status') == 'finished':
                self.ui(self.progress_set, 100, "다운로드 완료")
        def worker():
            ok, vtt_found = download_original_and_vtt(url, self.original_path, progress_cb=hook)
            if ok:
                self.media_duration = get_media_duration(self.original_path)
                txt = "다운로드 완료"
                if self.media_duration: txt += f" / 길이: {self.media_duration:.1f}s"
                self.ui(self.status_label.setText, txt)
                if vtt_found and not self.vtt_edit.text().strip():
                    self.ui(self.vtt_edit.setText, vtt_found)
                self.ui(self.log, "원본 다운로드 완료")
                self.ui(self.update_segments_header)
            else:
                self.ui(self.status_label.setText, "다운로드 실패")
                self.ui(self.log, "원본 다운로드 실패")
            self.ui(self.progress_release, tok)
        threading.Thread(target=worker, daemon=True).start()

    # ---------- VTT ----------
    def on_vtt_download(self):
        url = self.url_edit.text().strip()
        if not url:
            self.log("유튜브 URL을 먼저 입력하세요."); return
        out_dir = os.path.dirname(self.output_edit.text().strip() or OUTPUT_DEFAULT) or os.getcwd()
        self.status_label.setText("자막 다운로드 중...")
        tok = self.progress_acquire("자막 다운로드 중...")
        def worker():
            vtt = download_vtt_only(url, out_dir)
            if vtt and os.path.exists(vtt):
                self.ui(self.vtt_edit.setText, vtt)
                self.ui(self.status_label.setText, "자막 다운로드 완료")
                self.ui(self.progress_set, 100, "자막 다운로드 완료")
                self.ui(self.log, f"자막 다운로드 완료: {vtt}")
            else:
                self.ui(self.status_label.setText, "자막 다운로드 실패")
                self.ui(self.log, "자막 다운로드 실패")
            self.ui(self.progress_release, tok)
        threading.Thread(target=worker, daemon=True).start()

    def on_vtt_apply(self):
        vtt_path = self.vtt_edit.text().strip()
        if not vtt_path or not os.path.exists(vtt_path):
            self.log("유효한 VTT 파일을 선택하세요."); return
        if not HAS_WEBVTT:
            self.log("webvtt 패키지가 필요합니다. (pip install webvtt-py)"); return
        tok = self.progress_acquire("VTT 분석 중...")
        self.log("VTT 연결 시작"); print(f"[vtt-apply] token={tok} start", flush=True)
        def worker():
            released = False
            try:
                print("[vtt-apply] calling vtt_sentence_bounds()", flush=True)
                bounds = vtt_sentence_bounds(vtt_path, max_gap=0.4)
                print("[vtt-apply] vtt_sentence_bounds() returned", flush=True)
                self.ui(self.set_sentence_bounds, bounds)
                self.ui(self.log, f"VTT 연결 완료: 문장 경계 {len(bounds)}개")
                self.ui(self.progress_set, 100, "VTT 분석 완료")
                print("[vtt-apply] scheduled UI updates", flush=True)
            except Exception as ex:
                print(f"[vtt-apply] EXCEPTION: {ex}", flush=True)
                self.ui(self.log, f"VTT 연결 실패: {ex}")
                self.ui(self.progress_release, tok); released = True
            finally:
                try:
                    if not released:
                        print("[vtt-apply] finally -> release", flush=True)
                        self.ui(self.progress_release, tok)
                except Exception as ex2:
                    print(f"[vtt-apply] finally release exception: {ex2}", flush=True)
        threading.Thread(target=worker, daemon=True).start()

    # ---------- 세그먼트 조작 ----------
    def on_segment_selected(self):
        idx = self.current_segment_index()
        self.apply_selected_to_fields(idx)

    def navigate_segment(self, delta: int):
        if not self.segments: return
        idx = self.current_segment_index()
        idx = (idx + delta) % len(self.segments)
        self.list_segments.setCurrentRow(idx)
        self.apply_selected_to_fields(idx)

    def clamp_segment(self, s: float, e: float) -> Tuple[float,float]:
        if self.media_duration is not None:
            e = min(e, self.media_duration)
            s = max(0.0, min(s, max(0.0, self.media_duration - 0.1)))
        if e - s < 0.1:
            e = s + 0.1
            if self.media_duration is not None:
                e = min(e, self.media_duration)
        return s, e

    def adjust_start(self, delta_sec: int):
        if not self.segments: return
        idx = self.current_segment_index()
        s, e = self.segments[idx]
        s = max(0.0, s + delta_sec)
        s, e = self.clamp_segment(s, e)
        self.segments[idx] = (s, e)
        self.apply_selected_to_fields(idx); self.refresh_segment_list(); self.list_segments.setCurrentRow(idx)

    def adjust_end(self, delta_sec: int):
        if not self.segments: return
        idx = self.current_segment_index()
        s, e = self.segments[idx]
        e += delta_sec
        s, e = self.clamp_segment(s, e)
        self.segments[idx] = (s, e)
        self.apply_selected_to_fields(idx); self.refresh_segment_list(); self.list_segments.setCurrentRow(idx)

    def on_prev_recommended(self):
        if not self.segments: self.log("세그먼트가 없습니다."); return
        if not (self.silence_spans or self.sentence_bounds):
            self.log("먼저 [무음/문장 경계 추출]을 실행하세요."); return
        idx = self.current_segment_index()
        s, e = self.segments[idx]
        ns = prev_recommended_start(s, self.silence_spans, self.sentence_bounds)
        if ns == s: self.log("이전 추천 경계가 없습니다."); return
        ns = max(0.0, ns)
        if self.media_duration is not None: ns = min(ns, max(0.0, self.media_duration - 0.1))
        if e - ns < 0.1: self.log("추천 시작 지점이 너무 늦어 최소 길이를 만족하지 못합니다."); return
        self.segments[idx] = (ns, e)
        self.apply_selected_to_fields(idx); self.refresh_segment_list(); self.list_segments.setCurrentRow(idx)
        self.log(f"Start → 추천 {sec_to_hms(ns)}")

    def on_next_recommended(self):
        if not self.segments: self.log("세그먼트가 없습니다."); return
        if not (self.silence_spans or self.sentence_bounds):
            self.log("먼저 [무음/문장 경계 추출]을 실행하세요."); return
        idx = self.current_segment_index()
        s, e = self.segments[idx]
        ne = next_recommended_end(e, self.silence_spans, self.sentence_bounds, window=NEXT_END_EXTRA_WINDOW)
        if ne == e: self.log("더 먼 추천 경계가 없습니다."); return
        if self.media_duration is not None: ne = min(ne, self.media_duration)
        if ne - s < 0.1: self.log("추천 종료 지점이 너무 이르러 최소 길이를 만족하지 못합니다."); return
        self.segments[idx] = (s, ne)
        self.apply_selected_to_fields(idx); self.refresh_segment_list(); self.list_segments.setCurrentRow(idx)
        self.log(f"End → 추천 {sec_to_hms(ne)}")

    # ---------- 미리듣기 (정확도 개선) ----------
    def _play_preview(self, start: float, dur: float, media_path: str):
        # 정확한 길이 확보를 위해 -ss/-t를 입력 뒤에 배치(정밀 탐색)
        cmd = f'ffplay -loglevel error -autoexit -nodisp -i "{media_path}" -ss {start:.3f} -t {dur:.3f}'
        # 더 엄격하게 하려면 atrim 사용 (주석 해제해서 사용 가능)
        # cmd = f'ffplay -loglevel error -autoexit -nodisp -i "{media_path}" -ss {start:.3f} -af "atrim=0:{dur:.3f},asetpts=N/SR/TB"'
        threading.Thread(target=lambda: os.system(cmd), daemon=True).start()

    def on_head_preview(self):
        if not self.segments: self.log("세그먼트가 없습니다."); return
        out_path = self.output_edit.text().strip() or OUTPUT_DEFAULT
        self.original_path = original_video_path_of(out_path)
        if not (self.original_path and os.path.exists(self.original_path)):
            self.log("원본 영상이 없습니다. 먼저 다운로드 하세요."); return
        idx = self.current_segment_index()
        s, e = self.segments[idx]
        start = max(0.0, min(s, self.media_duration or s))
        dur = min(HEAD_PREVIEW_SEC, max(0.1, e - s))
        self._play_preview(start, dur, self.original_path)
        self.log(f"처음 미리듣기: {sec_to_hms(start)} ~ {sec_to_hms(start+dur)} ({dur:.2f}s)")

    def on_tail_preview(self):
        if not self.segments: self.log("세그먼트가 없습니다."); return
        out_path = self.output_edit.text().strip() or OUTPUT_DEFAULT
        self.original_path = original_video_path_of(out_path)
        if not (self.original_path and os.path.exists(self.original_path)):
            self.log("원본 영상이 없습니다. 먼저 다운로드 하세요."); return
        idx = self.current_segment_index()
        s, e = self.segments[idx]
        e = min(e, self.media_duration or e)
        start = max(s, e - TAIL_PREVIEW_SEC)
        dur = max(0.1, e - start)
        self._play_preview(start, dur, self.original_path)
        self.log(f"마지막 미리듣기: {sec_to_hms(start)} ~ {sec_to_hms(e)} ({dur:.2f}s)")

    def on_save_current_segment(self):
        try:
            s = hms_to_sec(self.start_edit.text().strip()); e = hms_to_sec(self.end_edit.text().strip())
            s, e = self.clamp_segment(s, e)
            idx = self.current_segment_index()
            self.segments[idx] = (s, e)
            self.refresh_segment_list(); self.list_segments.setCurrentRow(idx)
            self.log(f"세그먼트 #{idx+1} 저장: {sec_to_hms(s)} ~ {sec_to_hms(e)}")
        except Exception as ex:
            self.log(f"세그먼트 저장 오류: {ex}")

    # ---------- 추출/보정 ----------
    def on_extract(self):
        out_path = self.output_edit.text().strip() or OUTPUT_DEFAULT
        self.original_path = original_video_path_of(out_path)
        if not (self.original_path and os.path.exists(self.original_path)):
            self.log("원본 영상이 없습니다. 먼저 다운로드 하세요."); return
        if self.media_duration is None: self.media_duration = get_media_duration(self.original_path)
        tok = self.progress_acquire("무음/문장 경계 추출 중..."); self.log("무음/문장 경계 추출 시작")
        def worker():
            try:
                spans = get_silence_spans(self.original_path, SILENCE_DB, SILENCE_DUR,
                                          media_duration=self.media_duration, progress_cb=self.ui_progress_update, timeout_factor=1.4)
                self.ui(self._set_silence_spans, spans); self.ui(self.log, f"무음 구간 수: {len(spans)}")
                vtt = self.vtt_edit.text().strip() or None
                if vtt and os.path.exists(vtt) and HAS_WEBVTT:
                    self.ui(self.progress_set, 90 if self.media_duration else 100, "VTT 분석 중...")
                    bounds = vtt_sentence_bounds(vtt, max_gap=0.4)
                    self.ui(self.set_sentence_bounds, bounds); self.ui(self.log, f"문장 경계 수: {len(bounds)}")
                else:
                    self.ui(self.set_sentence_bounds, []); self.ui(self.log, "VTT 미지정 또는 webvtt 미설치 → 문장 경계 생략")
                self.ui(self.progress_set, 100, "경계 추출 완료")
            except TimeoutError as te:
                self.ui(self.log, f"[경계 추출 타임아웃] {te}")
            except Exception as ex:
                self.ui(self.log, f"[경계 추출 오류] {ex}")
            finally:
                self.ui(self.progress_release, tok); self.ui(self.log, "경계 추출 종료")
        threading.Thread(target=worker, daemon=True).start()

    def ui_progress_update(self, pct: int, msg: str = ""): self.ui(self.progress_set, pct, msg)

    def on_refine(self):
        if not self.segments: self.log("세그먼트가 없습니다."); return
        if not self.silence_spans: self.log("무음/문장 경계가 없습니다. [무음/문장 경계 추출]을 먼저 실행하세요."); return
        new_segments, reasons = refine_segments(self.segments, self.silence_spans, self.sentence_bounds,
                                                start_win=START_WINDOW, end_win=END_WINDOW, min_len=MIN_LEN)
        if self.media_duration is not None:
            clamped = []
            for (rs, re) in new_segments:
                rs = max(0.0, rs); re = min(self.media_duration, re)
                if re - rs < 0.1: re = min(self.media_duration, rs + 0.1)
                clamped.append((rs, re))
            new_segments = clamped
        self.segments = new_segments; self.refresh_segment_list()
        if self.segments:
            idx = min(self.current_segment_index(), len(self.segments)-1)
            self.list_segments.setCurrentRow(idx); self.apply_selected_to_fields(idx)
        for i, (s, e) in enumerate(self.segments):
            r1, r2 = reasons[i]; self.log(f"[보정] #{i+1} → {sec_to_hms(s)}~{sec_to_hms(e)}   (start:{r1}, end:{r2})")

    # ---------- 렌더 ----------
    def on_render(self):
        out_path = self.output_edit.text().strip() or OUTPUT_DEFAULT
        self.original_path = original_video_path_of(out_path)
        if not (self.original_path and os.path.exists(self.original_path)):
            self.log("원본 영상이 없습니다. 먼저 다운로드 하세요."); return
        specs = get_video_specs(self.original_path)
        if not specs: self.log("원본 스펙 확인 실패"); return
        if self.media_duration is None: self.media_duration = get_media_duration(self.original_path)
        w, h = specs['width'], specs['height']; fps, sar = specs['r_frame_rate'], specs['sar']
        audio_rate = specs['sample_rate']; has_audio = audio_rate is not None
        safe_segments = []
        for (s,e) in self.segments:
            if self.media_duration is not None:
                s = max(0.0, min(s, max(0.0, self.media_duration - 0.1))); e = min(e, self.media_duration)
                if e - s < 0.1: e = min(self.media_duration, s + 0.1)
            safe_segments.append((s,e))
        self.segments = safe_segments; work_dir = guess_workdir(out_path); self.log(f"작업 폴더: {work_dir}")
        total_steps = len(self.segments) + 1; tok = self.progress_acquire("렌더링 준비 중...")
        def worker():
            try:
                clip_paths = []
                for i, (s,e) in enumerate(self.segments, start=1):
                    cp = os.path.join(work_dir, f"clip_{i:02d}.mp4").replace("\\","/")
                    ok = cut_clip(self.original_path, s, e, cp, fps, w, h, sar, audio_rate,
                                  media_duration=self.media_duration, l_cut=L_CUT_SEC, afade_out=AFADE_OUT)
                    if not ok: self.ui(self.log, f"클립 생성 실패 #{i}"); return
                    clip_paths.append(cp); pct = int(i * 100 / total_steps)
                    self.ui(self.progress_set, pct, f"클립 생성 {i}/{len(self.segments)}")
                intro = self.intro_edit.text().strip() or None; outro = self.outro_edit.text().strip() or None
                self.ui(self.progress_set, min(99, int(100*len(self.segments)/total_steps)), "최종 병합 중...")
                ok, out = concat_all(intro, clip_paths, outro, has_audio, fps, w, h, sar, audio_rate, out_path)
                if ok: self.ui(self.log, f"최종 생성 완료: {out_path}"); self.ui(self.progress_set, 100, "렌더 완료")
                else: self.ui(self.log, f"[병합 실패]\n{out or ''}")
            finally:
                self.ui(self.progress_release, tok)
        threading.Thread(target=worker, daemon=True).start()

    # ---------- 프롬프트 & 클립보드 ----------
    def copy_topic_prompt(self):
        url = self.url_edit.text().strip()
        prompt = TOPIC_PROMPT.replace("분석 대상 영상 : {}", f"분석 대상 영상 : {url}")
        self.copy_to_clipboard(prompt, "분석 프롬프트 복사 완료")
    def copy_to_clipboard(self, text: str, ok_log: str):
        QGuiApplication.clipboard().setText(text); self.log(ok_log)
    def on_apply_segments_from_clipboard(self):
        txt = (QGuiApplication.clipboard().text() or "").strip()
        txt = re.sub(r"^```(json)?", "", txt, flags=re.IGNORECASE).strip(); txt = re.sub(r"```$", "", txt).strip()
        try:
            data = json.loads(txt)
            if not isinstance(data, list): raise ValueError("JSON 최상위가 배열이 아닙니다.")
            new_segments = []
            for i, it in enumerate(data):
                if not isinstance(it, dict): raise ValueError(f"{i}번째 항목이 객체가 아닙니다.")
                if "start" not in it or "end" not in it: raise ValueError(f"{i}번째 항목에 start/end가 없습니다.")
                s = hms_to_sec(str(it["start"])); e = hms_to_sec(str(it["end"]))
                if e - s <= 0: raise ValueError(f"{i}번째 항목: end가 start보다 작거나 같습니다.")
                new_segments.append((s,e))
            self.segments = new_segments; self.refresh_segment_list()
            if self.segments: self.list_segments.setCurrentRow(0); self.apply_selected_to_fields(0)
            self.log(f"클립보드 세그먼트 적용 완료: {len(self.segments)}개 업데이트")
        except Exception as ex:
            self.log(f"세그먼트 적용 실패: {ex}")
    def on_copy_segments_to_clipboard(self):
        arr = [{"start": sec_to_hms(s), "end": sec_to_hms(e)} for s, e in self.segments]
        QGuiApplication.clipboard().setText(json.dumps(arr, ensure_ascii=False, indent=2))
        self.log("현재 세그먼트 정보를 클립보드로 복사했습니다.")

def main():
    app = QApplication(sys.argv); w = MainWindow(); w.show(); sys.exit(app.exec())

if __name__ == "__main__":
    main()
