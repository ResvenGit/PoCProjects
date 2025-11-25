# -*- coding: utf-8 -*-
"""PySide6 UI layout and interactions."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import List, Optional, Tuple

from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QGuiApplication
from pathlib import Path

from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..analysis.segments import SegmentAnalyzer
from ..config import (
    DEFAULT_SEGMENTS_JSON,
    DEFAULT_URL,
    HEAD_PREVIEW_SEC,
    INTRO_DEFAULT,
    OUTRO_DEFAULT,
    OUTPUT_DEFAULT,
    TAIL_PREVIEW_SEC,
    THUMBNAIL_PROMPT,
    TOPIC_PROMPT,
    VTT_ANALYSIS_PROMPT,
)
from ..controller import VideoEditingController
from ..utils.paths import guess_workdir, original_video_path_of
from ..utils.timecode import hms_to_sec, sec_to_hms

LAST_URL_FILE = Path(__file__).resolve().parent.parent / "last_url.txt"

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
        self.controller = VideoEditingController()
        self.setWindowTitle("Auto Clip Refinement & Editor - PySide6")
        self.resize(1180, 780)

        # 상태
        self.segments: List[Tuple[float,float]] = []
        self.silence_spans: List[Tuple[float,float]] = []
        self.sentence_bounds: List[float] = []
        self.media_duration: Optional[float] = None
        self.original_path: Optional[str] = None
        self.topic_results: List[str] = []
        self._topic_list_updating = False
        self._last_url_file = LAST_URL_FILE
        self._initial_url = self._load_saved_url() or DEFAULT_URL

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
        self.refresh_topic_list()
        self._connect_signals()
        self._load_default_segments()

    # ---------- Widgets ----------
    def _build_widgets(self):
        self.url_edit = QLineEdit(self._initial_url)
        self.btn_download = QPushButton("원본 확인/다운로드")
        self.status_label = QLabel("확인 전")

        self.btn_copy_topic = QPushButton("분석 프롬프트 복사")
        self.btn_copy_vtt   = QPushButton("VTT 분석 프롬프트 복사")
        self.btn_copy_thumb = QPushButton("썸네일 프롬프트 복사")
        for b in (self.btn_copy_topic, self.btn_copy_vtt, self.btn_copy_thumb):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setMinimumHeight(36)
        self.btn_apply_topic_results = QPushButton("분석 결과 반영")
        self.btn_remove_topic = QPushButton("선택 토픽 제거")
        self.btn_add_topic = QPushButton("토픽 추가")
        self.lbl_topics = QLabel("토픽 목록 (0)")
        self.list_topics = QListWidget()
        self.list_topics.setAlternatingRowColors(True)
        self.list_topics.setEditTriggers(
            QAbstractItemView.DoubleClicked
            | QAbstractItemView.SelectedClicked
            | QAbstractItemView.EditKeyPressed
        )
        self.list_topics.setMaximumHeight(170)

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
        row_topics = QHBoxLayout(); row_topics.setSpacing(6)
        row_topics.addWidget(self.lbl_topics)
        row_topics.addStretch(1)
        row_topics.addWidget(self.btn_add_topic)
        row_topics.addWidget(self.btn_remove_topic)
        row_topics.addWidget(self.btn_apply_topic_results)
        root_layout.addLayout(row_topics)
        root_layout.addWidget(self.list_topics)

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
        self.btn_copy_vtt.clicked.connect(self.copy_vtt_prompt)
        self.btn_copy_thumb.clicked.connect(lambda: self.copy_to_clipboard(THUMBNAIL_PROMPT, "썸네일 프롬프트 복사 완료"))
        self.btn_apply_topic_results.clicked.connect(self.on_apply_topic_results)
        self.btn_add_topic.clicked.connect(self.on_add_topic)
        self.btn_remove_topic.clicked.connect(self.on_remove_topic)
        self.list_topics.itemChanged.connect(self.on_topic_item_changed)

    # ---------- URL Persistence ----------
    def _load_saved_url(self) -> Optional[str]:
        path = getattr(self, "_last_url_file", LAST_URL_FILE)
        try:
            if path.exists():
                data = path.read_text(encoding="utf-8").strip()
                if data:
                    return data
        except Exception as ex:
            print(f"[last-url] load failed: {ex}", flush=True)
        return None

    def _save_last_url(self, url: str):
        if not url:
            return
        path = getattr(self, "_last_url_file", LAST_URL_FILE)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(url.strip(), encoding="utf-8")
        except Exception as ex:
            print(f"[last-url] save failed: {ex}", flush=True)

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

    # ---------- 토픽 목록 ----------
    def refresh_topic_list(self):
        self._topic_list_updating = True
        self.list_topics.clear()
        count = len(self.topic_results)
        self.lbl_topics.setText(f"토픽 목록 ({count})")
        if not count:
            placeholder = QListWidgetItem("분석 결과를 반영하면 토픽이 표시됩니다.")
            placeholder.setFlags(Qt.NoItemFlags)
            self.list_topics.addItem(placeholder)
            self._topic_list_updating = False
            return
        for topic in self.topic_results:
            item = QListWidgetItem(topic)
            item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable)
            self.list_topics.addItem(item)
        self._topic_list_updating = False

    def on_add_topic(self):
        text, ok = QInputDialog.getText(self, "토픽 추가", "새 토픽 이름을 입력하세요.")
        if not ok:
            return
        topic = text.strip()
        if not topic:
            self.log("토픽 이름을 입력해야 합니다.")
            return
        self.topic_results.append(topic)
        self.refresh_topic_list()
        self.list_topics.setCurrentRow(len(self.topic_results) - 1)
        self.log(f"토픽을 추가했습니다: {topic}")

    def on_topic_item_changed(self, item: QListWidgetItem):
        if self._topic_list_updating:
            return
        row = self.list_topics.row(item)
        if not (0 <= row < len(self.topic_results)):
            return
        new_text = item.text().strip()
        if not new_text:
            self._topic_list_updating = True
            item.setText(self.topic_results[row])
            self._topic_list_updating = False
            self.log("토픽 이름은 비울 수 없습니다.")
            return
        if new_text == self.topic_results[row]:
            return
        self.topic_results[row] = new_text
        self.log(f"토픽 #{row+1} 이름을 수정했습니다.")

    def on_remove_topic(self):
        if not self.topic_results:
            self.log("제거할 토픽이 없습니다.")
            return
        idxs = self.list_topics.selectedIndexes()
        if not idxs:
            self.log("먼저 토픽 목록에서 항목을 선택하세요.")
            return
        idx = idxs[0].row()
        if not (0 <= idx < len(self.topic_results)):
            self.log("선택한 토픽을 찾을 수 없습니다.")
            return
        removed = self.topic_results.pop(idx)
        self.refresh_topic_list()
        if self.topic_results:
            self.list_topics.setCurrentRow(min(idx, len(self.topic_results) - 1))
        self.log(f"토픽을 제거했습니다: {removed}")

    def on_apply_topic_results(self):
        raw = (QGuiApplication.clipboard().text() or "").strip()
        if not raw:
            self.log("클립보드에 JSON 데이터가 없습니다.")
            return
        cleaned = re.sub(r"^```(json)?", "", raw, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as ex:
            self.log(f"분석 결과 JSON 파싱 실패: {ex}")
            return
        proposals = self._coerce_topic_payload(payload)
        if not proposals:
            self.log("JSON에서 video_proposals 데이터를 찾을 수 없습니다.")
            return
        topics: List[str] = []
        for idx, entry in enumerate(proposals, start=1):
            topic_value = None
            if isinstance(entry, dict):
                topic_value = entry.get("topic")
            elif isinstance(entry, str):
                topic_value = entry
            if topic_value is None:
                self.log(f"{idx}번째 항목에 topic 정보가 없습니다.")
                continue
            topic_text = str(topic_value).strip()
            if not topic_text:
                self.log(f"{idx}번째 topic 문자열이 비어 있습니다.")
                continue
            topics.append(topic_text)
        if not topics:
            self.log("유효한 topic 문자열을 찾지 못했습니다.")
            return
        self.topic_results = topics
        self.refresh_topic_list()
        self.log(f"분석 결과 토픽 {len(topics)}건을 반영했습니다.")

    def _coerce_topic_payload(self, payload):
        if isinstance(payload, dict):
            for key in ("video_proposals", "proposals", "topics"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
            return [payload]
        if isinstance(payload, list):
            return payload
        return []

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
        if url:
            self._save_last_url(url)
        out_path = self.output_edit.text().strip() or OUTPUT_DEFAULT
        self.original_path = original_video_path_of(out_path)
        if os.path.exists(self.original_path):
            specs = self.controller.get_video_specs(self.original_path)
            self.media_duration = self.controller.get_media_duration(self.original_path)
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
            ok, vtt_found = self.controller.download_original_and_vtt(url, self.original_path, progress_cb=hook)
            if ok:
                self.media_duration = self.controller.get_media_duration(self.original_path)
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
            vtt = self.controller.download_vtt_only(url, out_dir)
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
        if not SegmentAnalyzer.has_webvtt():
            self.log("webvtt 패키지가 필요합니다. (pip install webvtt-py)"); return
        tok = self.progress_acquire("VTT 분석 중...")
        self.log("VTT 연결 시작"); print(f"[vtt-apply] token={tok} start", flush=True)
        def worker():
            released = False
            try:
                print("[vtt-apply] calling sentence_bounds()", flush=True)
                bounds = self.controller.sentence_bounds(vtt_path, max_gap=0.4)
                print("[vtt-apply] sentence_bounds() returned", flush=True)
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
        ns = self.controller.prev_recommended_start(s, self.silence_spans, self.sentence_bounds)
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
        ne = self.controller.next_recommended_end(e, self.silence_spans, self.sentence_bounds)
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
        if self.media_duration is None: self.media_duration = self.controller.get_media_duration(self.original_path)
        tok = self.progress_acquire("무음/문장 경계 추출 중..."); self.log("무음/문장 경계 추출 시작")
        def worker():
            try:
                spans = self.controller.detect_silence(
                    self.original_path,
                    media_duration=self.media_duration,
                    progress_cb=self.ui_progress_update,
                    timeout_factor=1.4,
                )
                self.ui(self._set_silence_spans, spans); self.ui(self.log, f"무음 구간 수: {len(spans)}")
                vtt = self.vtt_edit.text().strip() or None
                if vtt and os.path.exists(vtt) and SegmentAnalyzer.has_webvtt():
                    self.ui(self.progress_set, 90 if self.media_duration else 100, "VTT 분석 중...")
                    bounds = self.controller.sentence_bounds(vtt, max_gap=0.4)
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
        new_segments, reasons = self.controller.refine_segments(
            self.segments, self.silence_spans, self.sentence_bounds
        )
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
        specs = self.controller.get_video_specs(self.original_path)
        if not specs: self.log("원본 스펙 확인 실패"); return
        if self.media_duration is None: self.media_duration = self.controller.get_media_duration(self.original_path)
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
                    ok = self.controller.cut_clip(
                        self.original_path,
                        s,
                        e,
                        cp,
                        fps,
                        w,
                        h,
                        sar,
                        audio_rate,
                        media_duration=self.media_duration,
                    )
                    if not ok: self.ui(self.log, f"클립 생성 실패 #{i}"); return
                    clip_paths.append(cp); pct = int(i * 100 / total_steps)
                    self.ui(self.progress_set, pct, f"클립 생성 {i}/{len(self.segments)}")
                intro = self.intro_edit.text().strip() or None; outro = self.outro_edit.text().strip() or None
                self.ui(self.progress_set, min(99, int(100*len(self.segments)/total_steps)), "최종 병합 중...")
                ok, out = self.controller.concat_all(
                    intro, clip_paths, outro, has_audio, fps, w, h, sar, audio_rate, out_path
                )
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
    def copy_vtt_prompt(self):
        topics = [t.strip() for t in self.topic_results if t.strip()]
        if topics:
            topic_block = "\n".join(f"- {t}" for t in topics)
        else:
            topic_block = "토픽을 여기 정리하세요"
            self.log("토픽 목록이 비어 있어 기본 안내 문구로 복사합니다.")
        prompt = VTT_ANALYSIS_PROMPT.replace("{주제}", topic_block)
        self.copy_to_clipboard(prompt, "VTT 분석 프롬프트 복사 완료")
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

