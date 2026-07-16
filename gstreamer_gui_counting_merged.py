"""
Factory Vision Control Dashboard — LIVE BACKEND INTEGRATION
A modern futuristic industrial PyQt5 dashboard for factory control rooms,
wired to a real YOLO + DeepSORT bag-counting pipeline.

Requirements:
    pip install PyQt5 opencv-python ultralytics deep-sort-realtime numpy
    (torch/torchvision as required by ultralytics/deep-sort-realtime)

Architecture:
    BagCounterThread (QThread) owns the model, tracker, video capture/writer,
    and all counting state. It runs the detect -> track -> count loop off the
    GUI thread and emits Qt signals (frame_ready, stats_updated,
    status_message, source_ended) that the main window connects to. This
    keeps the UI responsive regardless of inference speed, and is the only
    architecture that's safe with Qt — never touch QWidgets from a worker
    thread directly.
"""

import sys
import os
import time
import math
import random
from datetime import datetime, timedelta
from collections import defaultdict, deque

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QSizePolicy, QGraphicsDropShadowEffect
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation,
    QEasingCurve, QRect, pyqtProperty, QSequentialAnimationGroup,
    QParallelAnimationGroup
)
from PyQt5.QtGui import (
    QColor, QPainter, QPen, QBrush, QFont, QFontDatabase,
    QLinearGradient, QRadialGradient, QPainterPath, QPixmap,
    QImage
)


# ══════════════════════════════════════════════════════════════════════════
#  BACKEND CONFIG — edit for your deployment
# ══════════════════════════════════════════════════════════════════════════
MODEL_PATH   = r"C:\Users\siddh\Desktop\adhesive_bag\runs\detect\runs\added_whitebags\added_white_bag_detectort-2\weights\best.pt"   # TensorRT engine (.pt also works for testing off-Jetson)
VIDEO_SOURCE = r"C:\Users\siddh\Desktop\adhesive_bag\cctv_test.mp4"  # file path, or an int (e.g. 0) for a live camera
OUTPUT_PATH  = r"/Users/apple/Downloads/A tileadhisive cctv footage/results/processed_output.mp4"
SAVE_OUTPUT_VIDEO = True   # set False to skip writing an annotated copy to disk

FRAME_W, FRAME_H = 640, 480
USE_HW_VIDEO_IO   = True   # Jetson NVDEC/NVENC via GStreamer; auto-falls back to plain OpenCV if unavailable
# ══════════════════════════════════════════════════════════════════════════


# ─── Color Palette ────────────────────────────────────────────────────────────
# Backgrounds
BG_DEEP        = "#040917"   # Main background
BG_SECONDARY   = "#0A1022"   # Secondary background
BG_CARD        = "#162B4F"   # Card background
BG_CARD_HOVER  = "#1D3660"   # Card hover (slightly lighter than card)
BG_DARK_PANEL  = "#101A33"   # Dark panel
BG_SIDEBAR     = "#07101F"   # Sidebar / action bar

# Neon Blue Accents
ACCENT_BLUE    = "#00A8FF"   # Primary neon blue
ACCENT_CYAN    = "#00D4FF"   # Bright cyan blue (video glow)
ELECTRIC_BLUE  = "#1E90FF"   # Electric blue glow
BORDER_BLUE    = "#162BFF"   # Dashboard border blue
CARD_BLUE      = "#3B82F6"   # Card highlight blue

# Success / Live Status
LIVE_GREEN     = "#20DB14"   # Live indicator green
SUCCESS_GREEN  = "#209D24"   # Success green
BTN_START      = "#00C853"   # Start button green
NEON_GREEN     = "#00FF88"   # Bright neon green accent

# Warning / Overlap
ORANGE_ACCENT  = "#9B6611"   # Orange accent
AMBER_GLOW     = "#FFB300"   # Amber glow (overlap card accent)
GOLD_HIGH      = "#FFC107"   # Gold highlight

# Stop / Danger
BTN_STOP       = "#FF4D4F"   # Stop button red
DARK_RED       = "#D32F2F"   # Dark red
NEON_RED       = "#FF1744"   # Neon red glow

# Text
TEXT_PRIMARY   = "#E8ECF0"   # Primary text
TEXT_SECONDARY = "#A3B4CB"   # Secondary text
TEXT_MUTED     = "#9CA2A8"   # Muted text
TEXT_DISABLED  = "#76889A"   # Disabled text

# Industrial Box Colors (for bounding box overlays)
BOX_BROWN      = "#674630"   # Box brown
BOX_DARK_BROWN = "#583226"   # Dark box brown

# Convenience aliases kept for backward-compat with widget code
ACCENT_GREEN   = NEON_GREEN
ACCENT_AMBER   = AMBER_GLOW
ACCENT_RED     = BTN_STOP
BORDER_DIM     = "#1A2B4A"
BORDER_GLOW    = "#1E3A6F"


# ─── Utility: drop-shadow helper ──────────────────────────────────────────────
def make_shadow(color: str, blur: int = 24, x: int = 0, y: int = 4) -> QGraphicsDropShadowEffect:
    eff = QGraphicsDropShadowEffect()
    eff.setBlurRadius(blur)
    eff.setOffset(x, y)
    eff.setColor(QColor(color))
    return eff


# ═════════════════════════════════════════════════════════════════════════
#  BACKEND — real detection / tracking / counting pipeline, off the GUI thread
# ═════════════════════════════════════════════════════════════════════════
class BagCounterThread(QThread):
    """Owns the model, tracker, video I/O, and all counting state. Runs the
    full detect -> track -> line-cross-count loop and emits frame + stat
    updates to the GUI thread via signals. Never touch QWidgets from here."""

    frame_ready    = pyqtSignal(QImage)
    stats_updated  = pyqtSignal(int, int)   # total_count, overlap_count
    status_message = pyqtSignal(str)
    source_ended   = pyqtSignal()

    # Tuning knobs (unchanged business logic from the standalone script)
    LINE_Y_OFFSET             = FRAME_H//2 +30
    SMOOTH_ALPHA              = 0.25
    FORWARD_CONFIRM_FRAMES    = 2
    BACKWARD_CONFIRM_FRAMES   = 8
    FORWARD_MIN_DISPLACEMENT  = 10
    GRAVEYARD_TTL             = 35
    GRAVEYARD_MATCH_PX        = 70
    MERGE_RADIUS_GREEN        = 25
    MERGE_RADIUS_PINK         = 30
    COUNT_DEDUP_RADIUS        = 20
    FLASH_DURATION            = 0.7

    def __init__(self, model_path, video_source, output_path=None,
                 frame_w=FRAME_W, frame_h=FRAME_H, parent=None):
        super().__init__(parent)
        self.model_path   = model_path
        self.video_source = video_source
        self.output_path  = output_path
        self.frame_w, self.frame_h = frame_w, frame_h
        self.line_y = (frame_h // 2) + self.LINE_Y_OFFSET

        self._running = False

        # Counting state
        self.count = 0
        self.overlap_count = 0
        self.flash_event = None
        self.flash_time = 0.0
        self.frame_number = 0

        self.track_coords         = {}
        self.track_confirmed_side = {}
        self.track_is_overlap     = defaultdict(bool)
        self.pending               = {}
        self.graveyard              = {}
        self.recent_commits        = deque(maxlen=20)
        self.counted_tracks        = {}

        self.model = None
        self.tracker = None
        self.overlap_class_id = 1
        self.cap = None
        self.out = None
        self.src_fps = 25.0

    # ── lifecycle ──────────────────────────────────────────────────────────
    def stop(self):
        """Ask the loop to exit after the current frame. Call .wait() from
        the GUI thread afterward to block until it actually has."""
        self._running = False

    def run(self):
        try:
            self._load_backend()
        except Exception as e:
            self.status_message.emit(f"ERROR: {e}")
            return

        self._running = True
        self.status_message.emit("Running")

        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame.shape[1] != self.frame_w or frame.shape[0] != self.frame_h:
                frame = cv2.resize(frame, (self.frame_w, self.frame_h))

            self.frame_number += 1
            self._process_frame(frame)  # draws overlays in-place, updates self.count/overlap_count

            if self.out is not None:
                self.out.write(frame)

            self.frame_ready.emit(self._to_qimage(frame))
            self.stats_updated.emit(self.count, self.overlap_count)

        self._cleanup()
        self.source_ended.emit()

    def _load_backend(self):
        self.status_message.emit("Loading model…")
        from ultralytics import YOLO
        from deep_sort_realtime.deepsort_tracker import DeepSort

        # TensorRT engine: no .to(), no .fuse(), no half=/device= on calls —
        # all of that is baked in at export time.
        self.model = YOLO(self.model_path, task="detect")
        self.class_names = self.model.names
        self.overlap_class_id = next(
            (k for k, v in self.class_names.items() if v == 'overlapped'), 1)

        self.tracker = DeepSort(
            max_age=20, n_init=3, max_cosine_distance=0.60, nn_budget=100,
            max_iou_distance=0.7, embedder="mobilenet", half=True,
            bgr=True, embedder_gpu=True,
        )

        self.status_message.emit("Opening video source…")
        self.src_fps = self._probe_fps(self.video_source)
        self.cap, _ = self._open_capture(self.video_source, self.frame_w, self.frame_h)
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.video_source}")

        if self.output_path:
            self.out = self._open_writer(self.output_path, self.frame_w, self.frame_h, self.src_fps)

    def _cleanup(self):
        if self.cap is not None:
            self.cap.release()
        if self.out is not None:
            self.out.release()

    # ── video I/O helpers ─────────────────────────────────────────────────
    @staticmethod
    def _probe_fps(source):
        """Ground-truth fps via the plain OpenCV backend. Do not trust
        cap.get(CAP_PROP_FPS) on a custom GStreamer appsink pipeline —
        it's frequently wrong/zero there, which silently makes output play
        faster or slower than real time."""
        if isinstance(source, int):
            return 30.0
        probe = cv2.VideoCapture(source)
        fps = probe.get(cv2.CAP_PROP_FPS)
        probe.release()
        if not fps or fps <= 1 or fps > 240:
            fps = 25.0
        return fps

    def _open_capture(self, source, w, h):
        if isinstance(source, int):
            return cv2.VideoCapture(source), False
        if USE_HW_VIDEO_IO:
            gst_in = (
                f'filesrc location="{source}" ! qtdemux ! h264parse ! '
                f'nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx,width={w},height={h} ! '
                f'videoconvert ! video/x-raw,format=BGR ! '
                f'queue max-size-buffers=200 leaky=downstream ! appsink drop=0 sync=0'
            )
            cap = cv2.VideoCapture(gst_in, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                return cap, True
        return cv2.VideoCapture(source), False

    def _open_writer(self, path, w, h, fps):
        out_dir = os.path.dirname(path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if USE_HW_VIDEO_IO:
            gst_out = (
                f'appsrc ! videoconvert ! video/x-raw,format=BGRx ! '
                f'nvvidconv ! nvv4l2h264enc bitrate=8000000 ! h264parse ! '
                f'qtmux ! filesink location="{path}"'
            )
            writer = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, fps, (w, h), True)
            if writer.isOpened():
                return writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(path, fourcc, fps, (w, h))

    @staticmethod
    def _to_qimage(frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        # .copy() is required: the numpy buffer backing `rgb` gets reused/
        # overwritten on the next loop iteration, and QImage does not deep
        # copy by default — without this you'd get flickering/corrupt frames.
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()

    # ── counting logic (ported 1:1 from the standalone script) ────────────
    def _get_side(self, cy):
        return 'bottom' if cy > self.line_y else 'top'

    def _net_displacement(self, crossed_at_cy, current_cy, direction):
        return crossed_at_cy - current_cy if direction == 'forward' else current_cy - crossed_at_cy

    def _commit_cross(self, track_id, direction, cx, cy):
        is_ovr = self.track_is_overlap[track_id]
        if self.counted_tracks.get(track_id, False):
            return

        for rcx, rcy, rframe, r_is_ovr in self.recent_commits:
            if self.frame_number - rframe > 15:
                continue
            if r_is_ovr == is_ovr and np.hypot(cx - rcx, cy - rcy) < self.COUNT_DEDUP_RADIUS:
                self.counted_tracks[track_id] = True
                return

        self.recent_commits.append((cx, cy, self.frame_number, is_ovr))
        self.counted_tracks[track_id] = True

        if direction != 'forward':
            self.flash_event = None
            return

        if is_ovr:
            self.count += 1
            self.overlap_count += 1
            self.flash_event = 'overlap'
        else:
            self.count += 1
            self.flash_event = '+'
        self.flash_time = time.time()

    def _find_graveyard_match(self, cx, cy):
        best_id, best_dist = None, self.GRAVEYARD_MATCH_PX
        for old_id, state in self.graveyard.items():
            dist = np.hypot(cx - state['cx'], cy - state['cy'])
            if dist < best_dist:
                best_dist, best_id = dist, old_id
        return best_id

    def _purge_expired_graveyard(self):
        expired = [tid for tid, st in self.graveyard.items()
                   if self.frame_number - st['frame_dropped'] > self.GRAVEYARD_TTL]
        for tid in expired:
            self.graveyard.pop(tid, None)
            self.track_confirmed_side.pop(tid, None)
            self.track_is_overlap.pop(tid, None)
            self.track_coords.pop(tid, None)
            self.pending.pop(tid, None)

    def _run_detection(self, frame):
        results = self.model(frame, imgsz=640, conf=0.25, iou=0.45, verbose=False)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append((
                    [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    float(box.conf[0].cpu().numpy()),
                    int(box.cls[0].cpu().numpy())
                ))
        return detections

    def _process_frame(self, frame):
        """Mutates `frame` in-place with overlays, and updates
        self.count / self.overlap_count as crossings are confirmed."""
        seen_ids = set()
        detections = self._run_detection(frame)
        tracks = self.tracker.update_tracks(detections, frame=frame)

        cv2.line(frame, (0, self.line_y), (self.frame_w, self.line_y), (0, 0, 255), 1)

        candidates = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            l, t, r_c, b = track.to_ltrb()
            rcx, rcy = int((l + r_c) / 2), int((t + b) / 2)

            if track.det_class == self.overlap_class_id:
                self.track_is_overlap[tid] = True

            if tid not in self.track_coords:
                self.track_coords[tid] = (rcx, rcy)
            else:
                pcx, pcy = self.track_coords[tid]
                self.track_coords[tid] = (
                    int(self.SMOOTH_ALPHA * rcx + (1 - self.SMOOTH_ALPHA) * pcx),
                    int(self.SMOOTH_ALPHA * rcy + (1 - self.SMOOTH_ALPHA) * pcy))

            cx, cy = self.track_coords[tid]
            candidates.append({'id': tid, 'cx': cx, 'cy': cy, 'is_ovr': self.track_is_overlap[tid]})

        candidates.sort(key=lambda x: x['is_ovr'], reverse=True)
        processed_green, processed_pink = [], []

        for cand in candidates:
            tid, cx, cy, is_ovr = cand['id'], cand['cx'], cand['cy'], cand['is_ovr']

            if is_ovr:
                if any(np.hypot(cx - px, cy - py) < self.MERGE_RADIUS_PINK for px, py in processed_pink):
                    continue
                processed_pink.append((cx, cy))
            else:
                if any(np.hypot(cx - px, cy - py) < self.MERGE_RADIUS_GREEN for px, py in processed_green):
                    continue
                processed_green.append((cx, cy))

            seen_ids.add(tid)

            if tid not in self.track_confirmed_side:
                old_id = self._find_graveyard_match(cx, cy)
                if old_id:
                    self.track_confirmed_side[tid] = self.graveyard[old_id]['side']
                    self.track_is_overlap[tid] = self.track_is_overlap[old_id]
                    self.counted_tracks[tid] = self.graveyard[old_id].get('counted', False)
                    self.graveyard.pop(old_id, None)
                else:
                    self.track_confirmed_side[tid] = self._get_side(cy)

            confirmed_side, current_side = self.track_confirmed_side[tid], self._get_side(cy)

            if tid not in self.pending:
                if current_side != confirmed_side:
                    self.pending[tid] = {'direction': 'forward' if confirmed_side == 'bottom' else 'backward',
                                          'frames': 1, 'crossed_at_cy': cy}
            else:
                p = self.pending[tid]
                if current_side != confirmed_side:
                    p['frames'] += 1
                    dist = self._net_displacement(p['crossed_at_cy'], cy, p['direction'])
                    req_f = self.FORWARD_CONFIRM_FRAMES if p['direction'] == 'forward' else self.BACKWARD_CONFIRM_FRAMES
                    if p['frames'] >= req_f and dist >= self.FORWARD_MIN_DISPLACEMENT:
                        self._commit_cross(tid, p['direction'], cx, cy)
                        self.track_confirmed_side[tid] = current_side
                        self.pending.pop(tid, None)
                else:
                    self.pending.pop(tid, None)

            color = (255, 0, 255) if is_ovr else (0, 255, 0)
            if not is_ovr and current_side == 'top':
                color = (255, 140, 0)

            cv2.circle(frame, (cx, cy), 6, color, -1)
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1)
            if is_ovr:
                cv2.putText(frame, "OVR", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        for lost_id in set(self.track_confirmed_side.keys()) - seen_ids:
            if lost_id not in self.graveyard and lost_id in self.track_coords:
                cx, cy = self.track_coords[lost_id]
                self.graveyard[lost_id] = {'cx': cx, 'cy': cy, 'side': self.track_confirmed_side[lost_id],
                                            'frame_dropped': self.frame_number,
                                            'counted': self.counted_tracks.get(lost_id, False)}

        self._purge_expired_graveyard()

        if self.flash_event and (time.time() - self.flash_time) < self.FLASH_DURATION:
            cv2.putText(frame, f"EVENT: {self.flash_event}", (10, self.line_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)


# ─── Video Display Widget ──────────────────────────────────────────────────────
class VideoWidget(QWidget):
    def __init__(self):
        super().__init__()
        self._pixmap = None
        self._glow_alpha = 180
        self._glow_dir = -2
        self.setMinimumSize(640, 640)

        # Glow pulse timer
        self._glow_timer = QTimer(self)
        self._glow_timer.timeout.connect(self._pulse_glow)
        self._glow_timer.start(40)

    def _pulse_glow(self):
        self._glow_alpha += self._glow_dir
        if self._glow_alpha <= 80 or self._glow_alpha >= 220:
            self._glow_dir *= -1
        self.update()

    def set_frame(self, img: QImage):
        self._pixmap = QPixmap.fromImage(img)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        W, H = self.width(), self.height()
        radius = 24

        # Outer glow halo — cyan blue
        for i in range(8, 0, -1):
            alpha = int(self._glow_alpha * (i / 8) * 0.22)
            pen = QPen(QColor(0, 212, 255, alpha), i * 2)   # ACCENT_CYAN
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(i, i, W - i * 2, H - i * 2, radius + i, radius + i)

        # Clip inner area
        path = QPainterPath()
        path.addRoundedRect(6, 6, W - 12, H - 12, radius, radius)
        p.setClipPath(path)

        if self._pixmap:
            scaled = self._pixmap.scaled(W - 12, H - 12,
                                         Qt.KeepAspectRatioByExpanding,
                                         Qt.SmoothTransformation)
            x_off = (scaled.width() - (W - 12)) // 2
            y_off = (scaled.height() - (H - 12)) // 2
            p.drawPixmap(6 - x_off, 6 - y_off, scaled)
        else:
            # Placeholder
            p.fillRect(6, 6, W - 12, H - 12, QColor(BG_DARK_PANEL))
            p.setPen(QColor(TEXT_SECONDARY))
            p.setFont(QFont("Inter", 14))
            p.drawText(self.rect(), Qt.AlignCenter, "No Signal")

        p.setClipping(False)

        # Inner border ring — cyan blue
        pen = QPen(QColor(0, 212, 255, self._glow_alpha), 2)   # ACCENT_CYAN
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(6, 6, W - 12, H - 12, radius, radius)

        p.end()


# ─── Live Badge ───────────────────────────────────────────────────────────────
class LiveBadge(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(90, 28)
        self._dot_alpha = 255
        self._dot_dir = -8
        t = QTimer(self)
        t.timeout.connect(self._blink)
        t.start(50)

    def _blink(self):
        self._dot_alpha += self._dot_dir
        if self._dot_alpha <= 60 or self._dot_alpha >= 255:
            self._dot_dir *= -1
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        # Pill background
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(32, 219, 20, 28))   # LIVE_GREEN tint
        p.drawRoundedRect(0, 0, self.width(), self.height(), 14, 14)
        # Dot
        dot_color = QColor(LIVE_GREEN)
        dot_color.setAlpha(self._dot_alpha)
        p.setBrush(dot_color)
        p.drawEllipse(10, 9, 10, 10)
        # Text
        p.setPen(QColor(LIVE_GREEN))
        p.setFont(QFont("Inter", 9, QFont.Bold))
        p.drawText(28, 19, "● LIVE")
        p.end()


# ─── KPI Card ─────────────────────────────────────────────────────────────────
class KPICard(QWidget):
    def __init__(self, title: str, accent: str, parent=None):
        super().__init__(parent)
        self._accent = accent
        self._hover = False
        self._hover_anim = 0.0
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_Hover)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(20, 16, 20, 16)
        self._layout.setSpacing(6)

        # Title
        self._title_label = QLabel(title.upper())
        self._title_label.setFont(QFont("Inter", 9, 57))
        self._title_label.setStyleSheet(f"color: {TEXT_SECONDARY}; letter-spacing: 2px; background: transparent;")
        self._layout.addWidget(self._title_label)

        # Value
        self._value_label = QLabel("0")
        self._value_label.setFont(QFont("Inter", 44, QFont.Bold))
        self._value_label.setStyleSheet(f"color: {accent}; background: transparent;")
        self._layout.addWidget(self._value_label)

        # Hover timer
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._animate_hover)
        self._anim_timer.start(16)

    def set_value(self, v):
        self._value_label.setText(str(v))

    def _animate_hover(self):
        target = 1.0 if self._hover else 0.0
        self._hover_anim += (target - self._hover_anim) * 0.12
        self.update()

    def enterEvent(self, e):
        self._hover = True

    def leaveEvent(self, e):
        self._hover = False

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        a = self._hover_anim
        W, H = self.width(), self.height()

        # Card background with glassmorphism
        bg = QColor(BG_CARD)
        hover_bg = QColor(BG_CARD_HOVER)
        r = int(bg.red()   + (hover_bg.red()   - bg.red())   * a)
        g = int(bg.green() + (hover_bg.green() - bg.green()) * a)
        b = int(bg.blue()  + (hover_bg.blue()  - bg.blue())  * a)

        p.setPen(Qt.NoPen)
        p.setBrush(QColor(r, g, b))
        p.drawRoundedRect(0, 0, W, H, 16, 16)

        # Accent glow border
        accent = QColor(self._accent)
        border_alpha = int(60 + 120 * a)
        p.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), border_alpha), 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(1, 1, W - 2, H - 2, 15, 15)

        # Top accent bar
        bar_alpha = int(80 + 120 * a)
        grad = QLinearGradient(0, 0, W, 0)
        grad.setColorAt(0, QColor(accent.red(), accent.green(), accent.blue(), bar_alpha))
        grad.setColorAt(1, QColor(accent.red(), accent.green(), accent.blue(), 0))
        p.setPen(Qt.NoPen)
        p.setBrush(grad)
        p.drawRoundedRect(0, 0, W, 3, 2, 2)

        # Elevation shadow on hover
        if a > 0.05:
            shadow = QRadialGradient(W / 2, H + 10, W * 0.6)
            shadow.setColorAt(0, QColor(accent.red(), accent.green(), accent.blue(), int(30 * a)))
            shadow.setColorAt(1, QColor(0, 0, 0, 0))
            p.setBrush(shadow)
            p.drawEllipse(-W // 4, H - 10, W + W // 2, 40)

        p.end()
        super().paintEvent(event)


# ─── Session Info Card ────────────────────────────────────────────────────────
class SessionCard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._hover = False
        self._hover_anim = 0.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        title = QLabel("SESSION INFO")
        title.setFont(QFont("Inter", 9, 57))
        title.setStyleSheet(f"color: {TEXT_SECONDARY}; letter-spacing: 2px; background: transparent;")
        layout.addWidget(title)

        self._rows: dict[str, QLabel] = {}
        for key, default in [
            ("Start Time", "--:--:--"),
            ("Elapsed", "00:00:00"),
            ("Status", "Idle"),
        ]:
            row = QHBoxLayout()
            row.setSpacing(8)
            k_lbl = QLabel(key)
            k_lbl.setFont(QFont("Inter", 10))
            k_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")
            v_lbl = QLabel(default)
            v_lbl.setFont(QFont("Inter", 10, QFont.DemiBold))
            v_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; background: transparent;")
            v_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(k_lbl)
            row.addStretch()
            row.addWidget(v_lbl)
            layout.addLayout(row)
            self._rows[key] = v_lbl

        layout.addStretch()

        # Hover
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._animate)
        self._anim_timer.start(16)

    def set_value(self, key: str, val: str):
        if key in self._rows:
            self._rows[key].setText(val)

    def set_status(self, running: bool):
        if running:
            self._rows["Status"].setText("● Running")
            self._rows["Status"].setStyleSheet(f"color: {ACCENT_GREEN}; background: transparent; font-weight: 600;")
        else:
            self._rows["Status"].setText("● Idle")
            self._rows["Status"].setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent; font-weight: 600;")

    def _animate(self):
        target = 1.0 if self._hover else 0.0
        self._hover_anim += (target - self._hover_anim) * 0.12
        self.update()

    def enterEvent(self, e): self._hover = True
    def leaveEvent(self, e): self._hover = False

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        a = self._hover_anim
        W, H = self.width(), self.height()

        bg = QColor(BG_CARD)
        hov = QColor(BG_CARD_HOVER)
        r = int(bg.red()   + (hov.red()   - bg.red())   * a)
        g = int(bg.green() + (hov.green() - bg.green()) * a)
        b = int(bg.blue()  + (hov.blue()  - bg.blue())  * a)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(r, g, b))
        p.drawRoundedRect(0, 0, W, H, 16, 16)

        border_alpha = int(40 + 80 * a)
        p.setPen(QPen(QColor(59, 130, 246, border_alpha), 1))   # CARD_BLUE
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(1, 1, W - 2, H - 2, 15, 15)

        # Top bar — card highlight blue
        grad = QLinearGradient(0, 0, W, 0)
        grad.setColorAt(0, QColor(59, 130, 246, int(70 + 90 * a)))   # CARD_BLUE
        grad.setColorAt(1, QColor(59, 130, 246, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(grad)
        p.drawRoundedRect(0, 0, W, 3, 2, 2)
        p.end()
        super().paintEvent(event)


# ─── Glow Button ──────────────────────────────────────────────────────────────
class GlowButton(QPushButton):
    def __init__(self, text: str, accent: str, parent=None):
        super().__init__(text, parent)
        self._accent = accent
        self._hover_anim = 0.0
        self._pulse_anim = 0.0
        self._is_active = False
        self.setFixedHeight(52)
        self.setCursor(Qt.PointingHandCursor)
        self.setFont(QFont("Inter", 12, QFont.DemiBold))

        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._tick)
        self._anim_timer.start(16)
        self._pulse_t = 0.0

    def set_active(self, v: bool):
        self._is_active = v

    def _tick(self):
        target_hover = 1.0 if self.underMouse() else 0.0
        self._hover_anim += (target_hover - self._hover_anim) * 0.15

        if self._is_active:
            self._pulse_t += 0.05
            self._pulse_anim = 0.5 + 0.5 * math.sin(self._pulse_t)
        else:
            self._pulse_anim = 0.0
            self._pulse_t = 0.0

        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        a = self._hover_anim
        pulse = self._pulse_anim
        accent = QColor(self._accent)

        # Outer pulse ring
        if pulse > 0:
            ring_alpha = int(60 * pulse)
            spread = int(8 * pulse)
            p.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), ring_alpha), 2))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(-spread, -spread, W + spread * 2, H + spread * 2, 14 + spread, 14 + spread)

        # Glow halo
        for i in range(6, 0, -1):
            alpha = int((a * 0.6 + pulse * 0.4) * 35 * (i / 6))
            p.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), alpha), i * 2))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(i, i, W - i * 2, H - i * 2, 12, 12)

        # Button fill
        fill_alpha = int(30 + 60 * a + 30 * pulse)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(accent.red(), accent.green(), accent.blue(), fill_alpha))
        p.drawRoundedRect(0, 0, W, H, 12, 12)

        # Border
        border_alpha = int(140 + 115 * a)
        p.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), border_alpha), 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(1, 1, W - 2, H - 2, 11, 11)

        # Label
        text_alpha = int(180 + 75 * a)
        p.setPen(QColor(accent.red(), accent.green(), accent.blue(), text_alpha))
        p.setFont(QFont("Inter", 12, QFont.DemiBold))
        p.drawText(self.rect(), Qt.AlignCenter, self.text())
        p.end()


# ─── Main Window ──────────────────────────────────────────────────────────────
class FactoryDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Factory Vision · Control Dashboard")
        self.setMinimumSize(1280, 780)
        self.resize(1920, 1080)

        self._counting = False
        self._start_time: datetime | None = None
        self._backend_thread: BagCounterThread | None = None

        self._setup_ui()
        self._setup_timers()

    # ── UI Construction ────────────────────────────────────────────────────────
    def _setup_ui(self):
        # Root
        root = QWidget()
        root.setStyleSheet(f"background: {BG_DEEP};")
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────────
        header = self._build_header()
        outer.addWidget(header)

        # ── Content row ───────────────────────────────────────────────────────
        content = QWidget()
        content.setStyleSheet("background: transparent;")
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(28, 16, 28, 0)
        content_layout.setSpacing(24)

        # Left — video
        left = self._build_left_panel()
        content_layout.addWidget(left, 70)

        # Right — KPI cards
        right = self._build_right_panel()
        content_layout.addWidget(right, 30)

        outer.addWidget(content, 1)

        # ── Action bar ────────────────────────────────────────────────────────
        action_bar = self._build_action_bar()
        outer.addWidget(action_bar)

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(64)
        header.setStyleSheet(f"""
            background: {BG_CARD};
            border-bottom: 1px solid {BORDER_DIM};
        """)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(28, 0, 28, 0)

        # Logo + title
        logo_dot = QLabel("◈")
        logo_dot.setFont(QFont("Inter", 18))
        logo_dot.setStyleSheet(f"color: {ACCENT_BLUE}; background: transparent;")

        title = QLabel("FactoryVision")
        title.setFont(QFont("Inter", 16, QFont.Bold))
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; background: transparent; letter-spacing: 1px;")

        subtitle = QLabel("Control Dashboard")
        subtitle.setFont(QFont("Inter", 10))
        subtitle.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")

        layout.addWidget(logo_dot)
        layout.addSpacing(8)
        layout.addWidget(title)
        layout.addSpacing(12)
        layout.addWidget(subtitle)
        layout.addStretch()

        # System time
        self._clock_label = QLabel()
        self._clock_label.setFont(QFont("Inter", 13, 57))
        self._clock_label.setStyleSheet(f"color: {TEXT_PRIMARY}; background: transparent; font-variant: tabular-nums;")
        layout.addWidget(self._clock_label)

        # Status pill
        self._status_pill = QLabel("  SYSTEM READY  ")
        self._status_pill.setFont(QFont("Inter", 9, QFont.Bold))
        self._status_pill.setStyleSheet(f"""
            color: {ACCENT_GREEN};
            background: rgba(0, 255, 136, 0.12);
            border: 1px solid rgba(0, 255, 136, 0.35);
            border-radius: 12px;
            padding: 4px 12px;
            letter-spacing: 1.5px;
        """)
        layout.addSpacing(20)
        layout.addWidget(self._status_pill)

        return header

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Row: label + live badge
        top_row = QHBoxLayout()
        cam_label = QLabel("CAM-01  ·  FACTORY FLOOR")
        cam_label.setFont(QFont("Inter", 11, 57))
        cam_label.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent; letter-spacing: 1px;")
        top_row.addWidget(cam_label)
        top_row.addStretch()
        self._live_badge = LiveBadge()
        top_row.addWidget(self._live_badge)
        layout.addLayout(top_row)

        # Video
        self._video_widget = VideoWidget()
        self._video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._video_widget, 1)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        section_label = QLabel("KPI OVERVIEW")
        section_label.setFont(QFont("Inter", 9, 57))
        section_label.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent; letter-spacing: 2px;")
        layout.addWidget(section_label)

        # KPI 1: Total Count
        self._card_total = KPICard("Total Count", ACCENT_BLUE)
        self._card_total.set_value(0)
        layout.addWidget(self._card_total, 1)

        # KPI 2: Overlap Count
        self._card_overlap = KPICard("Overlapped", ACCENT_AMBER)
        self._card_overlap.set_value(0)
        layout.addWidget(self._card_overlap, 1)

        # Session Info
        self._session_card = SessionCard()
        layout.addWidget(self._session_card, 1)

        return panel

    def _build_action_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(88)
        bar.setStyleSheet(f"""
            background: {BG_CARD};
            border-top: 1px solid {BORDER_DIM};
        """)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(28, 16, 28, 16)
        layout.setSpacing(16)

        # Left hint / status message from the backend
        self._hint_label = QLabel("Select an action to begin object counting session")
        self._hint_label.setFont(QFont("Inter", 10))
        self._hint_label.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")
        layout.addWidget(self._hint_label)
        layout.addStretch()

        # Start
        self._btn_start = GlowButton("▶  Start Counting", ACCENT_GREEN)
        self._btn_start.setMinimumWidth(200)
        self._btn_start.clicked.connect(self._start_counting)
        layout.addWidget(self._btn_start)

        # Stop
        self._btn_stop = GlowButton("■  Stop Counting", ACCENT_RED)
        self._btn_stop.setMinimumWidth(200)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_counting)
        layout.addWidget(self._btn_stop)

        return bar

    # ── Timers ─────────────────────────────────────────────────────────────────
    def _setup_timers(self):
        # Clock
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(500)
        self._update_clock()

        # Session elapsed
        self._session_timer = QTimer(self)
        self._session_timer.timeout.connect(self._update_session)

    # ── Slots: clock / session ───────────────────────────────────────────────
    def _update_clock(self):
        now = datetime.now().strftime("%H:%M:%S")
        self._clock_label.setText(now)

    def _update_session(self):
        if self._start_time:
            elapsed = datetime.now() - self._start_time
            h, rem = divmod(int(elapsed.total_seconds()), 3600)
            m, s = divmod(rem, 60)
            self._session_card.set_value("Elapsed", f"{h:02d}:{m:02d}:{s:02d}")

    # ── Slots: backend thread signals ────────────────────────────────────────
    def _on_frame(self, img: QImage):
        self._video_widget.set_frame(img)

    def _on_stats(self, total: int, overlap: int):
        self._card_total.set_value(total)
        self._card_overlap.set_value(overlap)

    def _on_status_message(self, msg: str):
        self._hint_label.setText(msg)
        if msg.startswith("ERROR"):
            self._hint_label.setStyleSheet(f"color: {ACCENT_RED}; background: transparent;")
            self._set_status_pill("  ERROR  ", ACCENT_RED)
            self._stop_counting()
        elif msg == "Running":
            self._hint_label.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")
            self._set_status_pill("  COUNTING ACTIVE  ", ACCENT_AMBER)

    def _on_source_ended(self):
        # Video file ran out (won't normally fire for a live camera feed).
        if self._counting:
            self._hint_label.setText("Source ended.")
            self._stop_counting()

    # ── Start / Stop ──────────────────────────────────────────────────────────
    def _set_status_pill(self, text: str, color_hex: str):
        c = QColor(color_hex)
        self._status_pill.setText(text)
        self._status_pill.setStyleSheet(f"""
            color: {color_hex};
            background: rgba({c.red()}, {c.green()}, {c.blue()}, 0.12);
            border: 1px solid rgba({c.red()}, {c.green()}, {c.blue()}, 0.35);
            border-radius: 12px;
            padding: 4px 12px;
            letter-spacing: 1.5px;
        """)

    def _start_counting(self):
        if self._counting:
            return
        self._counting = True
        self._start_time = datetime.now()

        self._card_total.set_value(0)
        self._card_overlap.set_value(0)
        self._session_card.set_value("Start Time", self._start_time.strftime("%H:%M:%S"))
        self._session_card.set_value("Elapsed", "00:00:00")
        self._session_card.set_status(True)

        self._btn_start.setEnabled(False)
        self._btn_start.set_active(False)
        self._btn_stop.setEnabled(True)
        self._btn_stop.set_active(True)

        self._set_status_pill("  LOADING MODEL…  ", ACCENT_AMBER)
        self._hint_label.setText("Loading model and opening video source…")

        self._session_timer.start(1000)

        # Fresh thread instance each run — a finished QThread cannot be restarted.
        self._backend_thread = BagCounterThread(
            model_path=MODEL_PATH,
            video_source=VIDEO_SOURCE,
            output_path=OUTPUT_PATH if SAVE_OUTPUT_VIDEO else None,
            frame_w=FRAME_W, frame_h=FRAME_H,
        )
        self._backend_thread.frame_ready.connect(self._on_frame)
        self._backend_thread.stats_updated.connect(self._on_stats)
        self._backend_thread.status_message.connect(self._on_status_message)
        self._backend_thread.source_ended.connect(self._on_source_ended)
        self._backend_thread.start()

    def _stop_counting(self):
        if not self._counting:
            return
        self._counting = False
        self._session_timer.stop()

        if self._backend_thread is not None:
            self._backend_thread.stop()
            self._backend_thread.wait(5000)   # block briefly until the loop actually exits
            self._backend_thread = None

        self._btn_start.setEnabled(True)
        self._btn_start.set_active(False)
        self._btn_stop.setEnabled(False)
        self._btn_stop.set_active(False)

        self._session_card.set_status(False)
        self._hint_label.setText("Select an action to begin object counting session")
        self._hint_label.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")
        self._set_status_pill("  SYSTEM READY  ", ACCENT_GREEN)

    def closeEvent(self, event):
        if self._backend_thread is not None:
            self._backend_thread.stop()
            self._backend_thread.wait(5000)
        super().closeEvent(event)


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Load Inter font if available, else fall back gracefully
    QFontDatabase.addApplicationFont("Inter.ttf")

    # Global palette
    from PyQt5.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.Window,        QColor(BG_DEEP))
    palette.setColor(QPalette.WindowText,    QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.Base,          QColor(BG_CARD))
    palette.setColor(QPalette.AlternateBase, QColor(BG_CARD))
    palette.setColor(QPalette.Text,          QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.Button,        QColor(BG_CARD))
    palette.setColor(QPalette.ButtonText,    QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.Highlight,     QColor(ACCENT_BLUE))
    app.setPalette(palette)

    win = FactoryDashboard()
    win.show()
    sys.exit(app.exec_())