"""
Factory Vision Control Dashboard — Full Integrated Product
Merges PyQt5 GUI with YOLO + DeepSORT bag-counting backend.

Requirements:
    pip install PyQt5 opencv-python ultralytics deep-sort-realtime
"""

import sys
import cv2
import time
import math
import numpy as np
from datetime import datetime
from collections import defaultdict, deque

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import (
    QColor, QPainter, QPen, QFont, QFontDatabase,
    QLinearGradient, QRadialGradient, QPainterPath, QPixmap, QImage, QPalette
)


# ══════════════════════════════════════════════════════════════════════════════
#  COLOR PALETTE
# ══════════════════════════════════════════════════════════════════════════════
BG_MAIN        = "#040917"
BG_SECONDARY   = "#0A1022"
BG_CARD        = "#162B4F"
BG_CARD_HOVER  = "#1D3660"
BG_DARK_PANEL  = "#101A33"
BG_SIDEBAR     = "#07101F"

ACCENT_BLUE    = "#00A8FF"
ACCENT_CYAN    = "#00D4FF"
ELECTRIC_BLUE  = "#1E90FF"
BORDER_BLUE    = "#162BFF"
CARD_BLUE      = "#3B82F6"

LIVE_GREEN     = "#20DB14"
SUCCESS_GREEN  = "#209D24"
BTN_START      = "#00C853"
NEON_GREEN     = "#00FF88"

ORANGE_ACCENT  = "#9B6611"
AMBER_GLOW     = "#FFB300"
GOLD_HIGH      = "#FFC107"

BTN_STOP_CLR   = "#FF4D4F"
DARK_RED       = "#D32F2F"
NEON_RED       = "#FF1744"

TEXT_PRIMARY   = "#E8ECF0"
TEXT_SECONDARY = "#A3B4CB"
TEXT_MUTED     = "#9CA2A8"
TEXT_DISABLED  = "#76889A"

BOX_BROWN      = "#674630"
BOX_DARK_BROWN = "#583226"

BORDER_DIM     = "#1A2B4A"
BORDER_GLOW    = "#1E3A6F"


# ══════════════════════════════════════════════════════════════════════════════
#  BACKEND  —  YOLO + DeepSORT counting thread
# ══════════════════════════════════════════════════════════════════════════════

class CountingThread(QThread):
    """
    Runs YOLO detection + DeepSORT tracking + line-crossing counting in a
    background thread.  Emits:
      frame_ready(QImage)   — annotated frame for the video widget
      stats_updated(int, int) — (total_count, overlap_count)
    """
    frame_ready    = pyqtSignal(QImage)
    stats_updated  = pyqtSignal(int, int)

    # ── Tuning knobs (mirrors backend script) ─────────────────────────────────
    FRAME_W                   = 640
    FRAME_H                   = 480
    LINE_Y_OFFSET             = 40          # added to FRAME_H//2
    LINE_THICKNESS            = 2
    SMOOTH_WINDOW             = 8
    FORWARD_CONFIRM_FRAMES    = 1
    BACKWARD_CONFIRM_FRAMES   = 8
    FORWARD_MIN_DISPLACEMENT  = 5
    BACKWARD_MIN_DISPLACEMENT = 40
    DANGER_MARGIN             = 50
    GRAVEYARD_TTL             = 35
    GRAVEYARD_MATCH_PX        = 70
    DOUBLE_COUNT_RADIUS       = 90
    MERGE_RADIUS              = 40
    CONF_THRESH               = 0.70
    IOU_THRESH                = 0.90
    FLASH_DURATION            = 0.7        # seconds

    def __init__(self, model_path: str, video_path: str):
        super().__init__()
        self._model_path = r"C:\Users\siddh\Desktop\adhesive_bag\runs\detect\runs\adhesive_bag\bag_overlapped_head_only\weights\best.pt"
        self._video_path = r"C:\Users\siddh\Desktop\adhesive_bag\merged_overlapped.mp4"
        self._running    = False

    # ── Public API ────────────────────────────────────────────────────────────
    def stop(self):
        self._running = False
        self.wait()

    # ── Thread entry ──────────────────────────────────────────────────────────
    def run(self):
        self._running = True

        # Load model
        model = YOLO(self._model_path)
        CLASS_NAMES     = model.names
        BAG_CLASS_ID    = next((k for k, v in CLASS_NAMES.items() if v == 'bag'),        0)
        OVERLAP_CLASS_ID= next((k for k, v in CLASS_NAMES.items() if v == 'overlapped'), 1)

        # Tracker
        tracker = DeepSort(
            max_age             = 10,
            n_init              = 2,
            max_cosine_distance = 0.70,
            nn_budget           = 100,
            max_iou_distance    = 0.85,
            embedder            = "mobilenet",
            half                = True,
            bgr                 = True,
        )

        # Video source
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            return

        W  = self.FRAME_W
        H  = self.FRAME_H
        LY = (H // 2) + self.LINE_Y_OFFSET     # counting line Y

        # ── Counting state ────────────────────────────────────────────────────
        count         = 0
        overlap_count = 0
        frame_number  = 0
        flash_event   = None
        flash_time    = 0.0

        track_cx_hist        = defaultdict(lambda: deque(maxlen=self.SMOOTH_WINDOW))
        track_cy_hist        = defaultdict(lambda: deque(maxlen=4))
        track_confirmed_side = {}
        track_class          = {}
        track_count_label    = {}
        pending              = {}
        graveyard            = {}
        recent_commits       = deque(maxlen=10)

        # ── Inner helpers (closures over local state) ─────────────────────────
        def smoothed_cx(tid):
            h = list(track_cx_hist[tid])
            return int(np.mean(h)) if h else None

        def get_side(cy):
            return 'bottom' if cy > LY else 'top'

        def in_danger_zone(cy):
            return abs(cy - LY) <= self.DANGER_MARGIN

        def net_displacement(crossed_at_cy, cur_cy, direction):
            return (crossed_at_cy - cur_cy) if direction == 'forward' \
                   else (cur_cy - crossed_at_cy)

        def find_graveyard_match(cx, cy):
            best_id, best_dist = None, self.GRAVEYARD_MATCH_PX
            for old_id, st in list(graveyard.items()):
                if frame_number - st['frame_dropped'] > self.GRAVEYARD_TTL:
                    graveyard.pop(old_id, None)
                    continue
                d = np.hypot(cx - st['cx'], cy - st['cy'])
                if d < best_dist:
                    best_dist = d
                    best_id   = old_id
            return best_id

        def commit_cross(tid, direction):
            nonlocal count, overlap_count, flash_event, flash_time

            hcx = list(track_cx_hist[tid])
            hcy = list(track_cy_hist[tid])
            cx  = hcx[-1] if hcx else 0
            cy  = hcy[-1] if hcy else 0

            for rcx, rcy, rframe in recent_commits:
                if frame_number - rframe > 8:
                    continue
                if np.hypot(cx - rcx, cy - rcy) < self.DOUBLE_COUNT_RADIUS:
                    return   # spatial dedup

            recent_commits.append((cx, cy, frame_number))
            is_overlapped = track_class.get(tid) == OVERLAP_CLASS_ID

            if direction == 'forward':
                if is_overlapped:
                    count         += 2
                    overlap_count += 1
                    flash_event    = 'overlap'
                else:
                    count      += 1
                    flash_event = '+'
            else:
                if is_overlapped:
                    count         = max(0, count - 2)
                    overlap_count = max(0, overlap_count - 1)
                    flash_event   = 'overlap_back'
                else:
                    count      = max(0, count - 1)
                    flash_event = '-'

            track_count_label[tid] = count
            flash_time = time.time()
            self.stats_updated.emit(count, overlap_count)

        # ── Main loop ─────────────────────────────────────────────────────────
        while self._running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame        = cv2.resize(frame, (W, H))
            frame_number += 1
            seen_ids     = set()
            processed_positions = []

            # YOLO inference
            results    = model(frame, imgsz=640,
                               conf=self.CONF_THRESH,
                               iou=self.IOU_THRESH,
                               verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls = int(box.cls[0])
                    detections.append((
                        [x1, y1, x2 - x1, y2 - y1],
                        float(box.conf[0]),
                        cls
                    ))

            # DeepSORT
            tracks = tracker.update_tracks(detections, frame=frame)

            # ── Draw counting line  (neon red) ────────────────────────────────
            cv2.line(frame, (0, LY), (W, LY), (255, 23, 68), self.LINE_THICKNESS)
            # Glow duplicate lines
            cv2.line(frame, (0, LY), (W, LY), (180, 0, 40, ), 1)

            # ── Per-track processing ──────────────────────────────────────────
            for track in tracks:
                if not track.is_confirmed():
                    continue

                tid           = track.track_id
                l, t, rc, b  = track.to_ltrb()
                x1, y1, x2, y2 = int(l), int(t), int(rc), int(b)

                raw_cx = (x1 + x2) // 2
                cy     = (y1 + y2) // 2

                track_cx_hist[tid].append(raw_cx)
                track_cy_hist[tid].append(cy)
                cx = smoothed_cx(tid)

                if track.det_class is not None:
                    track_class[tid] = track.det_class

                # Duplicate suppression
                is_dup = any(
                    np.hypot(cx - px, cy - py) < self.MERGE_RADIUS
                    for px, py in processed_positions
                )
                if is_dup:
                    cv2.circle(frame, (cx, cy), 4, (80, 80, 80), -1)
                    continue
                processed_positions.append((cx, cy))

                current_side = get_side(cy)
                seen_ids.add(tid)

                # Graveyard inheritance
                if tid not in track_confirmed_side:
                    old_id = find_graveyard_match(cx, cy)
                    if old_id is not None:
                        track_confirmed_side[tid] = graveyard[old_id]['side']
                        if old_id in track_class:
                            track_class[tid] = track_class[old_id]
                        if old_id in track_count_label:
                            track_count_label[tid] = track_count_label[old_id]
                        if old_id in pending:
                            pending[tid] = pending.pop(old_id)
                        graveyard.pop(old_id, None)
                    else:
                        track_confirmed_side[tid] = current_side

                confirmed_side = track_confirmed_side[tid]

                # Crossing state machine
                if tid not in pending:
                    if current_side != confirmed_side:
                        direction = ('forward' if confirmed_side == 'bottom'
                                     else 'backward')
                        pending[tid] = {
                            'direction'         : direction,
                            'frames_on_new_side': 1,
                            'crossed_at_cy'     : cy,
                        }
                else:
                    p = pending[tid]
                    if current_side != confirmed_side:
                        p['frames_on_new_side'] += 1
                        direction      = p['direction']
                        confirm_needed = (self.FORWARD_CONFIRM_FRAMES
                                          if direction == 'forward'
                                          else self.BACKWARD_CONFIRM_FRAMES)
                        min_disp       = (self.FORWARD_MIN_DISPLACEMENT
                                          if direction == 'forward'
                                          else self.BACKWARD_MIN_DISPLACEMENT)
                        displacement   = net_displacement(
                            p['crossed_at_cy'], cy, direction)

                        if (p['frames_on_new_side'] >= confirm_needed
                                and displacement >= min_disp):
                            commit_cross(tid, direction)
                            track_confirmed_side[tid] = current_side
                            pending.pop(tid, None)
                    else:
                        pending.pop(tid, None)

                # ── Visual annotation ─────────────────────────────────────────
                in_zone      = in_danger_zone(cy)
                has_pend     = tid in pending
                is_overlapped= track_class.get(tid) == OVERLAP_CLASS_ID

                # Bounding box color
                if is_overlapped:
                    box_color = (255, 179, 0)      # AMBER_GLOW in BGR
                    dot_col   = (255, 0, 255)       # magenta
                elif has_pend and in_zone:
                    box_color = (0, 212, 255)       # ACCENT_CYAN
                    dot_col   = (0, 255, 255)
                elif current_side == 'bottom':
                    box_color = (20, 219, 20)       # LIVE_GREEN
                    dot_col   = (20, 219, 20)
                else:
                    box_color = (0, 168, 255)       # ACCENT_BLUE
                    dot_col   = (255, 140, 0)

                # Draw bounding box with corner ticks
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)
                tick = 10
                for (cx_, cy_, dx, dy) in [
                    (x1, y1, 1, 1), (x2, y1, -1, 1),
                    (x1, y2, 1, -1), (x2, y2, -1, -1)
                ]:
                    cv2.line(frame, (cx_, cy_), (cx_ + dx*tick, cy_), box_color, 2)
                    cv2.line(frame, (cx_, cy_), (cx_, cy_ + dy*tick), box_color, 2)

                # ID chip
                label_txt = f"OVR #{tid}" if is_overlapped else f"#{tid}"
                lbl_size, _ = cv2.getTextSize(
                    label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                lbl_x, lbl_y = x1, max(y1 - 4, 18)
                cv2.rectangle(frame,
                              (lbl_x, lbl_y - lbl_size[1] - 4),
                              (lbl_x + lbl_size[0] + 6, lbl_y + 2),
                              box_color, -1)
                cv2.putText(frame, label_txt,
                            (lbl_x + 3, lbl_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

                # Center dot
                cv2.circle(frame, (cx, cy), 6, dot_col, -1)
                cv2.circle(frame, (cx, cy), 6, (0, 0, 0), 1)

                # Running count label above dot
                if tid in track_count_label:
                    ct = str(track_count_label[tid])
                    ts, _ = cv2.getTextSize(ct, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
                    tx = max(2, min(W - ts[0] - 2, cx - ts[0] // 2))
                    ty = max(18, cy - 14)
                    cv2.putText(frame, ct, (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                                (255, 255, 255), 2)
                    cv2.putText(frame, ct, (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 0, 0), 1)

            # ── Move dropped tracks to graveyard ──────────────────────────────
            for lost_id in set(track_confirmed_side.keys()) - seen_ids:
                if lost_id not in graveyard:
                    hcx = list(track_cx_hist[lost_id])
                    hcy = list(track_cy_hist[lost_id])
                    graveyard[lost_id] = {
                        'cx'           : hcx[-1] if hcx else W // 2,
                        'cy'           : hcy[-1] if hcy else H,
                        'side'         : track_confirmed_side[lost_id],
                        'frame_dropped': frame_number,
                    }

            # ── Graveyard: auto-commit lost forward crossings ─────────────────
            for lost_id, p in list(pending.items()):
                if lost_id not in seen_ids:
                    st = graveyard.get(lost_id, {})
                    frames_since = frame_number - st.get(
                        'frame_dropped', frame_number)
                    if (p['direction'] == 'forward'
                            and frames_since >= 10
                            and p['frames_on_new_side'] >= self.FORWARD_CONFIRM_FRAMES):
                        commit_cross(lost_id, 'forward')
                        track_confirmed_side[lost_id] = 'top'
                        pending.pop(lost_id, None)
                    elif frames_since >= self.GRAVEYARD_TTL:
                        pending.pop(lost_id, None)

            # ── Flash overlay ─────────────────────────────────────────────────
            if flash_event and (time.time() - flash_time) < self.FLASH_DURATION:
                fmap = {
                    '+'           : ((20, 219, 20),   f"+1  [{count}]"),
                    '-'           : ((68, 23, 255),    f"-1  [{count}]"),
                    'overlap'     : ((255, 0, 255),    f"+2 OVR  [{count}]"),
                    'overlap_back': ((200, 0, 200),    f"-2 OVR  [{count}]"),
                }
                fc, txt = fmap.get(flash_event, ((255,255,255), ""))
                cv2.putText(frame, txt, (10, LY - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, fc, 2)
            else:
                flash_event = None

            # ── Line label ────────────────────────────────────────────────────
            cv2.putText(frame, "COUNTING LINE",
                        (W - 145, LY - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 23, 68), 1)

            # ── Convert BGR→RGB and emit ──────────────────────────────────────
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg  = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.frame_ready.emit(qimg.copy())

        cap.release()
        self._running = False


# ══════════════════════════════════════════════════════════════════════════════
#  GUI WIDGETS
# ══════════════════════════════════════════════════════════════════════════════

# ── Video Display ─────────────────────────────────────────────────────────────
class VideoWidget(QWidget):
    def __init__(self):
        super().__init__()
        self._pixmap     = None
        self._glow_alpha = 180
        self._glow_dir   = -2
        self.setMinimumSize(640, 480)
        t = QTimer(self)
        t.timeout.connect(self._pulse)
        t.start(40)

    def _pulse(self):
        self._glow_alpha = max(75, min(215, self._glow_alpha + self._glow_dir))
        if self._glow_alpha in (75, 215):
            self._glow_dir *= -1
        self.update()

    def set_frame(self, img: QImage):
        self._pixmap = QPixmap.fromImage(img)
        self.update()

    def clear(self):
        self._pixmap = None
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H, R = self.width(), self.height(), 20

        # Neon cyan glow halo
        for i in range(10, 0, -1):
            alpha = int(self._glow_alpha * (i / 10) * 0.18)
            p.setPen(QPen(QColor(0, 212, 255, alpha), i * 2))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(i, i, W - i*2, H - i*2, R + i, R + i)

        clip = QPainterPath()
        clip.addRoundedRect(6, 6, W - 12, H - 12, R, R)
        p.setClipPath(clip)

        if self._pixmap:
            sc = self._pixmap.scaled(W - 12, H - 12,
                                     Qt.KeepAspectRatioByExpanding,
                                     Qt.SmoothTransformation)
            xo = (sc.width()  - (W - 12)) // 2
            yo = (sc.height() - (H - 12)) // 2
            p.drawPixmap(6 - xo, 6 - yo, sc)
        else:
            p.fillRect(6, 6, W - 12, H - 12, QColor(BG_DARK_PANEL))
            p.setPen(QColor(TEXT_DISABLED))
            p.setFont(QFont("Inter", 14))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "No Signal\n\nSelect model & video, then press  ▶  Start Counting")

        p.setClipping(False)
        p.setPen(QPen(QColor(0, 212, 255, self._glow_alpha), 2))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(6, 6, W - 12, H - 12, R, R)
        p.end()


# ── Live Badge ────────────────────────────────────────────────────────────────
class LiveBadge(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(90, 28)
        self._alpha = 255
        self._dir   = -8
        t = QTimer(self)
        t.timeout.connect(self._blink)
        t.start(50)

    def _blink(self):
        self._alpha = max(55, min(255, self._alpha + self._dir))
        if self._alpha in (55, 255):
            self._dir *= -1
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(32, 219, 20, 28))
        p.drawRoundedRect(0, 0, self.width(), self.height(), 14, 14)
        p.setPen(QPen(QColor(32, 219, 20, 90), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(0, 0, self.width(), self.height(), 14, 14)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(32, 219, 20, self._alpha))
        p.drawEllipse(10, 9, 10, 10)
        p.setPen(QColor(LIVE_GREEN))
        p.setFont(QFont("Inter", 9, QFont.Bold))
        p.drawText(28, 19, "● LIVE")
        p.end()


# ── KPI Card ──────────────────────────────────────────────────────────────────
class KPICard(QWidget):
    def __init__(self, title: str, accent: str):
        super().__init__()
        self._accent = accent
        self._hover  = False
        self._anim   = 0.0
        self.setMouseTracking(True)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 18)
        lay.setSpacing(6)

        self._title = QLabel(title.upper())
        self._title.setFont(QFont("Inter", 9, 57))
        self._title.setStyleSheet(
            f"color: {TEXT_SECONDARY}; letter-spacing: 2px; background: transparent;")
        lay.addWidget(self._title)

        self._value = QLabel("0")
        self._value.setFont(QFont("Inter", 46, QFont.Bold))
        self._value.setStyleSheet(f"color: {accent}; background: transparent;")
        lay.addWidget(self._value)

        self._sub = QLabel("— items this session")
        self._sub.setFont(QFont("Inter", 9))
        self._sub.setStyleSheet(
            f"color: {TEXT_DISABLED}; background: transparent;")
        lay.addWidget(self._sub)

        t = QTimer(self)
        t.timeout.connect(self._tick)
        t.start(16)

    def set_value(self, v: int):
        self._value.setText(str(v))
        self._sub.setText(f"{v} items this session")

    def _tick(self):
        tgt = 1.0 if self._hover else 0.0
        self._anim += (tgt - self._anim) * 0.12
        self.update()

    def enterEvent(self, e): self._hover = True
    def leaveEvent(self, e): self._hover = False

    def paintEvent(self, event):
        p   = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        a   = self._anim
        W, H = self.width(), self.height()
        ac  = QColor(self._accent)

        base = QColor(BG_CARD)
        hov  = QColor(BG_CARD_HOVER)
        r = int(base.red()   + (hov.red()   - base.red())   * a)
        g = int(base.green() + (hov.green() - base.green()) * a)
        b = int(base.blue()  + (hov.blue()  - base.blue())  * a)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(r, g, b))
        p.drawRoundedRect(0, 0, W, H, 16, 16)

        p.setPen(QPen(QColor(ac.red(), ac.green(), ac.blue(),
                             int(55 + 120 * a)), 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(1, 1, W - 2, H - 2, 15, 15)

        grad = QLinearGradient(0, 0, W, 0)
        grad.setColorAt(0, QColor(ac.red(), ac.green(), ac.blue(),
                                  int(80 + 120 * a)))
        grad.setColorAt(1, QColor(ac.red(), ac.green(), ac.blue(), 0))
        p.setPen(Qt.NoPen); p.setBrush(grad)
        p.drawRoundedRect(0, 0, W, 3, 2, 2)

        if a > 0.05:
            sh = QRadialGradient(W / 2, H + 8, W * 0.55)
            sh.setColorAt(0, QColor(ac.red(), ac.green(), ac.blue(),
                                    int(28 * a)))
            sh.setColorAt(1, QColor(0, 0, 0, 0))
            p.setBrush(sh)
            p.drawEllipse(-W // 4, H - 8, W + W // 2, 36)

        p.end()
        super().paintEvent(event)


# ── Session Info Card ─────────────────────────────────────────────────────────
class SessionCard(QWidget):
    def __init__(self):
        super().__init__()
        self._hover = False
        self._anim  = 0.0

        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 18)
        lay.setSpacing(12)

        hdr = QLabel("SESSION INFO")
        hdr.setFont(QFont("Inter", 9, 57))
        hdr.setStyleSheet(
            f"color: {TEXT_SECONDARY}; letter-spacing: 2px; background: transparent;")
        lay.addWidget(hdr)

        self._rows: dict[str, QLabel] = {}
        for key, default in [
            ("Start Time", "--:--:--"),
            ("Elapsed",    "00:00:00"),
            ("Status",     "Idle"),
        ]:
            row = QHBoxLayout()
            kl  = QLabel(key)
            kl.setFont(QFont("Inter", 10))
            kl.setStyleSheet(f"color: {TEXT_MUTED}; background: transparent;")
            vl  = QLabel(default)
            vl.setFont(QFont("Inter", 10, QFont.DemiBold))
            vl.setStyleSheet(f"color: {TEXT_PRIMARY}; background: transparent;")
            vl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(kl); row.addStretch(); row.addWidget(vl)
            lay.addLayout(row)
            self._rows[key] = vl

        div = QWidget()
        div.setFixedHeight(1)
        div.setStyleSheet(f"background: {BORDER_DIM};")
        lay.addWidget(div)
        lay.addStretch()

        t = QTimer(self)
        t.timeout.connect(self._tick)
        t.start(16)

    def set_value(self, key: str, val: str):
        if key in self._rows:
            self._rows[key].setText(val)

    def set_status(self, running: bool):
        lbl = self._rows["Status"]
        if running:
            lbl.setText("● Running")
            lbl.setStyleSheet(
                f"color: {LIVE_GREEN}; background: transparent; font-weight: 600;")
        else:
            lbl.setText("● Idle")
            lbl.setStyleSheet(
                f"color: {TEXT_DISABLED}; background: transparent; font-weight: 600;")

    def _tick(self):
        tgt = 1.0 if self._hover else 0.0
        self._anim += (tgt - self._anim) * 0.12
        self.update()

    def enterEvent(self, e): self._hover = True
    def leaveEvent(self, e): self._hover = False

    def paintEvent(self, event):
        p   = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        a   = self._anim
        W, H = self.width(), self.height()

        base = QColor(BG_CARD)
        hov  = QColor(BG_CARD_HOVER)
        r = int(base.red()   + (hov.red()   - base.red())   * a)
        g = int(base.green() + (hov.green() - base.green()) * a)
        b = int(base.blue()  + (hov.blue()  - base.blue())  * a)
        p.setPen(Qt.NoPen); p.setBrush(QColor(r, g, b))
        p.drawRoundedRect(0, 0, W, H, 16, 16)

        p.setPen(QPen(QColor(59, 130, 246, int(45 + 90 * a)), 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(1, 1, W - 2, H - 2, 15, 15)

        grad = QLinearGradient(0, 0, W, 0)
        grad.setColorAt(0, QColor(59, 130, 246, int(70 + 100 * a)))
        grad.setColorAt(1, QColor(59, 130, 246, 0))
        p.setPen(Qt.NoPen); p.setBrush(grad)
        p.drawRoundedRect(0, 0, W, 3, 2, 2)

        p.end()
        super().paintEvent(event)


# ── Glow Button ───────────────────────────────────────────────────────────────
class GlowButton(QPushButton):
    def __init__(self, text: str, accent: str, glow: str = None):
        super().__init__(text)
        self._accent   = accent
        self._glow_hex = glow or accent
        self._hover_a  = 0.0
        self._pulse_a  = 0.0
        self._active   = False
        self._pulse_t  = 0.0
        self.setFixedHeight(52)
        self.setCursor(Qt.PointingHandCursor)
        self.setFont(QFont("Inter", 12, QFont.DemiBold))
        t = QTimer(self)
        t.timeout.connect(self._tick)
        t.start(16)

    def set_active(self, v: bool):
        self._active = v

    def _tick(self):
        tgt = 1.0 if self.underMouse() else 0.0
        self._hover_a += (tgt - self._hover_a) * 0.15
        if self._active:
            self._pulse_t += 0.05
            self._pulse_a  = 0.5 + 0.5 * math.sin(self._pulse_t)
        else:
            self._pulse_a = max(0.0, self._pulse_a - 0.04)
            self._pulse_t = 0.0
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H   = self.width(), self.height()
        a      = self._hover_a
        pulse  = self._pulse_a
        ac     = QColor(self._accent)
        gc     = QColor(self._glow_hex)

        if pulse > 0.02:
            sp = int(9 * pulse)
            p.setPen(QPen(QColor(gc.red(), gc.green(), gc.blue(),
                                 int(55 * pulse)), 2))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(-sp, -sp, W + sp*2, H + sp*2, 14+sp, 14+sp)

        for i in range(7, 0, -1):
            alpha = int((a * 0.55 + pulse * 0.45) * 32 * (i / 7))
            p.setPen(QPen(QColor(gc.red(), gc.green(), gc.blue(), alpha), i*2))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(i, i, W-i*2, H-i*2, 12, 12)

        fill_a = int(28 + 55*a + 28*pulse)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(ac.red(), ac.green(), ac.blue(), fill_a))
        p.drawRoundedRect(0, 0, W, H, 12, 12)

        border_a = int(130 + 125*a)
        p.setPen(QPen(QColor(ac.red(), ac.green(), ac.blue(), border_a), 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(1, 1, W-2, H-2, 11, 11)

        text_a = int(185 + 70*a)
        p.setPen(QColor(ac.red(), ac.green(), ac.blue(), text_a))
        p.setFont(QFont("Inter", 12, QFont.DemiBold))
        p.drawText(self.rect(), Qt.AlignCenter, self.text())
        p.end()


# ── Path picker row (model / video) ───────────────────────────────────────────
class PathPickerRow(QWidget):
    """A compact label + path display + Browse button row."""
    def __init__(self, label: str, file_filter: str, parent=None):
        super().__init__(parent)
        self._filter = file_filter
        self._path   = ""
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        lbl = QLabel(label)
        lbl.setFixedWidth(80)
        lbl.setFont(QFont("Inter", 9, 57))
        lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")
        lay.addWidget(lbl)

        self._path_lbl = QLabel("Not selected")
        self._path_lbl.setFont(QFont("Inter", 9))
        self._path_lbl.setStyleSheet(f"""
            color: {TEXT_MUTED};
            background: {BG_DARK_PANEL};
            border: 1px solid {BORDER_DIM};
            border-radius: 6px;
            padding: 4px 10px;
        """)
        self._path_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lay.addWidget(self._path_lbl)

        btn = QPushButton("Browse")
        btn.setFixedSize(72, 28)
        btn.setFont(QFont("Inter", 9))
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                color: {ACCENT_BLUE};
                background: rgba(0,168,255,0.10);
                border: 1px solid rgba(0,168,255,0.35);
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background: rgba(0,168,255,0.20);
            }}
        """)
        btn.clicked.connect(self._browse)
        lay.addWidget(btn)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select file", "", self._filter)
        if path:
            self._path = path
            short = path if len(path) < 55 else "…" + path[-52:]
            self._path_lbl.setText(short)
            self._path_lbl.setStyleSheet(f"""
                color: {TEXT_PRIMARY};
                background: {BG_DARK_PANEL};
                border: 1px solid rgba(0,168,255,0.4);
                border-radius: 6px;
                padding: 4px 10px;
            """)

    @property
    def path(self) -> str:
        return self._path


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class FactoryDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FactoryVision · Bag Counter")
        self.setMinimumSize(1280, 780)
        self.resize(1920, 1080)

        self._counting     = False
        self._start_time   = None
        self._total_count  = 0
        self._overlap_count= 0
        self._thread: CountingThread | None = None

        self._setup_ui()
        self._setup_timers()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QWidget()
        root.setStyleSheet(f"background: {BG_MAIN};")
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_header())

        # File picker bar
        outer.addWidget(self._build_picker_bar())

        content = QWidget()
        content.setStyleSheet("background: transparent;")
        cl = QHBoxLayout(content)
        cl.setContentsMargins(28, 14, 28, 0)
        cl.setSpacing(24)
        cl.addWidget(self._build_left_panel(),  70)
        cl.addWidget(self._build_right_panel(), 30)
        outer.addWidget(content, 1)

        outer.addWidget(self._build_action_bar())

    def _build_header(self) -> QWidget:
        hdr = QWidget()
        hdr.setFixedHeight(60)
        hdr.setStyleSheet(f"""
            background: {BG_DARK_PANEL};
            border-bottom: 1px solid {BORDER_DIM};
        """)
        lay = QHBoxLayout(hdr)
        lay.setContentsMargins(28, 0, 28, 0)

        logo = QLabel("◈")
        logo.setFont(QFont("Inter", 18))
        logo.setStyleSheet(f"color: {ACCENT_CYAN}; background: transparent;")

        title = QLabel("FactoryVision")
        title.setFont(QFont("Inter", 16, QFont.Bold))
        title.setStyleSheet(
            f"color: {TEXT_PRIMARY}; background: transparent; letter-spacing: 1px;")

        sub = QLabel("Bag Counter")
        sub.setFont(QFont("Inter", 10))
        sub.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")

        sep = QLabel("·")
        sep.setStyleSheet(f"color: {TEXT_DISABLED}; background: transparent;")

        lay.addWidget(logo); lay.addSpacing(8)
        lay.addWidget(title); lay.addSpacing(10)
        lay.addWidget(sep);   lay.addSpacing(10)
        lay.addWidget(sub);   lay.addStretch()

        self._clock_label = QLabel()
        self._clock_label.setFont(QFont("Inter", 13, 57))
        self._clock_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; background: transparent;")
        lay.addWidget(self._clock_label)

        self._status_pill = QLabel("  SYSTEM READY  ")
        self._status_pill.setFont(QFont("Inter", 9, QFont.Bold))
        self._apply_pill(running=False)
        lay.addSpacing(20)
        lay.addWidget(self._status_pill)
        return hdr

    def _build_picker_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(56)
        bar.setStyleSheet(f"""
            background: {BG_SECONDARY};
            border-bottom: 1px solid {BORDER_DIM};
        """)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(28, 8, 28, 8)
        lay.setSpacing(24)

        self._model_picker = PathPickerRow(
            "Model (.pt)", "Model weights (*.pt *.pth)")
        self._video_picker = PathPickerRow(
            "Video", "Video files (*.mp4 *.avi *.mov *.mkv)")

        lay.addWidget(self._model_picker, 1)
        lay.addWidget(self._video_picker, 1)
        return bar

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        top = QHBoxLayout()
        cam = QLabel("CAM-01  ·  FACTORY FLOOR")
        cam.setFont(QFont("Inter", 11, 57))
        cam.setStyleSheet(
            f"color: {TEXT_SECONDARY}; background: transparent; letter-spacing: 1px;")
        top.addWidget(cam); top.addStretch()
        self._live_badge = LiveBadge()
        top.addWidget(self._live_badge)
        lay.addLayout(top)

        self._video_widget = VideoWidget()
        self._video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self._video_widget, 1)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)

        sec = QLabel("KPI OVERVIEW")
        sec.setFont(QFont("Inter", 9, 57))
        sec.setStyleSheet(
            f"color: {TEXT_SECONDARY}; background: transparent; letter-spacing: 2px;")
        lay.addWidget(sec)

        self._card_total = KPICard("Total Count", ACCENT_BLUE)
        lay.addWidget(self._card_total, 1)

        self._card_overlap = KPICard("Overlapped", AMBER_GLOW)
        lay.addWidget(self._card_overlap, 1)

        self._session_card = SessionCard()
        lay.addWidget(self._session_card, 1)
        return panel

    def _build_action_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(88)
        bar.setStyleSheet(f"""
            background: {BG_SIDEBAR};
            border-top: 1px solid {BORDER_DIM};
        """)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(28, 16, 28, 16)
        lay.setSpacing(16)

        self._hint_label = QLabel(
            "Browse a model and video above, then press  ▶  Start Counting")
        self._hint_label.setFont(QFont("Inter", 10))
        self._hint_label.setStyleSheet(
            f"color: {TEXT_DISABLED}; background: transparent;")
        lay.addWidget(self._hint_label)
        lay.addStretch()

        self._btn_start = GlowButton("▶  Start Counting", BTN_START, NEON_GREEN)
        self._btn_start.setMinimumWidth(200)
        self._btn_start.clicked.connect(self._start_counting)
        lay.addWidget(self._btn_start)

        self._btn_stop = GlowButton("■  Stop Counting", BTN_STOP_CLR, NEON_RED)
        self._btn_stop.setMinimumWidth(200)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_counting)
        lay.addWidget(self._btn_stop)
        return bar

    # ── Pill helper ───────────────────────────────────────────────────────────

    def _apply_pill(self, running: bool):
        if running:
            self._status_pill.setText("  COUNTING ACTIVE  ")
            self._status_pill.setStyleSheet(f"""
                color: {AMBER_GLOW};
                background: rgba(255,179,0,0.12);
                border: 1px solid rgba(255,179,0,0.38);
                border-radius: 12px;
                padding: 4px 12px;
                letter-spacing: 1.5px;
            """)
        else:
            self._status_pill.setText("  SYSTEM READY  ")
            self._status_pill.setStyleSheet(f"""
                color: {NEON_GREEN};
                background: rgba(0,255,136,0.10);
                border: 1px solid rgba(0,255,136,0.32);
                border-radius: 12px;
                padding: 4px 12px;
                letter-spacing: 1.5px;
            """)

    # ── Timers ────────────────────────────────────────────────────────────────

    def _setup_timers(self):
        t = QTimer(self)
        t.timeout.connect(self._update_clock)
        t.start(500)
        self._update_clock()

        self._session_timer = QTimer(self)
        self._session_timer.timeout.connect(self._update_session)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _update_clock(self):
        self._clock_label.setText(datetime.now().strftime("%H:%M:%S"))

    def _update_session(self):
        if self._start_time:
            el = datetime.now() - self._start_time
            h, rem = divmod(int(el.total_seconds()), 3600)
            m, s   = divmod(rem, 60)
            self._session_card.set_value("Elapsed", f"{h:02d}:{m:02d}:{s:02d}")

    def _on_frame(self, img: QImage):
        self._video_widget.set_frame(img)

    def _on_stats(self, total: int, overlap: int):
        self._total_count   = total
        self._overlap_count = overlap
        self._card_total.set_value(total)
        self._card_overlap.set_value(overlap)

    def _on_thread_finished(self):
        """Called when the video file ends naturally."""
        if self._counting:
            self._stop_counting(natural_end=True)

    # ── Start / Stop ──────────────────────────────────────────────────────────

    def _start_counting(self):
        model_path = self._model_picker.path
        video_path = self._video_picker.path

        if not model_path:
            QMessageBox.warning(self, "No model selected",
                                "Please browse and select a YOLO .pt model file.")
            return
        if not video_path:
            QMessageBox.warning(self, "No video selected",
                                "Please browse and select a video file.")
            return

        self._counting       = True
        self._start_time     = datetime.now()
        self._total_count    = 0
        self._overlap_count  = 0
        self._card_total.set_value(0)
        self._card_overlap.set_value(0)

        self._session_card.set_value("Start Time",
                                     self._start_time.strftime("%H:%M:%S"))
        self._session_card.set_value("Elapsed", "00:00:00")
        self._session_card.set_status(True)

        self._btn_start.setEnabled(False)
        self._btn_start.set_active(False)
        self._btn_stop.setEnabled(True)
        self._btn_stop.set_active(True)
        self._apply_pill(running=True)
        self._hint_label.setText("Counting in progress…  press  ■  to stop.")
        self._session_timer.start(1000)

        # Start background thread
        self._thread = CountingThread(model_path, video_path)
        self._thread.frame_ready.connect(self._on_frame)
        self._thread.stats_updated.connect(self._on_stats)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    def _stop_counting(self, natural_end: bool = False):
        self._counting = False
        self._session_timer.stop()

        if self._thread and self._thread.isRunning():
            self._thread.stop()
        self._thread = None

        self._video_widget.clear()
        self._btn_start.setEnabled(True)
        self._btn_start.set_active(False)
        self._btn_stop.setEnabled(False)
        self._btn_stop.set_active(False)
        self._session_card.set_status(False)
        self._apply_pill(running=False)

        msg = ("Video ended.  " if natural_end else "") + \
              f"Final count: {self._total_count} bags  |  " \
              f"Overlapped: {self._overlap_count}"
        self._hint_label.setText(msg)

    def closeEvent(self, event):
        if self._thread and self._thread.isRunning():
            self._thread.stop()
        super().closeEvent(event)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    QFontDatabase.addApplicationFont("Inter.ttf")

    pal = QPalette()
    pal.setColor(QPalette.Window,           QColor(BG_MAIN))
    pal.setColor(QPalette.WindowText,       QColor(TEXT_PRIMARY))
    pal.setColor(QPalette.Base,             QColor(BG_CARD))
    pal.setColor(QPalette.AlternateBase,    QColor(BG_DARK_PANEL))
    pal.setColor(QPalette.Text,             QColor(TEXT_PRIMARY))
    pal.setColor(QPalette.Button,           QColor(BG_CARD))
    pal.setColor(QPalette.ButtonText,       QColor(TEXT_PRIMARY))
    pal.setColor(QPalette.Highlight,        QColor(ACCENT_BLUE))
    pal.setColor(QPalette.HighlightedText,  QColor(BG_MAIN))
    app.setPalette(pal)

    win = FactoryDashboard()
    win.show()
    sys.exit(app.exec_())