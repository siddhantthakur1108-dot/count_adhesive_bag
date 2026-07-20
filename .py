import sys
import cv2
import time
import math
import numpy as np
import torch
import os
from datetime import datetime
from collections import defaultdict, deque

from ultralytics import YOLO

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
#  BACKEND  —  YOLO + ByteTrack counting thread (ported from bag_counter_fixed.py)
# ══════════════════════════════════════════════════════════════════════════════

class CountingThread(QThread):
    """
    Runs YOLO detection + built-in ByteTrack tracking + line-crossing
    counting in a background thread.  Emits:
      frame_ready(QImage)      — annotated frame for the video widget
      stats_updated(int, int)  — (total_count, overlap_count)

    The detection/tracking/counting core mirrors bag_counter_fixed.py:
      - sticky overlap class (once overlapped, always overlapped)
      - EMA-smoothed centroids
      - class-specific confidence thresholds and merge radii
      - stack-pair suppression (bag directly above overlap = one stack)
      - class-aware graveyard re-matching and count de-duplication
    """
    frame_ready    = pyqtSignal(QImage)
    stats_updated  = pyqtSignal(int, int)

    # ── Tuning knobs (mirrors bag_counter_fixed.py) ───────────────────────────
    FRAME_W                   = 640
    FRAME_H                   = 480
    LINE_Y_OFFSET             = 40          # added to FRAME_H//2
    LINE_THICKNESS            = 2
    SMOOTH_ALPHA              = 0.25        # EMA smoothing factor
    FORWARD_CONFIRM_FRAMES    = 1
    BACKWARD_CONFIRM_FRAMES   = 8
    FORWARD_MIN_DISPLACEMENT  = 10
    DANGER_MARGIN             = 50
    GRAVEYARD_TTL             = 35
    GRAVEYARD_MATCH_PX        = 70
    MERGE_RADIUS_GREEN        = 25          # bag candidates
    MERGE_RADIUS_PINK         = 40          # overlapped candidates
    COUNT_DEDUP_RADIUS        = 20
    BAG_CONF                  = 0.25
    OVERLAP_CONF              = 0.75
    TRACK_IOU                 = 0.45
    FLASH_DURATION            = 0.7         # seconds

    # Stack-pair suppression: a 'bag' box directly above an 'overlapped' box
    # is the top half of the same physical stack, not a second object.
    STACK_MAX_DX   = 40   # max horizontal centroid offset for "same column"
    STACK_MIN_GAP  = -15  # allow slight vertical box overlap
    STACK_MAX_GAP  = 60   # max vertical gap between bag-bottom and overlap-top

    def __init__(self, model_path: str, video_path: str):
        super().__init__()
        self._model_path = model_path
        self._video_path = video_path
        self._running    = False
        self.writer = None
        self.output_path = ""

    # ── Public API ────────────────────────────────────────────────────────────
    def stop(self):
        self._running = False
        self.wait()

    @staticmethod
    def _resolve_class_id(class_names: dict, name: str) -> int:
        for k, v in class_names.items():
            if v == name:
                return k
        raise ValueError(
            f"[MODEL] Expected a class named '{name}' in model.names, "
            f"but got {class_names}."
        )

    # ── Thread entry ──────────────────────────────────────────────────────────
    def run(self):
        self._running = True

        device   = 'cuda' if torch.cuda.is_available() else 'cpu'
        USE_HW_VIDEO_IO = True if torch.cuda.is_available() else False
        use_half = (device == 'cuda')

        # Load model
        model = YOLO(self._model_path)
        #model.to(device)
        #try:
            #model.fuse()
        #except Exception as e:
           # print(f"[MODEL] fuse() skipped: {e}")

        class_names = model.names
        try:
            BAG_CLASS_ID     = self._resolve_class_id(class_names, 'bag')
            OVERLAP_CLASS_ID = self._resolve_class_id(class_names, 'overlapped')
        except ValueError as e:
            print(f"[MODEL] {e}")
            self._running = False
            return

        # Video source
        video_path = self._video_path
        # Probe the container for its real FPS so we can pace playback to
        # match it (cap.read() otherwise returns frames as fast as the
        # decoder can produce them, which on Jetson's hardware decoder is
        # much faster than real time — the video visibly races ahead).
        probe_cap = cv2.VideoCapture(video_path)
        source_fps = probe_cap.get(cv2.CAP_PROP_FPS) if probe_cap.isOpened() else 0
        probe_cap.release()
        if not source_fps or source_fps <= 0 or source_fps > 240 or np.isnan(source_fps):
            print(f"[VIDEO] Could not determine a valid source FPS "
                  f"({source_fps!r}), defaulting to 30.")
            source_fps = 30.0
        frame_interval = 1.0 / source_fps
        # ==================================================
        # Output Video
        # ==================================================

        output_dir = "/home/jetson/Downloads/workspace/results"
        os.makedirs(output_dir, exist_ok=True)

        video_name = os.path.splitext(
            os.path.basename(self._video_path)
        )[0]

        self.output_path = os.path.join(
            output_dir,
            f"{video_name}_processed.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        self.writer = cv2.VideoWriter(
        self.output_path,
        fourcc,
        source_fps,
        (self.FRAME_W, self.FRAME_H)
        )
        
        
        
        if USE_HW_VIDEO_IO:
            #video_path = self._video_path
            gst_in = (
                f'filesrc location="{video_path}" ! '
    		'qtdemux ! h264parse ! '
    		'nvv4l2decoder ! '
    		'nvvidconv ! '
    		f'video/x-raw,format=BGRx,width={self.FRAME_W},height={self.FRAME_H} ! '
    		'videoconvert ! '
    		'video/x-raw,format=BGR ! '
    		'appsink sync=true max-buffers=1'
                
            )
            
            cap = cv2.VideoCapture(gst_in, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(video_path)
        # cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            print(f"[VIDEO] Could not open: {self._video_path}")
            self._running = False
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

        track_coords          = {}                  # EMA-smoothed centroid
        track_confirmed_side  = {}
        track_is_overlap      = defaultdict(bool)    # sticky class
        track_count_label     = {}
        pending                = {}
        graveyard               = {}
        recent_commits          = deque(maxlen=20)    # (cx, cy, frame, is_ovr)
        counted_tracks           = {}

        # ── Inner helpers (closures over local state) ─────────────────────────
        def get_side(cy):
            return 'bottom' if cy > LY else 'top'

        def in_danger_zone(cy):
            return abs(cy - LY) <= self.DANGER_MARGIN

        def net_displacement(crossed_at_cy, cur_cy, direction):
            return (crossed_at_cy - cur_cy) if direction == 'forward' \
                   else (cur_cy - crossed_at_cy)

        def find_graveyard_match(cx, cy, is_ovr):
            """Class-aware: a lost 'bag' track may only re-match a lost
            'bag' track, never an 'overlapped' one (and vice versa)."""
            best_id, best_dist = None, self.GRAVEYARD_MATCH_PX
            for old_id, st in graveyard.items():
                if st.get('is_ovr') != is_ovr:
                    continue
                d = np.hypot(cx - st['cx'], cy - st['cy'])
                if d < best_dist:
                    best_dist, best_id = d, old_id
            return best_id

        def purge_expired_graveyard():
            expired = [tid for tid, st in graveyard.items()
                       if frame_number - st['frame_dropped'] > self.GRAVEYARD_TTL]
            for tid in expired:
                graveyard.pop(tid, None)
                track_confirmed_side.pop(tid, None)
                track_is_overlap.pop(tid, None)
                track_coords.pop(tid, None)
                pending.pop(tid, None)

        def suppress_paired_bag_candidates(candidates):
            """Drop a 'bag' candidate sitting directly above an 'overlapped'
            candidate — it's the top half of the same stack, not a second
            object, so it shouldn't get its own track/crossing/count."""
            bag_cands = [c for c in candidates if not c['is_ovr']]
            ovr_cands = [c for c in candidates if c['is_ovr']]
            kept_bags = []
            for bag in bag_cands:
                bcx, b_bottom = bag['cx'], bag['y2']
                paired = False
                for ovr in ovr_cands:
                    ocx, o_top = ovr['cx'], ovr['y1']
                    if (abs(bcx - ocx) < self.STACK_MAX_DX and
                            self.STACK_MIN_GAP < (o_top - b_bottom) < self.STACK_MAX_GAP):
                        paired = True
                        break
                if not paired:
                    kept_bags.append(bag)
            return kept_bags + ovr_cands

        def commit_cross(tid, direction, cx, cy):
            nonlocal count, overlap_count, flash_event, flash_time
            is_ovr = track_is_overlap[tid]

            if counted_tracks.get(tid, False):
                return

            # Class-aware spatial de-dup (bag and overlap don't block each other)
            for rcx, rcy, rframe, r_is_ovr in recent_commits:
                if frame_number - rframe > 15:
                    continue
                if r_is_ovr == is_ovr and np.hypot(cx - rcx, cy - rcy) < self.COUNT_DEDUP_RADIUS:
                    counted_tracks[tid] = True
                    return

            recent_commits.append((cx, cy, frame_number, is_ovr))
            counted_tracks[tid] = True

            if direction != 'forward':
                flash_event = None
                return

            if is_ovr:
                count += 1
                overlap_count += 1
                flash_event = 'overlap'
            else:
                count += 1
                flash_event = '+'

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

            # YOLO detection + ByteTrack tracking (single pass)
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                iou=self.TRACK_IOU,
                verbose=False,
                device=device,
                half=use_half,
            )

            # ── Draw counting line  (neon red) ────────────────────────────────
            cv2.line(frame, (W//2, LY), (W, LY), (255, 23, 68), self.LINE_THICKNESS)
            cv2.line(frame, (W//2, LY), (W, LY), (180, 0, 40), 1)

            # ── Collect candidates for this frame ─────────────────────────────
            candidates = []
            for r in results:
                boxes = r.boxes
                if boxes is None or boxes.id is None:
                    continue

                ids     = boxes.id.cpu().numpy().astype(int)
                xyxy    = boxes.xyxy.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                confs   = boxes.conf.cpu().numpy()

                for tid, box, class_id, conf in zip(ids, xyxy, classes, confs):
                    x1, y1, x2, y2 = box

                    # Class-specific confidence filtering
                    if class_id == BAG_CLASS_ID and conf < self.BAG_CONF:
                        continue
                    if class_id == OVERLAP_CLASS_ID and conf < self.OVERLAP_CONF:
                        continue

                    raw_cx = int((x1 + x2) / 2)
                    raw_cy = int((y1 + y2) / 2)

                    seen_ids.add(tid)

                    # Sticky class: once overlapped, always overlapped
                    if class_id == OVERLAP_CLASS_ID:
                        track_is_overlap[tid] = True

                    # EMA smoothing on centroid
                    if tid not in track_coords:
                        track_coords[tid] = (raw_cx, raw_cy)
                    else:
                        pcx, pcy = track_coords[tid]
                        track_coords[tid] = (
                            int(self.SMOOTH_ALPHA * raw_cx + (1 - self.SMOOTH_ALPHA) * pcx),
                            int(self.SMOOTH_ALPHA * raw_cy + (1 - self.SMOOTH_ALPHA) * pcy),
                        )

                    cx, cy = track_coords[tid]
                    candidates.append({
                        'id': tid, 'cx': cx, 'cy': cy,
                        'is_ovr': track_is_overlap[tid],
                        'x1': float(x1), 'y1': float(y1),
                        'x2': float(x2), 'y2': float(y2),
                    })

            # Collapse paired top-bag + bottom-overlap candidates into one
            candidates = suppress_paired_bag_candidates(candidates)

            # Class-specific merging: process overlaps (pink) first so they
            # have priority when two close-together candidates compete.
            candidates.sort(key=lambda c: c['is_ovr'], reverse=True)
            processed_green, processed_pink = [], []

            # ── Per-candidate processing ───────────────────────────────────────
            for cand in candidates:
                tid, cx, cy, is_ovr = cand['id'], cand['cx'], cand['cy'], cand['is_ovr']
                x1, y1, x2, y2 = int(cand['x1']), int(cand['y1']), int(cand['x2']), int(cand['y2'])

                if is_ovr:
                    if any(np.hypot(cx - px, cy - py) < self.MERGE_RADIUS_PINK
                           for px, py in processed_pink):
                        cv2.circle(frame, (cx, cy), 4, (80, 80, 80), -1)
                        continue
                    processed_pink.append((cx, cy))
                else:
                    if any(np.hypot(cx - px, cy - py) < self.MERGE_RADIUS_GREEN
                           for px, py in processed_green):
                        cv2.circle(frame, (cx, cy), 4, (80, 80, 80), -1)
                        continue
                    processed_green.append((cx, cy))

                current_side = get_side(cy)

                # Graveyard inheritance (class-aware match)
                if tid not in track_confirmed_side:
                    old_id = find_graveyard_match(cx, cy, is_ovr)
                    if old_id is not None:
                        track_confirmed_side[tid] = graveyard[old_id]['side']
                        track_is_overlap[tid] = track_is_overlap.get(old_id, is_ovr)
                        counted_tracks[tid] = graveyard[old_id].get('counted', False)
                        if old_id in track_count_label:
                            track_count_label[tid] = track_count_label[old_id]
                        graveyard.pop(old_id, None)
                    else:
                        track_confirmed_side[tid] = current_side

                confirmed_side = track_confirmed_side[tid]

                # Crossing state machine
                if tid not in pending:
                    if current_side != confirmed_side:
                        direction = 'forward' if confirmed_side == 'bottom' else 'backward'
                        pending[tid] = {
                            'direction'         : direction,
                            'frames_on_new_side': 1,
                            'crossed_at_cy'     : cy,
                        }
                else:
                    p = pending[tid]
                    if current_side != confirmed_side:
                        p['frames_on_new_side'] += 1
                        direction    = p['direction']
                        req_frames   = (self.FORWARD_CONFIRM_FRAMES if direction == 'forward'
                                        else self.BACKWARD_CONFIRM_FRAMES)
                        displacement = net_displacement(p['crossed_at_cy'], cy, direction)

                        if (p['frames_on_new_side'] >= req_frames
                                and displacement >= self.FORWARD_MIN_DISPLACEMENT):
                            commit_cross(tid, direction, cx, cy)
                            track_confirmed_side[tid] = current_side
                            pending.pop(tid, None)
                    else:
                        pending.pop(tid, None)

                # ── Visual annotation ─────────────────────────────────────────
                in_zone  = in_danger_zone(cy)
                has_pend = tid in pending

                if is_ovr:
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
                label_txt = f"OVR #{tid}" if is_ovr else f"#{tid}"
                lbl_size, _ = cv2.getTextSize(
                    label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                lbl_x, lbl_y = x1, max(y1 - 4, 18)
                cv2.rectangle(frame,
                              (lbl_x, lbl_y - lbl_size[1] - 4),
                              (lbl_x + lbl_size[0] + 6, lbl_y + 2),
                              box_color, -1)
                # cv2.putText(frame, label_txt,
                #             (lbl_x + 3, lbl_y - 2),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

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
                    # cv2.putText(frame, ct, (tx, ty),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    #             (0, 0, 0), 1)

            # ── Move dropped tracks to graveyard (class-aware) ────────────────
            for lost_id in set(track_confirmed_side.keys()) - seen_ids:
                if lost_id not in graveyard and lost_id in track_coords:
                    lcx, lcy = track_coords[lost_id]
                    graveyard[lost_id] = {
                        'cx'           : lcx,
                        'cy'           : lcy,
                        'side'         : track_confirmed_side[lost_id],
                        'frame_dropped': frame_number,
                        'counted'      : counted_tracks.get(lost_id, False),
                        'is_ovr'       : track_is_overlap.get(lost_id, False),
                    }

            purge_expired_graveyard()

            # ── Flash overlay ─────────────────────────────────────────────────
            if flash_event and (time.time() - flash_time) < self.FLASH_DURATION:
                fmap = {
                    '+'      : ((20, 219, 20), f"+1  [{count}]"),
                    'overlap': ((255, 0, 255), f"+1 OVR  [{count}]"),
                }
                fc, txt = fmap.get(flash_event, ((255, 255, 255), ""))
                # cv2.putText(frame, txt, (10, LY - 18),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, fc, 2)
            else:
                flash_event = None

            # ── Line label ────────────────────────────────────────────────────
            cv2.putText(frame, "COUNTING LINE",
                        (W - 145, LY - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 23, 68), 1)

            # ── Convert BGR→RGB and emit ──────────────────────────────────────
            
            if self.writer is not None:
                self.writer.write(frame)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg  = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.frame_ready.emit(qimg.copy())

        cap.release()
        
        if self.writer is not None:
            self.writer.release()
            
        self._running = False
        print(f"Saved video: {self.output_path}")


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
            "Model (.pt)", "Model weights (*.pt *.pth *.engine)")
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