#

import cv2
import time
import threading
import queue
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict, deque

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_HALF = (DEVICE == 'cuda')
print(f"[DEVICE] Running on {DEVICE}  (half precision: {USE_HALF})")

# ── Model ─────────────────────────────────────────────────────────────────────
model = YOLO(r"C:\Users\siddh\Desktop\adhesive_bag\runs\detect\improved_overlapped\train_head_improved_overlapped\weights\best.pt")
model.to(DEVICE)
try:
    model.fuse()  # fuse conv+bn layers -> faster inference, identical outputs
except Exception as e:
    print(f"[MODEL] fuse() skipped: {e}")

# ── Get class IDs from model ───────────────────────────────────────────────────
CLASS_NAMES      = model.names                          # {0: 'bag', 1: 'overlapped'}
BAG_CLASS_ID      = next((k for k, v in CLASS_NAMES.items() if v == 'bag'),        0)
OVERLAP_CLASS_ID  = next((k for k, v in CLASS_NAMES.items() if v == 'overlapped'), 1)
print(f"[MODEL] Classes → {CLASS_NAMES}")
print(f"[MODEL] bag={BAG_CLASS_ID}  overlapped={OVERLAP_CLASS_ID}")

# ── DeepSORT ──────────────────────────────────────────────────────────────────
tracker = DeepSort(
    max_age             = 10,
    n_init              = 2,
    max_cosine_distance = 0.70,
    nn_budget           = 100,
    max_iou_distance    = 0.85,
    embedder            = "mobilenet",
    half                = USE_HALF,
    bgr                 = True,
    embedder_gpu        = (DEVICE == 'cuda'),
)

# ── Video ─────────────────────────────────────────────────────────────────────
video_path = r"C:\Users\siddh\Desktop\adhesive_bag\merged_output_adhesive - Test.mp4"
cap        = cv2.VideoCapture(video_path)
fps        = cap.get(cv2.CAP_PROP_FPS)

FRAME_W = 640
FRAME_H = 480

# ── Video Writer ───────────────────────────────────────────────────────────────
output_path = r"C:\Users\siddh\Desktop\adhesive_bag\overlapped_test.mp4"
fourcc      = cv2.VideoWriter_fourcc(*'mp4v')
out         = cv2.VideoWriter(output_path, fourcc, fps, (FRAME_W, FRAME_H))

# Set to False to skip the live preview window entirely (pure background processing
# is noticeably faster — the encode/write still happens, you just don't watch it live).
SHOW_PREVIEW = True

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TUNING KNOBS  (unchanged — identical values/behaviour to the original)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LINE_Y                    = (FRAME_H // 2) + 40
LINE_THICKNESS            = 1
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
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── State ─────────────────────────────────────────────────────────────────────
count          = 0           # total individual bags (overlapped bags count as 2)
overlap_count  = 0           # how many overlapped detections crossed the line
flash_event    = None
flash_time     = 0.0
FLASH_DURATION = 0.7
frame_number   = 0

track_cx_hist        = defaultdict(lambda: deque(maxlen=SMOOTH_WINDOW))
track_cy_hist        = defaultdict(lambda: deque(maxlen=4))
track_confirmed_side = {}
track_class          = {}    # track_id → class ID (bag or overlapped)
pending              = {}
graveyard            = {}
recent_commits       = deque(maxlen=10)


# ── Threaded I/O (overlaps disk read / encode with GPU inference) ─────────────
class FrameReader(threading.Thread):
    """Reads + resizes frames in the background so disk I/O doesn't block inference."""
    def __init__(self, cap, size, queue_size=4):
        super().__init__(daemon=True)
        self.cap     = cap
        self.size    = size
        self.q       = queue.Queue(maxsize=queue_size)
        self.stopped = False

    def run(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.q.put(None)   # end-of-stream sentinel
                return
            frame = cv2.resize(frame, self.size)
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def stop(self):
        self.stopped = True


class FrameWriter(threading.Thread):
    """Writes/encodes frames in the background so mp4 encoding doesn't block inference."""
    def __init__(self, writer, queue_size=4):
        super().__init__(daemon=True)
        self.writer  = writer
        self.q       = queue.Queue(maxsize=queue_size)

    def run(self):
        while True:
            frame = self.q.get()
            if frame is None:
                return
            self.writer.write(frame)

    def write(self, frame):
        self.q.put(frame)

    def stop(self):
        self.q.put(None)
        self.join()


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_side(cy):
    return 'bottom' if cy > LINE_Y else 'top'

def in_danger_zone(cy):
    return abs(cy - LINE_Y) <= DANGER_MARGIN

def net_displacement(crossed_at_cy, current_cy, direction):
    if direction == 'forward':
        return crossed_at_cy - current_cy
    return current_cy - crossed_at_cy

def commit_cross(track_id, direction):
    global count, overlap_count, flash_event, flash_time

    # ── Fast spatial dedup (deque indexing is O(1) — no list() copy needed) ───
    cx_hist = track_cx_hist[track_id]
    cy_hist = track_cy_hist[track_id]
    cx = cx_hist[-1] if cx_hist else 0
    cy = cy_hist[-1] if cy_hist else 0

    for rcx, rcy, rframe in recent_commits:
        if frame_number - rframe > 8:
            continue
        if np.hypot(cx - rcx, cy - rcy) < DOUBLE_COUNT_RADIUS:
            print(f"[DEDUP] ID {track_id} skipped")
            return

    recent_commits.append((cx, cy, frame_number))

    # ── Determine if this is an overlapped bag ────────────────────────────────
    is_overlapped = track_class.get(track_id) == OVERLAP_CLASS_ID

    if direction == 'forward':
        if is_overlapped:
            count         += 1
            overlap_count += 1
            flash_event    = 'overlap'
            print(f"[++] ID {track_id} OVERLAPPED FORWARD  → "
                  f"count = {count}  overlap_count = {overlap_count}")
        else:
            count      += 1
            flash_event = '+'
            print(f"[+]  ID {track_id} BAG FORWARD  → count = {count}")
    else:
        if is_overlapped:
            count         = max(0, count - 2)
            overlap_count = max(0, overlap_count - 1)
            flash_event   = 'overlap_back'
            print(f"[--] ID {track_id} OVERLAPPED BACKWARD → "
                  f"count = {count}  overlap_count = {overlap_count}")
        else:
            count      = max(0, count - 1)
            flash_event = '-'
            print(f"[-]  ID {track_id} BAG BACKWARD → count = {count}")

    flash_time = time.time()

def find_graveyard_match(cx, cy):
    # TTL expiry is now handled once per frame by purge_expired_graveyard(),
    # so this just does the nearest-match scan — no more per-call mutation/pop.
    best_id, best_dist = None, GRAVEYARD_MATCH_PX
    for old_id, state in graveyard.items():
        dist = np.hypot(cx - state['cx'], cy - state['cy'])
        if dist < best_dist:
            best_dist = dist
            best_id   = old_id
    return best_id

def purge_expired_graveyard():
    """
    Removes a track from graveyard AND from every per-track dict once its TTL has
    expired. This is the fix for the main slowdown: without it, track_confirmed_side/
    track_class/track_cx_hist/track_cy_hist grow for the entire length of the video,
    and the per-frame `set(track_confirmed_side.keys()) - seen_ids` scan gets slower
    and slower as more dead IDs pile up. Now dict sizes stay bounded.
    """
    if not graveyard:
        return
    expired = [tid for tid, st in graveyard.items()
               if frame_number - st['frame_dropped'] > GRAVEYARD_TTL]
    for tid in expired:
        graveyard.pop(tid, None)
        track_confirmed_side.pop(tid, None)
        track_class.pop(tid, None)
        track_cx_hist.pop(tid, None)
        track_cy_hist.pop(tid, None)
        pending.pop(tid, None)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
reader = FrameReader(cap, (FRAME_W, FRAME_H))
reader.start()
writer = FrameWriter(out)
writer.start()

while True:
    frame = reader.read()
    if frame is None:
        break

    frame_number += 1
    seen_ids     = set()
    processed_positions = []

    # ── YOLO (batched tensor extraction — one GPU→CPU sync per frame, not per box) ─
    results    = model(frame, imgsz=640, conf=0.2, iou=0.90, verbose=False,
                        device=DEVICE, half=USE_HALF)
    detections = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy  = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss  = r.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
            detections.append((
                [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                float(conf),
                int(cls)
            ))

    # ── DeepSORT ──────────────────────────────────────────────────────────────
    tracks = tracker.update_tracks(detections, frame=frame)

    # ── Counting line ─────────────────────────────────────────────────────────
    cv2.line(frame, (0, LINE_Y), (FRAME_W, LINE_Y), (0, 0, 255), LINE_THICKNESS)

    # ── Per-track processing ───────────────────────────────────────────────────
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id      = track.track_id
        l, t, r_c, b  = track.to_ltrb()
        x1, y1, x2, y2 = int(l), int(t), int(r_c), int(b)

        raw_cx = (x1 + x2) // 2
        cy     = (y1 + y2) // 2

        cx_hist = track_cx_hist[track_id]
        cy_hist = track_cy_hist[track_id]
        cx_hist.append(raw_cx)
        cy_hist.append(cy)

        cx = int(np.mean(cx_hist))   # np.mean accepts the deque directly, no list() copy

        # ── Store / update class for this track ────────────────────────────────
        if track.det_class is not None:
            track_class[track_id] = track.det_class

        # ── Duplicate suppression ──────────────────────────────────────────────
        is_duplicate = any(
            np.hypot(cx - px, cy - py) < MERGE_RADIUS
            for px, py in processed_positions
        )
        if is_duplicate:
            cv2.circle(frame, (cx, cy), 4, (80, 80, 80), -1)  # grey = suppressed
            continue

        processed_positions.append((cx, cy))

        current_side = get_side(cy)
        seen_ids.add(track_id)

        # ── Graveyard inheritance ──────────────────────────────────────────────
        if track_id not in track_confirmed_side:
            old_id = find_graveyard_match(cx, cy)
            if old_id is not None:
                track_confirmed_side[track_id] = graveyard[old_id]['side']
                if old_id in track_class:
                    track_class[track_id] = track_class[old_id]
                if old_id in pending:
                    pending[track_id] = pending.pop(old_id)
                print(f"[RE-ID] {track_id} ← dead ID {old_id} "
                      f"(side='{graveyard[old_id]['side']}')")
                graveyard.pop(old_id, None)
            else:
                track_confirmed_side[track_id] = current_side

        confirmed_side = track_confirmed_side[track_id]

        # ── Crossing state machine ─────────────────────────────────────────────
        if track_id not in pending:
            if current_side != confirmed_side:
                direction = ('forward' if confirmed_side == 'bottom'
                             else 'backward')
                pending[track_id] = {
                    'direction'         : direction,
                    'frames_on_new_side': 1,
                    'crossed_at_cy'     : cy,
                    'origin_cy'         : cy_hist[0] if len(cy_hist) > 1 else cy,
                }
        else:
            p = pending[track_id]

            if current_side != confirmed_side:
                p['frames_on_new_side'] += 1

                direction      = p['direction']
                confirm_needed = (FORWARD_CONFIRM_FRAMES
                                  if direction == 'forward'
                                  else BACKWARD_CONFIRM_FRAMES)
                min_disp       = (FORWARD_MIN_DISPLACEMENT
                                  if direction == 'forward'
                                  else BACKWARD_MIN_DISPLACEMENT)
                displacement   = net_displacement(p['crossed_at_cy'], cy, direction)

                if (p['frames_on_new_side'] >= confirm_needed
                        and displacement >= min_disp):
                    commit_cross(track_id, direction)
                    track_confirmed_side[track_id] = current_side
                    pending.pop(track_id, None)
            else:
                print(f"[CANCEL] ID {track_id} returned — "
                      f"{p['direction']} cancelled "
                      f"(frames={p['frames_on_new_side']})")
                pending.pop(track_id, None)

        # ── Draw dot ───────────────────────────────────────────────────────────
        in_zone      = in_danger_zone(cy)
        has_pend     = track_id in pending
        is_overlapped = track_class.get(track_id) == OVERLAP_CLASS_ID

        if is_overlapped:
            dot_col = (255, 0, 255)    # magenta — overlapped bags
        elif has_pend and in_zone:
            dot_col = (0, 255, 255)    # cyan  — crossing in progress
        elif current_side == 'bottom':
            dot_col = (0, 255, 0)      # green — below line
        else:
            dot_col = (255, 140, 0)    # orange — above line

        cv2.circle(frame, (cx, cy), 6, dot_col, -1)
        cv2.circle(frame, (cx, cy), 6, (0, 0, 0), 1)

        label = "OVR" if is_overlapped else ""
        if label:
            cv2.putText(frame, label, (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, dot_col, 1)

    # ── Move dropped tracks to graveyard ──────────────────────────────────────
    for lost_id in set(track_confirmed_side.keys()) - seen_ids:
        if lost_id not in graveyard:
            hist_cx = track_cx_hist[lost_id]
            hist_cy = track_cy_hist[lost_id]
            graveyard[lost_id] = {
                'cx'           : hist_cx[-1] if hist_cx else FRAME_W // 2,
                'cy'           : hist_cy[-1] if hist_cy else FRAME_H,
                'side'         : track_confirmed_side[lost_id],
                'frame_dropped': frame_number,
            }

    # ── Graveyard: auto-commit lost forward crossings ──────────────────────────
    for lost_id, p in list(pending.items()):
        if lost_id not in seen_ids:
            state        = graveyard.get(lost_id, {})
            frames_since = frame_number - state.get('frame_dropped', frame_number)
            direction    = p['direction']

            if (direction == 'forward'
                    and frames_since >= 10
                    and p['frames_on_new_side'] >= FORWARD_CONFIRM_FRAMES):
                commit_cross(lost_id, 'forward')
                track_confirmed_side[lost_id] = 'top'
                pending.pop(lost_id, None)
                print(f"[GRAVEYARD] Committed FORWARD for lost ID {lost_id}")
            elif frames_since >= GRAVEYARD_TTL:
                pending.pop(lost_id, None)

    # ── Bound dict growth: purge anything whose graveyard TTL has expired ─────
    purge_expired_graveyard()

    # ── Flash ─────────────────────────────────────────────────────────────────
    if flash_event and (time.time() - flash_time) < FLASH_DURATION:
        if flash_event == '+':
            fc, txt = (0, 255, 0),   f"+1  [{count}]"
        elif flash_event == '-':
            fc, txt = (0, 0, 255),   f"-1  [{count}]"
        elif flash_event == 'overlap':
            fc, txt = (255, 0, 255), f"+2 OVR  [{count}]"
        else:
            fc, txt = (200, 0, 200), f"-2 OVR  [{count}]"
        cv2.putText(frame, txt, (10, LINE_Y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, fc, 2)
    else:
        flash_event = None

    # ── HUD ───────────────────────────────────────────────────────────────────
    cv2.putText(frame, f"Bags counted : {count}",
                (10, FRAME_H - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, f"Overlapped   : {overlap_count}",
                (10, FRAME_H - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
    cv2.putText(frame,
                f"Active: {len(seen_ids)}  Pending: {len(pending)}",
                (10, FRAME_H - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160, 160, 160), 1)

    # ── Save frame (offloaded to writer thread — doesn't block next inference) ─
    writer.write(frame)

    if SHOW_PREVIEW:
        cv2.imshow("Bag Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

reader.stop()
writer.stop()       # flushes any frames still queued before releasing
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\nFinal bag count    : {count}")
print(f"Overlapped crossings: {overlap_count}")
print(f"Saved video        : {output_path}")