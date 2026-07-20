import cv2
import time
import threading
import queue
import numpy as np
import torch
import os
from ultralytics import YOLO
from collections import defaultdict, deque

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_HALF = (DEVICE == 'cuda')
print(f"[DEVICE] Running on {DEVICE}  (half precision: {USE_HALF})")
USE_HW_VIDEO_IO = False

# ── Model ─────────────────────────────────────────────────────────────────────
model = YOLO(r"C:\Users\siddh\Desktop\adhesive_bag\runs\detect\improved_overlapped\train_head_improved_overlapped\weights\best.pt")
model.to(DEVICE)
try:
    model.fuse()
except Exception as e:
    print(f"[MODEL] fuse() skipped: {e}")

CLASS_NAMES = model.names


def _resolve_class_id(name):
    for k, v in CLASS_NAMES.items():
        if v == name:
            return k
    raise ValueError(
        f"[MODEL] Expected a class named '{name}' in model.names, "
        f"but got {CLASS_NAMES}. Fix the label name or update this lookup."
    )


BAG_CLASS_ID     = _resolve_class_id('bag')
OVERLAP_CLASS_ID = _resolve_class_id('overlapped')

# ── Video ─────────────────────────────────────────────────────────────────────
FRAME_W, FRAME_H = 640, 480

video_path = r"C:\Users\siddh\Desktop\adhesive_bag\merged_overlapped.mp4"
if USE_HW_VIDEO_IO:
    gst_in = (
        f'filesrc location="{video_path}" ! qtdemux ! h264parse ! '
        f'nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx,width={FRAME_W},height={FRAME_H} ! '
        f'videoconvert ! video/x-raw,format=BGR ! '
        f'queue max-size-buffers=200 leaky=downstream ! appsink sync=true'
    )
    cap = cv2.VideoCapture(gst_in, cv2.CAP_GSTREAMER)
else:
    cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0 or np.isnan(fps):
    print("[WARN] Could not read a valid FPS from source, defaulting to 30.")
    fps = 30.0

# ── Video Writer ───────────────────────────────────────────────────────────────
output_path = r"C:\Users\siddh\Desktop\adhesive_bag\Test Videos\test1707.mp4"
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"[INFO] Created directory: {output_dir}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (FRAME_W, FRAME_H))

SHOW_PREVIEW = True

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TUNING KNOBS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LINE_Y                    = (FRAME_H // 2) + 40
SMOOTH_ALPHA              = 0.25
FORWARD_CONFIRM_FRAMES    = 1
BACKWARD_CONFIRM_FRAMES   = 8
FORWARD_MIN_DISPLACEMENT  = 10
DANGER_MARGIN             = 50
GRAVEYARD_TTL             = 35
GRAVEYARD_MATCH_PX        = 70

MERGE_RADIUS_GREEN        = 25
MERGE_RADIUS_PINK         = 30
COUNT_DEDUP_RADIUS        = 20

BAG_CONF     = 0.25
OVERLAP_CONF = 0.75

# ── STACK-PAIR SUPPRESSION ───────────────────────────────────────────────────
# A 'bag' box directly above an 'overlapped' box is the top half of the same
# physical stack, not a second object. These knobs define "directly above":
# tune against your own footage — start here and adjust if pairs are
# missed (widen) or unrelated bags get merged (narrow).
STACK_MAX_DX   = 40   # max horizontal centroid offset to count as "same column"
STACK_MIN_GAP  = -15  # allow slight vertical box overlap (negative = boxes overlap)
STACK_MAX_GAP  = 60   # max vertical gap between bag-bottom and overlap-top
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── State ─────────────────────────────────────────────────────────────────────
count          = 0
overlap_count  = 0
flash_event    = None
flash_time     = 0.0
FLASH_DURATION = 0.7
frame_number   = 0

track_coords         = {}  # For EMA smoothing
track_confirmed_side = {}
track_is_overlap     = defaultdict(bool)  # Sticky class logic
pending              = {}
graveyard            = {}
recent_commits       = deque(maxlen=20)  # (cx, cy, frame, is_ovr)
counted_tracks       = {}


# ── Threaded I/O ──────────────────────────────────────────────────────────────
class FrameReader(threading.Thread):
    def __init__(self, cap, size, queue_size=1):
        super().__init__(daemon=True)
        self.cap, self.size, self.q, self.stopped = cap, size, queue.Queue(maxsize=queue_size), False

    def run(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.q.put(None)
                break
            self.q.put(cv2.resize(frame, self.size))

    def read(self):
        try:
            return self.q.get(timeout=0.2)
        except queue.Empty:
            return None

    def stop(self):
        self.stopped = True


class FrameWriter(threading.Thread):
    def __init__(self, writer, queue_size=30):
        super().__init__(daemon=True)
        self.writer, self.q = writer, queue.Queue(maxsize=queue_size)

    def run(self):
        while True:
            f = self.q.get()
            if f is None:
                break
            self.writer.write(f)

    def write(self, frame):
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            pass

    def stop(self):
        self.q.put(None)
        self.join()


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_side(cy):
    return 'bottom' if cy > LINE_Y else 'top'


def net_displacement(crossed_at_cy, current_cy, direction):
    return crossed_at_cy - current_cy if direction == 'forward' else current_cy - crossed_at_cy


def suppress_paired_bag_candidates(candidates):
    """A 'bag' candidate sitting directly above an 'overlapped' candidate is
    the top half of the same physical stack, not a separate object. Drop
    that bag candidate so only the overlapped candidate (representing the
    whole stack) reaches the state machine — this prevents one physical
    stack from producing two tracks, two crossings, and a double count.

    candidates: list of dicts with keys id, cx, cy, is_ovr, x1, y1, x2, y2
    """
    bag_cands = [c for c in candidates if not c['is_ovr']]
    ovr_cands = [c for c in candidates if c['is_ovr']]

    kept_bags = []
    for bag in bag_cands:
        bcx, b_bottom = bag['cx'], bag['y2']
        paired = False
        for ovr in ovr_cands:
            ocx, o_top = ovr['cx'], ovr['y1']
            if abs(bcx - ocx) < STACK_MAX_DX and STACK_MIN_GAP < (o_top - b_bottom) < STACK_MAX_GAP:
                paired = True
                break
        if not paired:
            kept_bags.append(bag)

    return kept_bags + ovr_cands


def commit_cross(track_id, direction, cx, cy):
    global count, overlap_count, flash_event, flash_time
    is_ovr = track_is_overlap[track_id]

    if counted_tracks.get(track_id, False):
        return

    # Class-aware deduplication (bag and overlap don't block each other)
    for rcx, rcy, rframe, r_is_ovr in recent_commits:
        if frame_number - rframe > 15:
            continue
        if r_is_ovr == is_ovr:
            if np.hypot(cx - rcx, cy - rcy) < COUNT_DEDUP_RADIUS:
                counted_tracks[track_id] = True
                return

    recent_commits.append((cx, cy, frame_number, is_ovr))
    counted_tracks[track_id] = True

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
    flash_time = time.time()


def find_graveyard_match(cx, cy, is_ovr):
    """Class-aware graveyard matching: a lost 'bag' track must only be
    re-matched to a lost 'bag' track, never to an 'overlapped' one (and
    vice versa). Without this, two spatially-close-but-different-class
    tracks can swap side/counted state and silently corrupt the count."""
    best_id, best_dist = None, GRAVEYARD_MATCH_PX
    for old_id, state in graveyard.items():
        if state.get('is_ovr') != is_ovr:
            continue
        dist = np.hypot(cx - state['cx'], cy - state['cy'])
        if dist < best_dist:
            best_dist, best_id = dist, old_id
    return best_id


def purge_expired_graveyard():
    expired = [tid for tid, st in graveyard.items() if frame_number - st['frame_dropped'] > GRAVEYARD_TTL]
    for tid in expired:
        graveyard.pop(tid, None)
        track_confirmed_side.pop(tid, None)
        track_is_overlap.pop(tid, None)
        track_coords.pop(tid, None)
        pending.pop(tid, None)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
reader = FrameReader(cap, (FRAME_W, FRAME_H))
reader.start()
writer = FrameWriter(out)
writer.start()

try:
    while True:
        frame = reader.read()
        if frame is None:
            break

        frame_number += 1
        seen_ids = set()

        # 1. Detection + tracking (single pass, single loop)
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            iou=0.45,
            verbose=False,
            device=DEVICE,
            half=USE_HALF,
        )

        cv2.line(frame, (0, LINE_Y), (FRAME_W, LINE_Y), (0, 0, 255), 1)

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
                if class_id == BAG_CLASS_ID and conf < BAG_CONF:
                    continue
                if class_id == OVERLAP_CLASS_ID and conf < OVERLAP_CONF:
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
                        int(SMOOTH_ALPHA * raw_cx + (1 - SMOOTH_ALPHA) * pcx),
                        int(SMOOTH_ALPHA * raw_cy + (1 - SMOOTH_ALPHA) * pcy),
                    )

                cx, cy = track_coords[tid]
                candidates.append({
                    'id': tid, 'cx': cx, 'cy': cy,
                    'is_ovr': track_is_overlap[tid],
                    'x1': float(x1), 'y1': float(y1),
                    'x2': float(x2), 'y2': float(y2),
                })

        # 1b. Collapse paired top-bag + bottom-overlap candidates into one
        # so a single stack isn't tracked/counted as two objects.
        candidates = suppress_paired_bag_candidates(candidates)

        # 2. Class-specific merging: process overlaps (pink) first so they
        # have priority when two close-together candidates compete.
        candidates.sort(key=lambda c: c['is_ovr'], reverse=True)
        processed_green, processed_pink = [], []

        for cand in candidates:
            tid, cx, cy, is_ovr = cand['id'], cand['cx'], cand['cy'], cand['is_ovr']

            if is_ovr:
                if any(np.hypot(cx - px, cy - py) < MERGE_RADIUS_PINK for px, py in processed_pink):
                    continue
                processed_pink.append((cx, cy))
            else:
                if any(np.hypot(cx - px, cy - py) < MERGE_RADIUS_GREEN for px, py in processed_green):
                    continue
                processed_green.append((cx, cy))

            # 3. Graveyard inheritance (class-aware match)
            if tid not in track_confirmed_side:
                old_id = find_graveyard_match(cx, cy, is_ovr)
                if old_id:
                    track_confirmed_side[tid] = graveyard[old_id]['side']
                    track_is_overlap[tid] = track_is_overlap[old_id]
                    counted_tracks[tid] = graveyard[old_id].get('counted', False)
                    graveyard.pop(old_id, None)
                else:
                    track_confirmed_side[tid] = get_side(cy)

            confirmed_side, current_side = track_confirmed_side[tid], get_side(cy)

            # 4. Crossing state machine
            if tid not in pending:
                if current_side != confirmed_side:
                    pending[tid] = {
                        'direction': 'forward' if confirmed_side == 'bottom' else 'backward',
                        'frames': 1,
                        'crossed_at_cy': cy,
                    }
            else:
                p = pending[tid]
                if current_side != confirmed_side:
                    p['frames'] += 1
                    dist = net_displacement(p['crossed_at_cy'], cy, p['direction'])
                    req_f = FORWARD_CONFIRM_FRAMES if p['direction'] == 'forward' else BACKWARD_CONFIRM_FRAMES
                    if p['frames'] >= req_f and dist >= FORWARD_MIN_DISPLACEMENT:
                        commit_cross(tid, p['direction'], cx, cy)
                        track_confirmed_side[tid] = current_side
                        pending.pop(tid, None)
                else:
                    pending.pop(tid, None)

            # 5. Visualization
            color = (255, 0, 255) if is_ovr else (0, 255, 0)
            if not is_ovr and current_side == 'top':
                color = (255, 140, 0)  # Orange

            cv2.circle(frame, (cx, cy), 6, color, -1)
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1)
            if is_ovr:
                cv2.putText(frame, "OVR", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        # Move lost tracks to graveyard (stores is_ovr for class-aware match)
        for lost_id in set(track_confirmed_side.keys()) - seen_ids:
            if lost_id not in graveyard and lost_id in track_coords:
                lcx, lcy = track_coords[lost_id]
                graveyard[lost_id] = {
                    'cx': lcx, 'cy': lcy,
                    'side': track_confirmed_side[lost_id],
                    'frame_dropped': frame_number,
                    'counted': counted_tracks.get(lost_id, False),
                    'is_ovr': track_is_overlap.get(lost_id, False),
                }

        purge_expired_graveyard()

        # Flash / HUD
        if flash_event and (time.time() - flash_time) < FLASH_DURATION:
            txt = f"EVENT: {flash_event}"
            cv2.putText(frame, txt, (10, LINE_Y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.putText(frame, f"Bags: {count} | Overlaps: {overlap_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        writer.write(frame)
        if SHOW_PREVIEW:
            cv2.imshow("Detection Logic Integrated", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    reader.stop()
    reader.join()

    writer.stop()  # pushes sentinel + joins internally

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    for _ in range(5):
        cv2.waitKey(1)

    print("Cleanup complete.")