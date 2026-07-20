import cv2
import time
import threading
import queue
import numpy as np
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict, deque

# ══════════════════════════════════════════════════════════════════════════
#  JETSON ORIN NANO SUPER — CONFIG
# ══════════════════════════════════════════════════════════════════════════
# 1) Use the exported TensorRT engine, NOT the .pt file. This is the #1 fix.
MODEL_PATH = r"/Users/apple/Downloads/best(2:07:2026).engine"

VIDEO_PATH  = r"/Users/apple/Downloads/A tileadhisive cctv footage/footage4_merged_clips.mp4"
OUTPUT_PATH = r"/Users/apple/Downloads/A tileadhisive cctv footage/results/processed_output.mp4"

FRAME_W, FRAME_H = 640, 480

# Preview costs real time per-frame (imshow + waitKey). Turn off for actual
# throughput runs; only enable when you're actively debugging.
SHOW_PREVIEW = False

# If, after all fixes, you're still short of real-time, set this >1 to
# run detection every Nth frame (tracker coasts on cached boxes in between).
# Start at 1 and only raise it if benchmarking (see below) shows you need it.
DETECT_EVERY_N_FRAMES = 1

# Try to use Jetson hardware decode/encode via GStreamer. Falls back to
# plain OpenCV if GStreamer/NVENC/NVDEC aren't available in your OpenCV build.
USE_HW_VIDEO_IO = True
# ══════════════════════════════════════════════════════════════════════════

# ── Model (TensorRT engine — no .to(), no .fuse(), no half=/device= on calls:
#    all of that is baked in at export time and passing it again just adds
#    overhead or gets silently ignored) ───────────────────────────────────────
print(f"[MODEL] Loading {MODEL_PATH}")
model = YOLO(MODEL_PATH, task="detect")

CLASS_NAMES      = model.names
BAG_CLASS_ID      = next((k for k, v in CLASS_NAMES.items() if v == 'bag'),        0)
OVERLAP_CLASS_ID  = next((k for k, v in CLASS_NAMES.items() if v == 'overlapped'), 1)

# ── DeepSORT ──────────────────────────────────────────────────────────────────
# NOTE: This runs a second CNN (mobilenet) on GPU for every detection, every
# frame, competing with your YOLO engine for the same shared memory/compute.
# If you're still not hitting real-time after the fixes below, the next
# biggest lever is replacing DeepSort with ultralytics' built-in ByteTrack
# (model.track(..., tracker="bytetrack.yaml")), which needs NO appearance
# embedder at all — see the note at the bottom of this file.
tracker = DeepSort(
    max_age             = 30,
    n_init              = 3,
    max_cosine_distance = 0.60,
    nn_budget           = 100,
    max_iou_distance    = 0.7,
    embedder            = "mobilenet",
    half                = True,
    bgr                 = True,
    embedder_gpu        = True,
)

def probe_source_fps(path):
    """Get the REAL source fps using the plain OpenCV backend, before we
    touch any custom GStreamer pipeline. Custom appsink pipelines often
    don't report CAP_PROP_FPS correctly, which silently falls back to a
    wrong guess and makes the output play faster/slower than real time."""
    probe = cv2.VideoCapture(path)
    fps = probe.get(cv2.CAP_PROP_FPS)
    frame_count = probe.get(cv2.CAP_PROP_FRAME_COUNT)
    probe.release()
    if not fps or fps <= 1 or fps > 240:
        print(f"[WARNING] Could not reliably read source FPS (got {fps}). "
              f"Defaulting to 25 — VERIFY THIS against your actual footage "
              f"(e.g. `ffprobe -v error -select_streams v:0 -show_entries "
              f"stream=r_frame_rate {path}`), or your output speed will be wrong.")
        fps = 25.0
    print(f"[PROBE] Source reports {fps:.3f} fps, {frame_count:.0f} frames "
          f"(~{frame_count/fps:.1f}s)")
    return fps


# ── Hardware-accelerated video I/O ───────────────────────────────────────────
def open_capture(path, w, h):
    """Try Jetson NVDEC hardware decode via GStreamer; fall back to OpenCV.
    NOTE: sync=0 is fine (we're not racing a live clock), but we do NOT drop
    frames — for a file source, dropping frames on read shortens the frame
    count while the writer still assumes 1/fps seconds per frame, which
    makes the output look sped up. Use a leaky-downstream queue with a
    generous limit instead, and only drop if something is actually stuck."""
    if USE_HW_VIDEO_IO:
        gst_in = (
            f'filesrc location="{path}" ! qtdemux ! h264parse ! '
            f'nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx,width={w},height={h} ! '
            f'videoconvert ! video/x-raw,format=BGR ! '
            f'queue max-size-buffers=200 leaky=downstream ! appsink sync=true'
        )
        cap = cv2.VideoCapture(gst_in, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print("[VIDEO IN] Using NVDEC hardware decode")
            return cap, True
        print("[VIDEO IN] GStreamer/NVDEC pipeline failed, falling back to OpenCV default backend")
    cap = cv2.VideoCapture(path)
    return cap, False


def open_writer(path, w, h, fps):
    """Try Jetson NVENC hardware encode via GStreamer; fall back to OpenCV mp4v (CPU)."""
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
            print("[VIDEO OUT] Using NVENC hardware encode")
            return writer
        print("[VIDEO OUT] GStreamer/NVENC pipeline failed, falling back to OpenCV mp4v (CPU, slower)")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(path, fourcc, fps, (w, h))


# Probe fps with the plain backend FIRST — this is the ground truth.
# Do not trust cap.get(CAP_PROP_FPS) after opening the custom GStreamer
# pipeline below; it's frequently wrong/zero for appsink-based pipelines.
src_fps = probe_source_fps(VIDEO_PATH)

cap, hw_decode = open_capture(VIDEO_PATH, FRAME_W, FRAME_H)
out = open_writer(OUTPUT_PATH, FRAME_W, FRAME_H, src_fps)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TUNING KNOBS  (unchanged business logic)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LINE_Y                    = (FRAME_H // 2) + 40
SMOOTH_ALPHA              = 0.25
FORWARD_CONFIRM_FRAMES    = 2
BACKWARD_CONFIRM_FRAMES   = 8
FORWARD_MIN_DISPLACEMENT  = 8
DANGER_MARGIN             = 50
GRAVEYARD_TTL             = 35
GRAVEYARD_MATCH_PX        = 70
MERGE_RADIUS_GREEN        = 10
MERGE_RADIUS_PINK         = 20
COUNT_DEDUP_RADIUS        = 10
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── State ─────────────────────────────────────────────────────────────────────
count          = 0
overlap_count  = 0
flash_event    = None
flash_time     = 0.0
FLASH_DURATION = 0.7
frame_number   = 0

track_coords         = {}
track_confirmed_side = {}
track_is_overlap     = defaultdict(bool)
pending              = {}
graveyard            = {}
recent_commits       = deque(maxlen=20)
counted_tracks       = {}

last_results = None  # cached detections for frame-skip mode

# ── Threaded I/O ──────────────────────────────────────────────────────────────
class FrameReader(threading.Thread):
    def __init__(self, cap, size, queue_size=10):
        super().__init__(daemon=True)
        self.cap, self.size, self.q, self.stopped = cap, size, queue.Queue(maxsize=queue_size), False
    def run(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.q.put(None); break
            if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
                frame = cv2.resize(frame, self.size)
            self.q.put(frame)
    def read(self): return self.q.get()
    def stop(self): self.stopped = True

class FrameWriter(threading.Thread):
    def __init__(self, writer, queue_size=30):
        super().__init__(daemon=True)
        self.writer, self.q = writer, queue.Queue(maxsize=queue_size)
    def run(self):
        while True:
            f = self.q.get()
            if f is None: break
            self.writer.write(f)
    def write(self, frame): self.q.put(frame)
    def stop(self): self.q.put(None); self.join()

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_side(cy): return 'bottom' if cy > LINE_Y else 'top'

def net_displacement(crossed_at_cy, current_cy, direction):
    return crossed_at_cy - current_cy if direction == 'forward' else current_cy - crossed_at_cy

def commit_cross(track_id, direction, cx, cy):
    global count, overlap_count, flash_event, flash_time
    is_ovr = track_is_overlap[track_id]

    if counted_tracks.get(track_id, False):
        return

    for rcx, rcy, rframe, r_is_ovr in recent_commits:
        if frame_number - rframe > 15: continue
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
        count += 1; overlap_count += 1; flash_event = 'overlap'
    else:
        count += 1; flash_event = '+'
    flash_time = time.time()

def find_graveyard_match(cx, cy):
    best_id, best_dist = None, GRAVEYARD_MATCH_PX
    for old_id, state in graveyard.items():
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

def run_detection(frame):
    """Single call, no redundant device=/half= kwargs — those are fixed
    at TensorRT export time for an .engine model."""
    results = model(frame, imgsz=640, conf=0.25, iou=0.45, verbose=False)
    detections = []
    for r in results:
        if r.boxes is None: continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append((
                [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                float(box.conf[0].cpu().numpy()),
                int(box.cls[0].cpu().numpy())
            ))
    return detections

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
reader = FrameReader(cap, (FRAME_W, FRAME_H)); reader.start()
writer = FrameWriter(out); writer.start()

bench_start = time.time()
bench_frames = 0

while True:
    frame = reader.read()
    if frame is None: break

    frame_number += 1
    seen_ids = set()

    # Optional detection frame-skipping (only reach for this if you've
    # already fixed 1-4 above and are still below real-time)
    if DETECT_EVERY_N_FRAMES <= 1 or frame_number % DETECT_EVERY_N_FRAMES == 1:
        detections = run_detection(frame)
        last_results = detections
    else:
        detections = last_results if last_results is not None else []

    tracks = tracker.update_tracks(detections, frame=frame)
    cv2.line(frame, (0, LINE_Y), (FRAME_W, LINE_Y), (0, 0, 255), 1)

    candidates = []
    for track in tracks:
        if not track.is_confirmed(): continue
        tid = track.track_id
        l, t, r_c, b = track.to_ltrb()
        rcx, rcy = int((l + r_c) / 2), int((t + b) / 2)

        if track.det_class == OVERLAP_CLASS_ID:
            track_is_overlap[tid] = True

        if tid not in track_coords:
            track_coords[tid] = (rcx, rcy)
        else:
            pcx, pcy = track_coords[tid]
            track_coords[tid] = (int(SMOOTH_ALPHA * rcx + (1 - SMOOTH_ALPHA) * pcx),
                                 int(SMOOTH_ALPHA * rcy + (1 - SMOOTH_ALPHA) * pcy))

        cx, cy = track_coords[tid]
        candidates.append({'id': tid, 'cx': cx, 'cy': cy, 'is_ovr': track_is_overlap[tid]})

    candidates.sort(key=lambda x: x['is_ovr'], reverse=True)
    processed_green, processed_pink = [], []

    for cand in candidates:
        tid, cx, cy, is_ovr = cand['id'], cand['cx'], cand['cy'], cand['is_ovr']

        if is_ovr:
            if any(np.hypot(cx - px, cy - py) < MERGE_RADIUS_PINK for px, py in processed_pink): continue
            processed_pink.append((cx, cy))
        else:
            if any(np.hypot(cx - px, cy - py) < MERGE_RADIUS_GREEN for px, py in processed_green): continue
            processed_green.append((cx, cy))

        seen_ids.add(tid)

        if tid not in track_confirmed_side:
            old_id = find_graveyard_match(cx, cy)
            if old_id:
                track_confirmed_side[tid] = graveyard[old_id]['side']
                track_is_overlap[tid] = track_is_overlap[old_id]
                counted_tracks[tid] = graveyard[old_id].get('counted', False)
                graveyard.pop(old_id, None)
            else:
                track_confirmed_side[tid] = get_side(cy)

        confirmed_side, current_side = track_confirmed_side[tid], get_side(cy)

        if tid not in pending:
            if current_side != confirmed_side:
                pending[tid] = {'direction': 'forward' if confirmed_side == 'bottom' else 'backward',
                                 'frames': 1, 'crossed_at_cy': cy}
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

        color = (255, 0, 255) if is_ovr else (0, 255, 0)
        if not is_ovr and current_side == 'top': color = (255, 140, 0)

        cv2.circle(frame, (cx, cy), 6, color, -1)
        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1)
        if is_ovr:
            cv2.putText(frame, "OVR", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    for lost_id in set(track_confirmed_side.keys()) - seen_ids:
        if lost_id not in graveyard and lost_id in track_coords:
            cx, cy = track_coords[lost_id]
            graveyard[lost_id] = {'cx': cx, 'cy': cy, 'side': track_confirmed_side[lost_id],
                                   'frame_dropped': frame_number, 'counted': counted_tracks.get(lost_id, False)}

    purge_expired_graveyard()

    if flash_event and (time.time() - flash_time) < FLASH_DURATION:
        txt = f"EVENT: {flash_event}"
        cv2.putText(frame, txt, (10, LINE_Y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    cv2.putText(frame, f"Bags: {count} | Overlaps: {overlap_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    writer.write(frame)
    if SHOW_PREVIEW:
        cv2.imshow("Detection Logic Integrated", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # ── Lightweight throughput benchmark, printed every 100 frames ──────────
    bench_frames += 1
    if bench_frames % 100 == 0:
        elapsed = time.time() - bench_start
        achieved_fps = bench_frames / elapsed
        realtime_ratio = achieved_fps / src_fps
        print(f"[BENCH] {achieved_fps:5.1f} FPS processed  "
              f"(source is {src_fps:.1f} FPS -> {realtime_ratio:5.2f}x real-time)")

reader.stop(); writer.stop(); cap.release(); out.release()
if SHOW_PREVIEW:
    cv2.destroyAllWindows()

total_elapsed = time.time() - bench_start
print(f"[DONE] Processed {frame_number} frames in {total_elapsed:.1f}s "
      f"({frame_number/total_elapsed:.1f} FPS avg, source={src_fps:.1f} FPS)")
print(f"[CHECK] Output should play for ~{frame_number/src_fps:.1f}s at {src_fps:.1f} fps. "
      f"Compare that to the original file's duration — if it doesn't match, "
      f"frames were dropped somewhere in the read pipeline and src_fps or "
      f"the capture pipeline still needs adjusting.")
