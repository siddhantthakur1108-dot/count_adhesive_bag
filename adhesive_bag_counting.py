import cv2
import time
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict, deque

# ── Model ─────────────────────────────────────────────────────────────────────
model = YOLO(r"runs\detect\runs\adhesive_bag\phase1\weights\best.pt")

# ── DeepSORT ──────────────────────────────────────────────────────────────────
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

# ── Video ─────────────────────────────────────────────────────────────────────
video_path = r"C:\Users\siddh\Desktop\adhesive_bag\merged_output2.mp4"
cap        = cv2.VideoCapture(video_path)
fps        = cap.get(cv2.CAP_PROP_FPS)


FRAME_W = 640
FRAME_H = 480

# ── Video Writer ──────────────────────────────────────────────────────────── 
output_path = r"C:\Users\siddh\Desktop\adhesive_bag\output2_inference.mp4"   
fourcc      = cv2.VideoWriter_fourcc(*'mp4v')                                 
out         = cv2.VideoWriter(output_path, fourcc, fps, (FRAME_W, FRAME_H))  

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TUNING KNOBS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LINE_Y                    = (FRAME_H//2) + 40
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
count          = 0
flash_event    = None
flash_time     = 0.0
FLASH_DURATION = 0.7
frame_number   = 0

track_cx_hist        = defaultdict(lambda: deque(maxlen=SMOOTH_WINDOW))
track_cy_hist        = defaultdict(lambda: deque(maxlen=4))
track_confirmed_side = {}
pending              = {}
graveyard            = {}
recent_commits       = deque(maxlen=10) 

# ── Helpers ───────────────────────────────────────────────────────────────────
def smoothed_cx(track_id):
    hist = list(track_cx_hist[track_id])
    if not hist:
        return None
    return int(np.mean(hist))

def get_side(cy):
    return 'bottom' if cy > LINE_Y else 'top'

def in_danger_zone(cy):
    return abs(cy - LINE_Y) <= DANGER_MARGIN

def net_displacement(crossed_at_cy, current_cy, direction):
    if direction == 'forward':
        return crossed_at_cy - current_cy
    else:
        return current_cy - crossed_at_cy

def commit_cross(track_id, direction):
    global count, flash_event, flash_time
    if direction == 'forward':
        count      += 1
        flash_event = '+'
        print(f"[+] ID {track_id} FORWARD  → count = {count}")
    else:
        count       = max(0, count - 1)
        flash_event = '-'
        print(f"[-] ID {track_id} BACKWARD → count = {count}")
    flash_time = time.time()

def find_graveyard_match(cx, cy):
    best_id, best_dist = None, GRAVEYARD_MATCH_PX
    for old_id, state in list(graveyard.items()):
        if frame_number - state['frame_dropped'] > GRAVEYARD_TTL:
            graveyard.pop(old_id, None)
            continue
        dist = np.hypot(cx - state['cx'], cy - state['cy'])
        if dist < best_dist:
            best_dist = dist
            best_id   = old_id
    return best_id

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame        = cv2.resize(frame, (FRAME_W, FRAME_H))
    frame_number += 1
    seen_ids     = set()
    processed_positions = []

    # ── YOLO ──────────────────────────────────────────────────────────────────
    results    = model(frame, imgsz=640, conf=0.65, iou=0.90, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append((
                [x1, y1, x2 - x1, y2 - y1],
                float(box.conf[0]),
                int(box.cls[0])
            ))

    # ── DeepSORT ──────────────────────────────────────────────────────────────
    tracks = tracker.update_tracks(detections, frame=frame)

    # ── Danger zone tint ──────────────────────────────────────────────────────
    # overlay = frame.copy()
    # cv2.rectangle(overlay,
    #               (LINE_X - DANGER_MARGIN, 0),
    #               (LINE_X + DANGER_MARGIN, FRAME_H),
    #               (0, 50, 50), -1)
    # cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

    # ── Counting line — thick yellow ───────────────────────────────────────────
    cv2.line(frame,
             (0,LINE_Y),
             (FRAME_W, LINE_Y),
             (0, 0, 255),      # yellow (BGR)
             LINE_THICKNESS)

    # ── Per-track processing ───────────────────────────────────────────────────
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r_c, b = track.to_ltrb()
        x1, y1, x2, y2 = int(l), int(t), int(r_c), int(b)

        raw_cx = int((x1 + x2) / 2)
        cy     = int((y1 + y2) / 2)

        track_cx_hist[track_id].append(raw_cx)
        track_cy_hist[track_id].append(cy)

        cx           = smoothed_cx(track_id)

        is_duplicate = any(
            np.hypot(cx - px, cy - py) < MERGE_RADIUS
            for px, py in processed_positions
        )
        if is_duplicate:
            # Still draw a dim dot so you can see it was suppressed
            cv2.circle(frame, (cx, cy), 4, (80, 80, 80), -1)   # grey = suppressed
            continue   # skip state machine, skip counting

        processed_positions.append((cx, cy))   # mark this position as handled
        # ── rest of per-track logic continues unchanged below ─────────────────

        current_side = get_side(cy)
        seen_ids.add(track_id)

        # ── Graveyard inheritance ──────────────────────────────────────────
        if track_id not in track_confirmed_side:
            old_id = find_graveyard_match(cx, cy)
            if old_id is not None:
                track_confirmed_side[track_id] = graveyard[old_id]['side']
                if old_id in pending:
                    pending[track_id] = pending.pop(old_id)
                print(f"[RE-ID] {track_id} ← dead ID {old_id} "
                      f"(side='{graveyard[old_id]['side']}')")
                graveyard.pop(old_id, None)
            else:
                track_confirmed_side[track_id] = current_side

        confirmed_side = track_confirmed_side[track_id]

        # ── Crossing state machine ─────────────────────────────────────────
        if track_id not in pending:
            if current_side != confirmed_side:
                direction = ('forward'  if confirmed_side == 'bottom'
                             else 'backward')
                pending[track_id] = {
                    'direction'          : direction,
                    'frames_on_new_side' : 1,
                    'crossed_at_cy'      : cy,
                    'origin_cy'          : list(track_cy_hist[track_id])[0]
                                          if len(track_cy_hist[track_id]) > 1
                                          else cy,
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

                displacement = net_displacement(p['crossed_at_cy'],
                                                cy, direction)

                if (p['frames_on_new_side'] >= confirm_needed
                        and displacement >= min_disp):
                    commit_cross(track_id, direction)
                    track_confirmed_side[track_id] = current_side
                    pending.pop(track_id, None)
            else:
                direction = p['direction']
                print(f"[CANCEL] ID {track_id} returned — "
                      f"{direction} cancelled "
                      f"(frames={p['frames_on_new_side']})")
                pending.pop(track_id, None)

        # ── Draw: center dot + ID label only (no box, no arrows) ──────────
        in_zone  = in_danger_zone(cy)
        has_pend = track_id in pending

        # Dot colour: cyan = pending in zone, green = left, orange = right
        if has_pend and in_zone:
            dot_col = (0, 255, 255)
        elif current_side == 'bottom':
            dot_col = (0, 255, 0)
        else:
            dot_col = (255, 140, 0)

        # Center dot
        cv2.circle(frame, (cx, cy), 6, dot_col, -1)
        cv2.circle(frame, (cx, cy), 6, (0, 0, 0), 1)   

        # ID label (with confirmation progress if pending)
        if has_pend:
            p          = pending[track_id]
            direction  = p['direction']
            needed     = (FORWARD_CONFIRM_FRAMES if direction == 'forward'
                          else BACKWARD_CONFIRM_FRAMES)
            frames_str = f" {p['frames_on_new_side']}/{needed}"
        else:
            frames_str = ""

        # cv2.putText(frame,
        #             f"ID:{track_id}{frames_str}",
        #             (cx + 8, cy - 8),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.46, dot_col, 1)

    # ── Move dropped tracks to graveyard ──────────────────────────────────────
    for lost_id in set(track_confirmed_side.keys()) - seen_ids:
        if lost_id not in graveyard:
            hist_cx = list(track_cx_hist[lost_id])
            hist_cy = list(track_cy_hist[lost_id])
            graveyard[lost_id] = {
                'cx'           : hist_cx[-1] if hist_cx else FRAME_W //2,
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

    # ── Flash ─────────────────────────────────────────────────────────────────
    if flash_event and (time.time() - flash_time) < FLASH_DURATION:
        fc  = (0, 255, 0) if flash_event == '+' else (0, 0, 255)
        #txt = {count} if flash_event == '+' else "-1"
        cv2.putText(frame, f"{count}", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2)
    else:
        flash_event = None

    # ── HUD ───────────────────────────────────────────────────────────────────
    cv2.putText(frame, f"Bags counted: {count}", (10, FRAME_H -20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(frame,
                f"Active: {len(seen_ids)}  Pending: {len(pending)}",
                (10, FRAME_H-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160, 160, 160), 1)

    # ── Save frame ────────────────────────────────────────────────────────── 
    out.write(frame)                                                           

    cv2.imshow("Bag Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()                                                                 
cv2.destroyAllWindows()
print(f"\nFinal bag count : {count}")
print(f"Saved video     : {output_path}")