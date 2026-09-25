"""
collect_data.py — ISLBridge Live Data Recorder
────────────────────────────────────────────────────────────────────────────────
Records MediaPipe hand-landmark samples from your webcam for each ISL sign.
Run this to build YOUR OWN real dataset before training.

Usage:
    python collect_data.py                    # interactive mode (prompts each sign)
    python collect_data.py --sign A           # record only sign A
    python collect_data.py --all              # loop through every sign automatically
    python collect_data.py --list             # print all available signs

Controls during recording:
    SPACE  — start/pause recording for current sign
    S      — save collected samples and move to next sign
    R      — reset / discard samples for current sign
    Q      — quit

Output:
    data/raw/<SIGN>.npy    — (N, 63) float32 landmark arrays per sign
    data/label_map.json    — index → sign name (matches train_model.py format)
    data/dataset.npy       — merged X  (all signs)
    data/labels.npy        — merged y  (all signs)
"""

import argparse, os, json, time, sys
import cv2
import numpy as np
import mediapipe as mp

# ── Sign catalogue (must match train_model.py) ─────────────────────────────────
ALPHABETS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUMBERS   = [str(i) for i in range(10)]
PHRASES   = [
    "Hello", "Thank You", "Please", "Sorry", "Yes", "No", "Help", "Water",
    "Food", "Good", "Bad", "I Love You", "Name", "Friend", "Family",
    "Home", "School", "Doctor", "Where", "What",
]
ALL_LABELS = ALPHABETS + NUMBERS + PHRASES

SAMPLES_TARGET = 200   # samples to collect per sign (increase for better accuracy)
DATA_DIR       = "data/raw"

# ── MediaPipe ──────────────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

# ── Landmark extraction (same normalisation as app.py) ────────────────────────
def extract_landmarks(frame_bgr):
    rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if not results.multi_hand_landmarks:
        return None, frame_bgr
    lms = results.multi_hand_landmarks[0]
    pts = np.array([[l.x, l.y, l.z] for l in lms.landmark], dtype=np.float32)
    pts -= pts[0]                                                # wrist at origin
    pts /= (np.max(np.linalg.norm(pts, axis=1)) + 1e-8)         # scale-invariant
    annotated = frame_bgr.copy()
    mp_drawing.draw_landmarks(
        annotated, lms, mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 140), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(255, 160,  0), thickness=2),
    )
    return pts.flatten(), annotated

# ── Drawing helpers ───────────────────────────────────────────────────────────
def overlay(frame, sign, collected, target, recording, msg=""):
    h, w = frame.shape[:2]
    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 90), (15, 15, 20), -1)
    cv2.putText(frame, f"Sign: {sign}", (16, 38),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 200, 60), 2)
    pct = int(collected / target * 100)
    bar_w = int((w - 32) * collected / target)
    cv2.rectangle(frame, (16, 55), (w - 16, 75), (50, 50, 50), -1)
    cv2.rectangle(frame, (16, 55), (16 + bar_w, 75),
                  (0, 220, 100) if recording else (100, 100, 100), -1)
    cv2.putText(frame, f"{collected}/{target}  ({pct}%)", (16, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    # Status dot
    dot_color = (0, 220, 80) if recording else (60, 60, 60)
    cv2.circle(frame, (w - 24, 24), 10, dot_color, -1)
    label = "REC" if recording else "PAUSED"
    cv2.putText(frame, label, (w - 80, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, dot_color, 1)
    # Bottom controls
    cv2.rectangle(frame, (0, h - 40), (w, h), (15, 15, 20), -1)
    cv2.putText(frame, "SPACE: rec/pause   S: save   R: reset   Q: quit",
                (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)
    # Message
    if msg:
        cv2.putText(frame, msg, (16, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 255, 160), 2)
    return frame

# ── Core recorder ─────────────────────────────────────────────────────────────
def record_sign(cap, sign, target=SAMPLES_TARGET):
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, f"{sign}.npy")

    # Load existing samples if any
    samples = list(np.load(out_path)) if os.path.exists(out_path) else []
    recording = False
    msg       = ""
    msg_until = 0

    print(f"\n▶  Sign: [{sign}]  |  target: {target}  |  existing: {len(samples)}")
    print("   Press SPACE to start recording, S to save, R to reset, Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        lm, annotated = extract_landmarks(frame)

        if recording and lm is not None:
            samples.append(lm)
            if len(samples) >= target:
                recording = False
                msg = f"✓ {target} samples collected! Press S to save."
                msg_until = time.time() + 4

        if time.time() > msg_until:
            msg = ""

        annotated = overlay(annotated, sign, len(samples), target, recording, msg)
        cv2.imshow("ISLBridge — Data Recorder", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            recording = not recording
            msg = "Recording..." if recording else "Paused"
            msg_until = time.time() + 1.5
        elif key == ord('s'):
            if samples:
                np.save(out_path, np.array(samples, dtype=np.float32))
                print(f"   Saved {len(samples)} samples → {out_path}")
                return True, len(samples)
            else:
                msg = "No samples to save!"
                msg_until = time.time() + 2
        elif key == ord('r'):
            samples = []
            recording = False
            msg = "Reset — samples cleared."
            msg_until = time.time() + 2
            print("   Reset.")
        elif key == ord('q'):
            return False, len(samples)

    return False, len(samples)

# ── Merge all raw files into single dataset arrays ────────────────────────────
def build_dataset():
    os.makedirs("data", exist_ok=True)
    X_all, y_all = [], []
    label_map    = {}

    for idx, label in enumerate(ALL_LABELS):
        path = os.path.join(DATA_DIR, f"{label}.npy")
        if os.path.exists(path):
            arr = np.load(path)
            X_all.append(arr)
            y_all.extend([idx] * len(arr))
            label_map[str(idx)] = label
            print(f"  {label:12s}: {len(arr)} samples")

    if not X_all:
        print("No data found in data/raw/")
        return

    X = np.vstack(X_all).astype(np.float32)
    y = np.array(y_all, dtype=np.int32)

    np.save("data/dataset.npy", X)
    np.save("data/labels.npy", y)
    with open("data/label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # Also copy to model/ so train_model.py can find them
    os.makedirs("model", exist_ok=True)
    np.save("model/X_train.npy", X)
    np.save("model/y_train.npy", y)
    with open("model/label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\n✓ Dataset: {len(X)} samples  |  {len(label_map)} classes")
    print("  Saved → data/dataset.npy, data/labels.npy, data/label_map.json")
    print("  Copied → model/X_train.npy, model/y_train.npy, model/label_map.json")
    print("\nNext step:  python train_model.py")

# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ISLBridge Live Data Recorder")
    parser.add_argument("--sign",   help="Record a specific sign (e.g. --sign A)")
    parser.add_argument("--all",    action="store_true", help="Record all signs in sequence")
    parser.add_argument("--list",   action="store_true", help="List all signs")
    parser.add_argument("--merge",  action="store_true", help="Merge raw files into dataset only")
    parser.add_argument("--target", type=int, default=SAMPLES_TARGET,
                        help=f"Samples per sign (default: {SAMPLES_TARGET})")
    args = parser.parse_args()

    if args.list:
        print("Available signs:")
        print("  Alphabets:", " ".join(ALPHABETS))
        print("  Numbers  :", " ".join(NUMBERS))
        print("  Phrases  :", ", ".join(PHRASES))
        return

    if args.merge:
        print("Merging raw data files…")
        build_dataset()
        return

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if args.sign:
        sign = args.sign.strip()
        if sign not in ALL_LABELS:
            print(f"Unknown sign '{sign}'. Use --list to see all signs.")
            cap.release(); sys.exit(1)
        ok, n = record_sign(cap, sign, args.target)
        if ok:
            print(f"✓ Saved {n} samples for [{sign}]")

    elif args.all:
        print("Recording all signs. Press S after each to save and continue, Q to quit.")
        for sign in ALL_LABELS:
            ok, n = record_sign(cap, sign, args.target)
            if not ok:
                print("Stopped by user.")
                break
            print(f"  → [{sign}] {n} samples saved")
        print("\nAll done! Building merged dataset…")
        build_dataset()

    else:
        # Interactive mode
        print("ISLBridge Data Recorder")
        print("Available signs:", ", ".join(ALL_LABELS[:10]), "… (use --list to see all)")
        while True:
            sign = input("\nEnter sign to record (or 'merge' / 'quit'): ").strip()
            if sign.lower() in ('quit', 'q', 'exit'):
                break
            if sign.lower() == 'merge':
                build_dataset()
                continue
            if sign not in ALL_LABELS:
                print(f"  Unknown sign '{sign}'.")
                continue
            ok, n = record_sign(cap, sign, args.target)
            if not ok:
                break

        print("\nBuilding dataset from collected files…")
        build_dataset()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
