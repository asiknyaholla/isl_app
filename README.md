# ISLBridge — Indian Sign Language Translator
### MediaPipe Hands + TensorFlow + 3D Avatar

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## ⚡ QUICK START (copy-paste these 4 commands)

```bash
pip install -r requirements.txt
python generate_training_data.py
python train_model.py
python app.py
```
Then open → http://localhost:5000

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 📦 FULL INSTALLATION

### Step 1 — Create a virtual environment (recommended)

  Windows:
    python -m venv venv
    venv\Scripts\activate

  macOS / Linux:
    python -m venv venv
    source venv/bin/activate

### Step 2 — Install dependencies

    pip install -r requirements.txt

  ⏳ Takes 3–8 minutes (TensorFlow + MediaPipe are large).
  If TF install fails on your machine, try the CPU-only version:
    pip install tensorflow-cpu

### Step 3 — Generate synthetic training data

    python generate_training_data.py

  Creates:
    model/X_train.npy       ← 16,800 landmark feature vectors
    model/y_train.npy       ← class label indices
    model/label_map.json    ← index → sign name

### Step 4 — Train the classifier

    python train_model.py

  Creates:
    model/isl_classifier.keras   ← main saved model
    model/isl_classifier.tflite  ← lightweight TFLite version
    model/training_history.json  ← loss/accuracy log

  ⏳ Takes 30–120 seconds depending on CPU/GPU.

### Step 5 — Run the app

    python app.py

### Step 6 — Open in browser

    http://localhost:5000

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 🎥 RECORDING YOUR OWN REAL DATA (Optional but recommended)

Instead of synthetic data, record real signs from your webcam
for much better accuracy:

    # Interactive mode — prompts you sign by sign
    python collect_data.py

    # Record one specific sign
    python collect_data.py --sign A

    # Record all 56 signs in sequence
    python collect_data.py --all

    # After recording, merge all files into training arrays
    python collect_data.py --merge

  Controls during recording:
    SPACE  → start / pause recording
    S      → save and move to next sign
    R      → reset (discard current samples)
    Q      → quit

  After recording, train as normal:
    python train_model.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 🌐 PAGES

  /             Home page
  /detect       Live webcam ISL detection
  /dictionary   Sign library — browse A–Z, 0–9, phrases
  /avatar       3D hand avatar — animated signs in browser

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 🧠 ML PIPELINE

  Webcam frame (JPEG)
       ↓
  MediaPipe Hands  →  21 landmarks (x, y, z)
       ↓
  Normalise  →  wrist at origin, scale-invariant  →  63-D vector
       ↓
  TensorFlow MLP  →  Dense(256)→BN→Dropout  ×3  →  Dense(56, softmax)
       ↓
  Predicted sign + confidence score
       ↓
  Translation (9 Indian languages) + Text-to-Speech

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 📚 ISL DATASETS (for production training)

  1. INCLUDE ISL Dataset — 263 word signs, 4287 videos
     https://zenodo.org/record/4010759

  2. ISL-CSLTR — sentence-level ISL dataset
     https://data.mendeley.com/datasets/kcmpdxky7p/1

  3. ISL Alphabets (GitHub) — 12,700 images, 26 classes
     https://github.com/ayeshatasnim-h/Indian-Sign-Language-dataset

  4. Emergency ISL Words (Mendeley) — 8 emergency signs, 824 videos
     https://data.mendeley.com/datasets/2vfdm42337/1

  5. ISLVT Sentences (Mendeley) — video + gloss pairs
     https://data.mendeley.com/datasets/98mzk82wbb/1

  HOW TO USE A DATASET:
    1. Extract MediaPipe landmarks from each video/image
       using the same extract_landmarks() function in app.py
    2. Save as (N, 63) numpy arrays per sign
    3. Place in model/X_train.npy + model/y_train.npy
    4. Run: python train_model.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 🗂️ PROJECT STRUCTURE

  isl-app/
  ├── app.py                      Flask backend + full ML pipeline
  ├── collect_data.py             Live webcam data recorder
  ├── generate_training_data.py   Synthetic landmark data generator
  ├── train_model.py              TF model trainer
  ├── requirements.txt            Python dependencies
  ├── README.md                   This file
  ├── model/                      Created after training
  │   ├── isl_classifier.keras
  │   ├── isl_classifier.tflite
  │   ├── label_map.json
  │   ├── X_train.npy
  │   └── y_train.npy
  └── templates/
      ├── index.html              Home page
      ├── detect.html             Live detection
      ├── dictionary.html         Sign library
      └── avatar.html             3D avatar viewer

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 🌏 SUPPORTED SIGNS

  Alphabets : A – Z  (26 signs)
  Numbers   : 0 – 9  (10 signs)
  Phrases   : Hello, Thank You, Please, Sorry, Yes, No,
              Help, Water, Food, Good, Bad, I Love You,
              Name, Friend, Family, Home, School, Doctor,
              Where, What  (20 signs)
  TOTAL     : 56 classes

## 🌐 OUTPUT LANGUAGES

  Hindi · Tamil · Telugu · Kannada · Malayalam
  Bengali · Marathi · Gujarati · Punjabi

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 🔧 FALLBACK MODES

  The app detects what's installed and falls back gracefully:

  Mode              Requires                  Behaviour
  ─────────────────────────────────────────────────────────
  Full ML           MediaPipe + TF + model    Real detection
  MediaPipe only    MediaPipe (no model)      Skeleton only
  Demo              Nothing extra             Skin + simulated

  Check mode at: http://localhost:5000/api/status

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 🛠️ TROUBLESHOOTING

  Problem                     Fix
  ─────────────────────────────────────────────────────────
  Camera not working          Allow browser webcam permission; use Chrome
  mediapipe install fails     pip install mediapipe --no-cache-dir
  TF install fails            pip install tensorflow-cpu
  Port 5000 in use            Change port=5000 to port=5001 in app.py
  No TTS Indian voices        Install language packs in OS settings
  Low detection accuracy      Good lighting, plain bg, hold sign 1–2 sec
  Model not found             Run generate_training_data.py + train_model.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Made with ❤️ for accessibility and inclusion 🇮🇳
