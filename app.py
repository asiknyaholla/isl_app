"""
app.py — ISLBridge Flask Backend (fixed)
Fixes: avatar route, detection mode filtering, landmark normalisation
"""

import os, json, time, base64, logging
import cv2
import numpy as np
from flask import Flask, render_template, jsonify, request

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ISLBridge")

ALPHABETS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUMBERS   = [str(i) for i in range(10)]
PHRASES   = [
    "Hello","Thank You","Please","Sorry","Yes","No","Help","Water",
    "Food","Good","Bad","I Love You","Name","Friend","Family",
    "Home","School","Doctor","Where","What"
]
ALL_LABELS = ALPHABETS + NUMBERS + PHRASES

ISL_ALPHABET_INFO = {
    'A':{'description':'Closed fist with thumb beside index finger','color':'#FF6B6B'},
    'B':{'description':'Four fingers straight up, thumb tucked','color':'#FF8E53'},
    'C':{'description':'Curved hand like letter C','color':'#FFA726'},
    'D':{'description':'Index up, others curved touching thumb','color':'#FFCC02'},
    'E':{'description':'All fingers bent, thumb tucked under','color':'#66BB6A'},
    'F':{'description':'Index and thumb circle, others up','color':'#26C6DA'},
    'G':{'description':'Index pointing sideways, thumb parallel','color':'#42A5F5'},
    'H':{'description':'Index and middle pointing sideways','color':'#7E57C2'},
    'I':{'description':'Pinky up, others closed','color':'#EC407A'},
    'J':{'description':'Pinky up, draw J motion','color':'#FF6B6B'},
    'K':{'description':'Index and middle up, thumb between','color':'#FF8E53'},
    'L':{'description':'Index up, thumb out - L shape','color':'#FFA726'},
    'M':{'description':'Three fingers over thumb','color':'#FFCC02'},
    'N':{'description':'Two fingers over thumb','color':'#66BB6A'},
    'O':{'description':'All fingers and thumb form circle','color':'#26C6DA'},
    'P':{'description':'Index pointing down, thumb out','color':'#42A5F5'},
    'Q':{'description':'Index and thumb point down','color':'#7E57C2'},
    'R':{'description':'Index and middle crossed','color':'#EC407A'},
    'S':{'description':'Fist with thumb over fingers','color':'#FF6B6B'},
    'T':{'description':'Thumb between index and middle','color':'#FF8E53'},
    'U':{'description':'Index and middle up together','color':'#FFA726'},
    'V':{'description':'Index and middle spread - V sign','color':'#FFCC02'},
    'W':{'description':'Three fingers spread out','color':'#66BB6A'},
    'X':{'description':'Index finger bent/hooked','color':'#26C6DA'},
    'Y':{'description':'Thumb and pinky out','color':'#42A5F5'},
    'Z':{'description':'Index finger draws Z in air','color':'#7E57C2'},
}
ISL_NUMBERS_INFO = {
    '0':{'description':'O-shape with all fingers and thumb','color':'#FF6B6B'},
    '1':{'description':'Index finger pointing up','color':'#FF8E53'},
    '2':{'description':'Index and middle up, spread','color':'#FFA726'},
    '3':{'description':'Thumb, index, middle extended','color':'#FFCC02'},
    '4':{'description':'Four fingers up, thumb tucked','color':'#66BB6A'},
    '5':{'description':'All five fingers spread open','color':'#26C6DA'},
    '6':{'description':'Thumb and pinky touch, others up','color':'#42A5F5'},
    '7':{'description':'Thumb and ring touch, others up','color':'#7E57C2'},
    '8':{'description':'Thumb and middle touch, others up','color':'#EC407A'},
    '9':{'description':'Thumb and index touch, others up','color':'#FF6B6B'},
}
ISL_PHRASES_INFO = {
    'Hello':{'description':'Wave hand side to side near face','color':'#FF6B6B'},
    'Thank You':{'description':'Flat hand from chin moving forward','color':'#FF8E53'},
    'Please':{'description':'Circular motion on chest with flat hand','color':'#FFA726'},
    'Sorry':{'description':'Fist circular motion on chest','color':'#FFCC02'},
    'Yes':{'description':'Fist nodding up and down like head nod','color':'#66BB6A'},
    'No':{'description':'Index and middle tap thumb repeatedly','color':'#26C6DA'},
    'Help':{'description':'Fist on flat palm, lift upward','color':'#42A5F5'},
    'Water':{'description':'W handshape tapping chin','color':'#7E57C2'},
    'Food':{'description':'Flat fingers to mouth motion','color':'#EC407A'},
    'Good':{'description':'Flat hand from chin forward','color':'#FF6B6B'},
    'Bad':{'description':'Flat hand from chin, flip down','color':'#FF8E53'},
    'I Love You':{'description':'Thumb, index, pinky extended (ILY sign)','color':'#FFA726'},
    'Name':{'description':'H-hand tapping side to side','color':'#FFCC02'},
    'Friend':{'description':'Hook index fingers together','color':'#66BB6A'},
    'Family':{'description':'F-hands circle outward','color':'#26C6DA'},
    'Home':{'description':'Flat O to cheek then temple','color':'#42A5F5'},
    'School':{'description':'Clap hands with dominant hand on flat palm','color':'#7E57C2'},
    'Doctor':{'description':'D-hand tapping inner wrist','color':'#EC407A'},
    'Where':{'description':'Index finger wagging side to side','color':'#FF6B6B'},
    'What':{'description':'Index finger wagging while shrugging','color':'#FF8E53'},
}

TRANSLATIONS = {
    'hi':{'Hello':'नमस्ते','Thank You':'धन्यवाद','Please':'कृपया','Sorry':'माफ करें',
          'Yes':'हाँ','No':'नहीं','Help':'मदद','Water':'पानी','Food':'खाना',
          'Good':'अच्छा','Bad':'बुरा','I Love You':'मैं तुमसे प्यार करता हूँ',
          'Name':'नाम','Friend':'दोस्त','Family':'परिवार','Home':'घर',
          'School':'स्कूल','Doctor':'डॉक्टर','Where':'कहाँ','What':'क्या'},
    'ta':{'Hello':'வணக்கம்','Thank You':'நன்றி','Please':'தயவுசெய்து','Sorry':'மன்னிக்கவும்',
          'Yes':'ஆம்','No':'இல்லை','Help':'உதவி','Water':'தண்ணீர்','Food':'உணவு',
          'Good':'நல்லது','Bad':'கெட்டது','I Love You':'நான் உன்னை நேசிக்கிறேன்',
          'Name':'பெயர்','Friend':'நண்பர்','Family':'குடும்பம்','Home':'வீடு',
          'School':'பள்ளி','Doctor':'மருத்துவர்','Where':'எங்கே','What':'என்ன'},
    'te':{'Hello':'నమస్కారం','Thank You':'ధన్యవాదాలు','Please':'దయచేసి','Sorry':'క్షమించండి',
          'Yes':'అవును','No':'కాదు','Help':'సహాయం','Water':'నీరు','Food':'ఆహారం',
          'Good':'మంచిది','Bad':'చెడ్డది','I Love You':'నేను మిమ్మల్ని ప్రేమిస్తున్నాను',
          'Name':'పేరు','Friend':'స్నేహితుడు','Family':'కుటుంబం','Home':'ఇల్లు',
          'School':'పాఠశాల','Doctor':'వైద్యుడు','Where':'ఎక్కడ','What':'ఏమిటి'},
    'kn':{'Hello':'ನಮಸ್ಕಾರ','Thank You':'ಧನ್ಯವಾದಗಳು','Please':'ದಯವಿಟ್ಟು','Sorry':'ಕ್ಷಮಿಸಿ',
          'Yes':'ಹೌದು','No':'ಇಲ್ಲ','Help':'ಸಹಾಯ','Water':'ನೀರು','Food':'ಆಹಾರ',
          'Good':'ಒಳ್ಳೆಯದು','Bad':'ಕೆಟ್ಟದ್ದು','I Love You':'ನಾನು ನಿನ್ನನ್ನು ಪ್ರೀತಿಸುತ್ತೇನೆ',
          'Name':'ಹೆಸರು','Friend':'ಸ್ನೇಹಿತ','Family':'ಕುಟುಂಬ','Home':'ಮನೆ',
          'School':'ಶಾಲೆ','Doctor':'ವೈದ್ಯ','Where':'ಎಲ್ಲಿ','What':'ಏನು'},
    'ml':{'Hello':'നമസ്കാരം','Thank You':'നന്ദി','Please':'ദയവായി','Sorry':'ക്ഷമിക്കൂ',
          'Yes':'അതെ','No':'ഇല്ല','Help':'സഹായം','Water':'വെള്ളം','Food':'ഭക്ഷണം',
          'Good':'നല്ലത്','Bad':'മോശം','I Love You':'ഞാൻ നിന്നെ സ്നേഹിക്കുന്നു',
          'Name':'പേര്','Friend':'സുഹൃത്ത്','Family':'കുടുംബം','Home':'വീട്',
          'School':'സ്കൂൾ','Doctor':'ഡോക്ടർ','Where':'എവിടെ','What':'എന്ത്'},
    'bn':{'Hello':'নমস্কার','Thank You':'ধন্যবাদ','Please':'অনুগ্রহ করে','Sorry':'দুঃখিত',
          'Yes':'হ্যাঁ','No':'না','Help':'সাহায্য','Water':'জল','Food':'খাবার',
          'Good':'ভালো','Bad':'খারাপ','I Love You':'আমি তোমাকে ভালোবাসি',
          'Name':'নাম','Friend':'বন্ধু','Family':'পরিবার','Home':'বাড়ি',
          'School':'স্কুল','Doctor':'ডাক্তার','Where':'কোথায়','What':'কী'},
    'mr':{'Hello':'नमस्कार','Thank You':'धन्यवाद','Please':'कृपया','Sorry':'माफ करा',
          'Yes':'होय','No':'नाही','Help':'मदत','Water':'पाणी','Food':'अन्न',
          'Good':'चांगले','Bad':'वाईट','I Love You':'मी तुझ्यावर प्रेम करतो',
          'Name':'नाव','Friend':'मित्र','Family':'कुटुंब','Home':'घर',
          'School':'शाळा','Doctor':'डॉक्टर','Where':'कुठे','What':'काय'},
    'gu':{'Hello':'નમસ્તે','Thank You':'આભાર','Please':'કૃપા કરીને','Sorry':'માફ કરો',
          'Yes':'હા','No':'ના','Help':'મદદ','Water':'પાણી','Food':'ભોજન',
          'Good':'સારું','Bad':'ખરાબ','I Love You':'હું તને પ્રેમ કરું છું',
          'Name':'નામ','Friend':'મિત્ર','Family':'પરિવાર','Home':'ઘર',
          'School':'શાળા','Doctor':'ડૉક્ટર','Where':'ક્યાં','What':'શું'},
    'pa':{'Hello':'ਸਤ ਸ੍ਰੀ ਅਕਾਲ','Thank You':'ਧੰਨਵਾਦ','Please':'ਕਿਰਪਾ ਕਰਕੇ','Sorry':'ਮਾਫ਼ ਕਰੋ',
          'Yes':'ਹਾਂ','No':'ਨਹੀਂ','Help':'ਮਦਦ','Water':'ਪਾਣੀ','Food':'ਖਾਣਾ',
          'Good':'ਚੰਗਾ','Bad':'ਮਾੜਾ','I Love You':'ਮੈਂ ਤੁਹਾਨੂੰ ਪਿਆਰ ਕਰਦਾ ਹਾਂ',
          'Name':'ਨਾਮ','Friend':'ਦੋਸਤ','Family':'ਪਰਿਵਾਰ','Home':'ਘਰ',
          'School':'ਸਕੂਲ','Doctor':'ਡਾਕਟਰ','Where':'ਕਿੱਥੇ','What':'ਕੀ'},
}

# ── MediaPipe ──────────────────────────────────────────────────────────────────
if MP_AVAILABLE:
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands_detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.50,
    )

# ── TF model ───────────────────────────────────────────────────────────────────
clf_model = None
label_map = None
MODEL_PATH = os.path.join("model", "isl_classifier.keras")
LMAP_PATH  = os.path.join("model", "label_map.json")

if TF_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
        clf_model = tf.keras.models.load_model(MODEL_PATH)
        with open(LMAP_PATH) as f:
            label_map = json.load(f)
        log.info(f"Model loaded: {len(label_map)} classes")
    except Exception as e:
        log.warning(f"Model load failed: {e}")
else:
    log.warning("No model found. Run: python generate_training_data.py && python train_model.py")

# ── Landmark extraction ────────────────────────────────────────────────────────
def extract_landmarks(image_bgr):
    rgb     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)
    if not results.multi_hand_landmarks:
        return None, image_bgr, False
    lms = results.multi_hand_landmarks[0]
    annotated = image_bgr.copy()
    mp_drawing.draw_landmarks(
        annotated, lms, mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0,255,140), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(255,160,0), thickness=2),
    )
    pts = np.array([[l.x, l.y, l.z] for l in lms.landmark], dtype=np.float32)
    pts -= pts[0]                                          # wrist at origin
    pts /= (np.max(np.linalg.norm(pts, axis=1)) + 1e-8)   # scale-invariant
    return pts.flatten(), annotated, True

# ── Prediction with mode filter ───────────────────────────────────────────────
def predict_sign(landmarks_flat, mode='all'):
    """mode: 'all' | 'alpha' | 'num' | 'phrase'"""
    if clf_model is None or label_map is None:
        return None, 0.0

    probs = clf_model.predict(landmarks_flat.reshape(1,-1), verbose=0)[0]

    # Build allowed index set based on mode
    if mode == 'alpha':
        allowed = set(str(i) for i, l in label_map.items() if l in ALPHABETS)
    elif mode == 'num':
        allowed = set(str(i) for i, l in label_map.items() if l in NUMBERS)
    elif mode == 'phrase':
        allowed = set(str(i) for i, l in label_map.items() if l in PHRASES)
    else:
        allowed = set(label_map.keys())

    # Zero out disallowed classes, pick best from allowed
    masked = np.array([p if str(i) in allowed else 0.0 for i, p in enumerate(probs)])
    if masked.sum() == 0:
        return None, 0.0

    idx   = int(np.argmax(masked))
    conf  = float(masked[idx])
    label = label_map.get(str(idx), "Unknown")
    return label, conf

# ── Demo fallback ──────────────────────────────────────────────────────────────
import random, time as _time
_demo_labels    = ALL_LABELS[:12]
_demo_idx       = 0
_demo_last_time = 0.0

def demo_detection(frame):
    global _demo_idx, _demo_last_time
    now = _time.time()
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0,20,70],np.uint8), np.array([20,255,255],np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    annotated = frame.copy()
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 4000:
            cv2.drawContours(annotated, [c], -1, (0,255,140), 2)
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(annotated, (x,y), (x+w,y+h), (100,200,255), 2)
    if now - _demo_last_time >= 2.5:
        _demo_idx = (_demo_idx+1) % len(_demo_labels)
        _demo_last_time = now
    return annotated, _demo_labels[_demo_idx], random.uniform(0.78, 0.97)

# ── State ──────────────────────────────────────────────────────────────────────
_last_result      = None
_last_result_time = 0.0
_MIN_INTERVAL     = 0.8

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():      return render_template('index.html')

@app.route('/detect')
def detect():     return render_template('detect.html')

@app.route('/dictionary')
def dictionary(): return render_template('dictionary.html')

@app.route('/avatar')
def avatar():     return render_template('avatar.html')

@app.route('/api/dictionary')
def api_dictionary():
    return jsonify({'alphabets':ISL_ALPHABET_INFO,'numbers':ISL_NUMBERS_INFO,'phrases':ISL_PHRASES_INFO})

@app.route('/api/translations')
def api_translations():
    return jsonify(TRANSLATIONS)

@app.route('/api/status')
def api_status():
    return jsonify({
        'mediapipe': MP_AVAILABLE,
        'tensorflow': TF_AVAILABLE,
        'model_loaded': clf_model is not None,
        'num_classes': len(label_map) if label_map else 0,
        'mode': 'ML' if (clf_model and MP_AVAILABLE) else 'demo',
    })

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    global _last_result, _last_result_time
    data       = request.json or {}
    image_data = data.get('image','')
    language   = data.get('language','en')
    det_mode   = data.get('mode','all')   # ← detection mode from frontend

    if ',' in image_data:
        image_data = image_data.split(',')[1]
    try:
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(image_data), np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({'error':'Invalid image'}), 400
    if frame is None:
        return jsonify({'error':'Decode failed'}), 400

    now = _time.time()
    gesture = None; confidence = 0.0; mode_used = 'none'

    if MP_AVAILABLE and clf_model is not None:
        landmarks_flat, annotated, hand_found = extract_landmarks(frame)
        if hand_found:
            if now - _last_result_time >= _MIN_INTERVAL:
                gesture, confidence = predict_sign(landmarks_flat, det_mode)
                _last_result = (gesture, confidence)
                _last_result_time = now
            elif _last_result:
                gesture, confidence = _last_result
        else:
            annotated = frame
        if gesture:
            cv2.putText(annotated, f"{gesture}  {confidence*100:.0f}%",
                        (20,48), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,255,255), 3, cv2.LINE_AA)
            cv2.putText(annotated, f"{gesture}  {confidence*100:.0f}%",
                        (20,48), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,107,0), 2, cv2.LINE_AA)
        mode_used = 'mediapipe+tf'
    elif MP_AVAILABLE:
        _, annotated, _ = extract_landmarks(frame)
        mode_used = 'mediapipe_only'
    else:
        annotated, gesture, confidence = demo_detection(frame)
        mode_used = 'demo'

    _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
    proc_b64 = base64.b64encode(buf).decode('utf-8')

    translation = None
    if gesture and language != 'en':
        translation = TRANSLATIONS.get(language, {}).get(gesture)

    return jsonify({
        'processed_image': f'data:image/jpeg;base64,{proc_b64}',
        'gesture':    gesture,
        'confidence': round(confidence*100, 1) if confidence else 0,
        'translation': translation,
        'mode':       mode_used,
    })

if __name__ == '__main__':
    log.info("─────────────────────────────────────────")
    log.info("  ISLBridge → http://localhost:5000")
    log.info(f"  MediaPipe : {'OK' if MP_AVAILABLE else 'MISSING'}")
    log.info(f"  TensorFlow: {'OK' if TF_AVAILABLE else 'MISSING'}")
    log.info(f"  ML Model  : {'loaded' if clf_model else 'NOT FOUND – run train_model.py'}")
    log.info("─────────────────────────────────────────")
    app.run(debug=False, host='0.0.0.0', port=5000)
