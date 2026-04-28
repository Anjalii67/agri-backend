import os, io, time, json
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageOps
from flask import Flask, request, jsonify
from flask_cors import CORS
 
app = Flask(__name__)
CORS(app)
 
# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MODEL_PATH     = 'model/mobileplantvit_v5_final.pt'
CLASS_PATH     = 'class_names.txt'
THRESHOLD_PATH = 'model/class_thresholds_v5.json'
CONF_WARN      = 0.50
TEMPERATURE    = 0.7
 
# ─────────────────────────────────────────────
#  Load class names
# ─────────────────────────────────────────────
with open(CLASS_PATH) as f:
    CLASS_NAMES = [l.strip() for l in f if l.strip()]
NUM_CLASSES = len(CLASS_NAMES)
print(f"Classes loaded: {NUM_CLASSES}")
for i, c in enumerate(CLASS_NAMES):
    print(f"  {i:02d}: {c}")
 
# ─────────────────────────────────────────────
#  Load per-class thresholds
# ─────────────────────────────────────────────
if os.path.exists(THRESHOLD_PATH):
    with open(THRESHOLD_PATH) as f:
        THRESHOLDS = json.load(f)
    print(f"\nPer-class thresholds loaded from {THRESHOLD_PATH}")
else:
    THRESHOLDS = {cls: 0.50 for cls in CLASS_NAMES}
    print(f"\nWARNING: {THRESHOLD_PATH} not found — using 0.5 for all classes.")
    print("Download class_thresholds_v5.json from Drive and place in model\\ folder.")
 
# ─────────────────────────────────────────────
#  Load model
# ─────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel not found: {MODEL_PATH}"
        "\nDownload mobileplantvit_v5_final.pt from Drive"
        "\nand copy to: agri_app_backend\\model\\"
    )
 
device = torch.device('cpu')
print(f"\nLoading model: {MODEL_PATH}")
model = torch.jit.load(MODEL_PATH, map_location=device)
model.eval()
print("Model ready.\n")
 
# ─────────────────────────────────────────────
#  Preprocessing helpers
# ─────────────────────────────────────────────
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
 
def pad_to_square(img: Image.Image) -> Image.Image:
    """
    Aspect-ratio-preserving resize + pad to 224x224.
    Prevents distortion of 4:3 / 16:9 phone photos.
    """
    img.thumbnail((224, 224), Image.BILINEAR)
    w, h   = img.size
    pad_l  = (224 - w) // 2;  pad_r = 224 - w - pad_l
    pad_t  = (224 - h) // 2;  pad_b = 224 - h - pad_t
    return ImageOps.expand(img, (pad_l, pad_t, pad_r, pad_b), fill=(124, 116, 104))
 
def to_t(pil: Image.Image) -> torch.Tensor:
    return NORM(T.ToTensor()(pil)).unsqueeze(0)
 
# ─────────────────────────────────────────────
#  7-transform TTA
# ─────────────────────────────────────────────
def tta_predict(img: Image.Image) -> torch.Tensor:
    base = pad_to_square(img.convert('RGB'))
    ts   = [to_t(base)]
    ts.append(to_t(base.transpose(Image.FLIP_LEFT_RIGHT)))
    try:    ts.append(to_t(T.Compose([T.CenterCrop(200), T.Resize(224)])(base)))
    except: ts.append(to_t(base))
    ts.append(to_t(base.rotate(12,  resample=Image.BILINEAR)))
    ts.append(to_t(base.rotate(-12, resample=Image.BILINEAR)))
    try:    ts.append(to_t(T.ColorJitter(brightness=0.35)(base)))
    except: ts.append(to_t(base))
    try:    ts.append(to_t(T.RandomPerspective(distortion_scale=0.2, p=1.0)(base)))
    except: ts.append(to_t(base))
 
    ls = None
    with torch.no_grad():
        for t in ts:
            l  = model(t) / TEMPERATURE
            ls = l if ls is None else ls + l
    return F.softmax(ls / len(ts), dim=1).squeeze(0)
 
# ─────────────────────────────────────────────
#  Per-class threshold correction
#  Prevents mosaic_virus false positives
# ─────────────────────────────────────────────
def apply_thresholds(probs: torch.Tensor):
    for idx in probs.argsort(descending=True):
        cls  = CLASS_NAMES[idx.item()]
        conf = probs[idx].item()
        if conf >= THRESHOLDS.get(cls, 0.50):
            return idx.item(), conf
    # Fallback: just return top prediction
    idx = probs.argmax()
    return idx.item(), probs[idx].item()
 
# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────
@app.route('/', methods=['GET'])
def health():
    return jsonify({
        'status'     : 'ok',
        'model'      : MODEL_PATH,
        'classes'    : NUM_CLASSES,
        'tta'        : 7,
        'temperature': TEMPERATURE,
    })
 
@app.route('/predict', methods=['POST'])
def predict():
    t0 = time.time()
 
    if 'file' not in request.files:
        return jsonify({'error': "Missing key 'file' in request"}), 400
    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400
 
    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Cannot open image: {e}'}), 400
 
    try:
        probs = tta_predict(img)
    except Exception as e:
        return jsonify({'error': f'Inference failed: {e}'}), 500
 
    pred_idx, confidence = apply_thresholds(probs)
    cls_name = CLASS_NAMES[pred_idx]
    parts    = cls_name.split('_', 1)
    crop     = parts[0]
    disease  = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'
 
    top3p, top3i = probs.topk(3)
    top3 = [
        {
            'class'     : CLASS_NAMES[top3i[i].item()],
            'confidence': round(top3p[i].item() * 100, 1),
        }
        for i in range(3)
    ]
 
    ms   = round((time.time() - t0) * 1000, 1)
    resp = {
        'crop'          : crop,
        'disease'       : disease,
        'class'         : cls_name,
        'confidence'    : round(confidence * 100, 1),
        'low_confidence': confidence < CONF_WARN,
        'top3'          : top3,
        'elapsed_ms'    : ms,
    }
 
    print(
        f"[{ms}ms] {cls_name} | "
        f"{confidence*100:.1f}% | "
        f"size={img.size} | "
        f"{'LOW' if confidence < CONF_WARN else 'ok'}"
    )
    return jsonify(resp)
 
 
if __name__ == '__main__':
    print("="*50)
    print("Flask running on 0.0.0.0:5000")
    print("Test from phone browser: http://192.168.0.8:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
 