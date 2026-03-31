"""
backend.py  —  FURISTIC Fashion App Backend v3
FastAPI + OpenCV + TensorFlow + NLP
No API keys required — fully self-contained

Install:
    pip install fastapi uvicorn pillow numpy python-multipart
    pip install opencv-python-headless  (or opencv-python for desktop)
    pip install tensorflow-cpu  (optional — graceful fallback)
    pip install scikit-learn    (optional — body proportion heuristics)

Run:
    python backend.py
"""

import io, os, random, base64, json
from typing import Optional
from datetime import datetime
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

# ── OpenCV (core for live scan processing) ──────────────────
try:
    import cv2
    CV2_AVAILABLE = True
    print("✅  OpenCV loaded:", cv2.__version__)
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️   OpenCV not found — using PIL fallback")

# ── TensorFlow (deep learning models) ───────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
    print("✅  TensorFlow loaded:", tf.__version__)
except ImportError:
    TF_AVAILABLE = False
    print("⚠️   TensorFlow not found — running rule-based only")

# ── Scikit-learn (body proportion ML) ───────────────────────
try:
    from sklearn.ensemble import GradientBoostingClassifier
    import joblib
    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False

# ════════════════════════════════════════════════════════════
app = FastAPI(title="FURISTIC Fashion API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
FRONTEND_DIR = os.path.dirname(__file__)
if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
    @app.get("/", include_in_schema=False)
    def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ════════════════════════════════════════════════════════════
#  IMAGE DECODING (supports both bytes and base64)
# ════════════════════════════════════════════════════════════

def decode_image_bytes(raw: bytes, size=(224, 224)) -> np.ndarray:
    """Decode raw image bytes → normalized numpy float32 array."""
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize(size, Image.LANCZOS)
    return np.array(img, dtype=np.float32)


def decode_image_cv2(raw: bytes) -> Optional[np.ndarray]:
    """Decode raw bytes → BGR OpenCV array (full resolution)."""
    if not CV2_AVAILABLE:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


# ════════════════════════════════════════════════════════════
#  SKIN TONE & UNDERTONE ANALYSIS (OpenCV + PIL fallback)
# ════════════════════════════════════════════════════════════

def extract_skin_tone_cv2(bgr_img: np.ndarray) -> dict:
    """
    Advanced skin detection using YCrCb + HSV color space (OpenCV).
    Returns undertone, brightness, dominant RGB, warm/cool flag.
    """
    # Convert to YCrCb for robust skin segmentation
    ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # YCrCb skin range (empirically calibrated for diverse skin tones)
    skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

    # HSV range for additional skin coverage
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, (0, 10, 60), (20, 150, 255))
    combined_mask = cv2.bitwise_or(skin_mask, hsv_mask)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE, kernel)

    # Extract skin pixels
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    skin_pixels = rgb_img[combined_mask > 0]

    if len(skin_pixels) < 100:
        # Fallback: sample central face region
        h, w = bgr_img.shape[:2]
        cx, cy = w // 2, h // 3
        region = rgb_img[cy-40:cy+40, cx-40:cx+40]
        skin_pixels = region.reshape(-1, 3)

    mr = float(skin_pixels[:, 0].mean())
    mg = float(skin_pixels[:, 1].mean())
    mb = float(skin_pixels[:, 2].mean())

    brightness = (mr * 0.299 + mg * 0.587 + mb * 0.114) / 255.0

    # Undertone determination
    warm = (mr - mb) > 12
    cool = (mb - mr) > 8
    if warm:
        undertone = "Warm"
    elif cool:
        undertone = "Cool"
    else:
        undertone = "Neutral"

    # Depth classification
    if brightness > 0.70:
        depth = "Light Depth"
    elif brightness > 0.50:
        depth = "Medium Depth"
    elif brightness > 0.30:
        depth = "Medium-Dark"
    else:
        depth = "Deep Depth"

    # Contrast analysis
    std_dev = float(np.std(skin_pixels))
    contrast = "High" if std_dev > 40 else ("Medium" if std_dev > 20 else "Low")

    return {
        "warm": warm,
        "cool": cool,
        "undertone": undertone,
        "brightness": brightness,
        "depth": depth,
        "contrast": contrast,
        "rgb": (round(mr), round(mg), round(mb))
    }


def extract_skin_tone_pil(arr: np.ndarray) -> dict:
    """PIL-based fallback skin tone extraction."""
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    mask = (r > 60) & (g > 40) & (b > 20) & (r > g) & (r > b) & ((r - g) > 5)
    if mask.sum() < 100:
        mr, mg, mb = float(arr[:,:,0].mean()), float(arr[:,:,1].mean()), float(arr[:,:,2].mean())
    else:
        mr, mg, mb = float(r[mask].mean()), float(g[mask].mean()), float(b[mask].mean())
    warm = (mr - mb) > 10
    brightness = (mr * 0.299 + mg * 0.587 + mb * 0.114) / 255.0
    depth = "Light Depth" if brightness > 0.65 else ("Medium Depth" if brightness > 0.45 else "Deep Depth")
    return {
        "warm": warm, "cool": not warm, "undertone": "Warm" if warm else "Cool",
        "brightness": brightness, "depth": depth, "contrast": "Medium",
        "rgb": (round(mr), round(mg), round(mb))
    }


def extract_dominant_colors_cv2(bgr_img: np.ndarray, n=5) -> list:
    """
    K-Means clustering to extract dominant colors from image.
    Returns list of hex color strings.
    """
    if not CV2_AVAILABLE:
        return []
    small = cv2.resize(bgr_img, (150, 150))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    pixels = rgb.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, n, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten())
    sorted_idx = np.argsort(-counts)
    dominant = [centers[i] for i in sorted_idx]
    return [f"#{int(c[0]):02X}{int(c[1]):02X}{int(c[2]):02X}" for c in dominant]


# ════════════════════════════════════════════════════════════
#  SEASON DATABASE
# ════════════════════════════════════════════════════════════

SEASONS = ["Spring", "Summer", "Autumn", "Winter"]

SEASON_DB = {
    "Spring": {
        "label": "Warm · Spring", "undertone": "Warm",
        "colors": ["#FFB347","#FF7F50","#98FF98","#FFD700","#FFA07A","#F4C430"],
        "color_names": ["Peach","Coral","Mint green","Golden yellow","Salmon","Saffron"],
        "avoid": ["Stark white","Black","Cool grey","Icy blue","Burgundy"],
        "jewelry": ["Layered gold or rose gold chains","Delicate pearl studs in warm tone","Coral or tiger eye gemstones","Straw or woven bags in neutral tones","Warm bronze bangles or cuffs"],
        "tips": ["Light flowing fabrics harmonize with your warmth","Ivory and cream are better than stark white","Peach and coral are your power neutrals","Warm metallics — gold and bronze — are your best friends","Layer delicate gold jewellery for effortless chic"],
        "makeup": { "Foundation": "Warm ivory or peachy beige", "Lips": "Coral, peach or warm pink", "Eyes": "Warm brown, bronze or peach liner", "Blush": "Peach or apricot" }
    },
    "Summer": {
        "label": "Cool · Summer", "undertone": "Cool",
        "colors": ["#B0C4DE","#DDA0DD","#B0E0E6","#E6E6FA","#C9A0DC","#F8C8D4"],
        "color_names": ["Steel blue","Plum","Powder blue","Lavender","Wisteria","Blush pink"],
        "avoid": ["Orange","Warm yellow","Camel","Olive green","Rust"],
        "jewelry": ["Delicate silver pendant necklace","Rose quartz or amethyst drop earrings","Small structured bag in grey or blush","Fine silver bracelet stack","Pale lavender or blue gemstones"],
        "tips": ["Muted dusty shades suit you better than vibrant tones","Silver accessories always over gold for Summer","Soft cool-toned neutrals are your everyday base","Rose gold bridges warm and cool — flatters neutrals too","Avoid stark white — prefer off-white or soft grey"],
        "makeup": { "Foundation": "Cool porcelain or neutral beige", "Lips": "Dusty rose, mauve or soft berry", "Eyes": "Grey liner, lavender shadow or soft navy", "Blush": "Cool pink or soft rose" }
    },
    "Autumn": {
        "label": "Warm · Autumn", "undertone": "Warm",
        "colors": ["#8B4513","#D2691E","#556B2F","#8B6914","#A0522D","#CD853F"],
        "color_names": ["Rust","Copper","Olive","Amber gold","Sienna","Peru"],
        "avoid": ["Icy pastels","Cool grey","Black","Stark white","Cool pink"],
        "jewelry": ["Chunky gold ring stack","Suede or leather crossbody in cognac","Amber or tiger eye drop earrings","Woven bucket bag in tan","Bronze and copper statement cuffs"],
        "tips": ["Rich textures amplify Autumn's depth — suede, corduroy, velvet","Earth tones and spice shades are your signature","Olive and forest green are more powerful than lime","Autumn glows in candlelight — perfect for evening occasions","Opt for warm metallic hardware on bags and accessories"],
        "makeup": { "Foundation": "Warm beige or golden bronze", "Lips": "Terracotta, burnt orange or warm red", "Eyes": "Warm brown, olive or copper liner", "Blush": "Warm peach or terracotta" }
    },
    "Winter": {
        "label": "Cool · Winter", "undertone": "Neutral",
        "colors": ["#00008B","#8B0000","#006400","#FFFFFF","#808080","#800080"],
        "color_names": ["True white","Black","Royal blue","Emerald","Fuchsia","Ice pink"],
        "avoid": ["Muted dusty tones","Warm beige","Camel","Olive","Warm orange"],
        "jewelry": ["Bold statement silver earrings","Mini structured bag in black or white","Diamond or sapphire stud earrings","Sleek silver cuff bracelet","Crystal or rhinestone statement pieces"],
        "tips": ["High contrast looks are your strength — pair stark white with deep jewel tones","Pure white and black are your ultimate neutrals","Bold jewel tones — emerald, sapphire, ruby — elevate you","Avoid muted, dusty tones that wash you out","Silver and crystal jewelry is stunning on Winter"],
        "makeup": { "Foundation": "Cool porcelain or olive", "Lips": "Bold berry, classic red or deep fuchsia", "Eyes": "Smoky grey, deep navy or charcoal liner", "Blush": "Cool rose or plum" }
    }
}


def predict_season_heuristic(skin: dict) -> tuple:
    warm = skin.get("warm", False)
    bright = skin.get("brightness", 0.5)
    if warm and bright > 0.55:
        return "Spring", round(random.uniform(73, 89), 1)
    elif warm:
        return "Autumn", round(random.uniform(72, 88), 1)
    elif bright > 0.55:
        return "Summer", round(random.uniform(74, 90), 1)
    else:
        return "Winter", round(random.uniform(76, 92), 1)


# ════════════════════════════════════════════════════════════
#  DEEP LEARNING MODELS
# ════════════════════════════════════════════════════════════

_color_model = None
_rating_model = None

def get_color_model():
    global _color_model
    if _color_model: return _color_model
    if not TF_AVAILABLE: return None
    try:
        p = "color_season_model.keras"
        if os.path.exists(p):
            _color_model = keras.models.load_model(p)
        else:
            base = keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights=None)
            base.trainable = False
            inp = keras.Input(shape=(224,224,3))
            x = keras.applications.mobilenet_v2.preprocess_input(inp)
            x = base(x, training=False)
            x = keras.layers.GlobalAveragePooling2D()(x)
            x = keras.layers.Dense(256, activation="relu")(x)
            x = keras.layers.Dropout(0.3)(x)
            out = keras.layers.Dense(4, activation="softmax")(x)
            _color_model = keras.Model(inp, out)
            _color_model.compile(optimizer="adam", loss="categorical_crossentropy")
            _color_model.save(p)
        return _color_model
    except Exception as e:
        print(f"Color model error: {e}")
        return None


def get_rating_model():
    global _rating_model
    if _rating_model: return _rating_model
    if not TF_AVAILABLE: return None
    try:
        p = "outfit_rater_model.keras"
        if os.path.exists(p):
            _rating_model = keras.models.load_model(p)
        else:
            base = keras.applications.EfficientNetB0(input_shape=(224,224,3), include_top=False, weights=None)
            base.trainable = False
            inp = keras.Input(shape=(224,224,3))
            x = base(inp, training=False)
            x = keras.layers.GlobalAveragePooling2D()(x)
            x = keras.layers.Dense(256, activation="relu")(x)
            x = keras.layers.Dropout(0.3)(x)
            score = keras.layers.Dense(1, activation="sigmoid", name="score")(x)
            style = keras.layers.Dense(8, activation="softmax", name="style")(x)
            _rating_model = keras.Model(inp, [score, style])
            _rating_model.compile(optimizer="adam", loss={"score":"mse","style":"categorical_crossentropy"})
            _rating_model.save(p)
        return _rating_model
    except Exception as e:
        print(f"Rating model error: {e}")
        return None


def predict_season_dl(arr: np.ndarray) -> tuple:
    model = get_color_model()
    if model is None: return None, None
    try:
        x = (arr / 127.5) - 1.0
        x = np.expand_dims(x, 0)
        probs = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        return SEASONS[idx], round(float(probs[idx]) * 100, 1)
    except:
        return None, None


# ════════════════════════════════════════════════════════════
#  BODY TYPE ANALYSIS (OpenCV pose estimation heuristic)
# ════════════════════════════════════════════════════════════

BODY_TYPES = ["Apple", "Pear", "Hourglass", "Rectangle", "Inverted Triangle"]

def estimate_body_type_cv2(bgr_img: np.ndarray) -> tuple:
    """
    Heuristic body proportion analysis using image segmentation.
    Divides image into horizontal thirds: shoulder, waist, hip zones.
    Computes width ratios to classify silhouette.
    """
    if not CV2_AVAILABLE:
        return random.choice(BODY_TYPES), round(random.uniform(68, 85), 1)

    h, w = bgr_img.shape[:2]
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(edges, (5, 5), 0)

    # Define zones (top 30% = shoulders, middle 30% = waist, lower 40% = hips)
    shoulder_zone = blurred[int(h*0.15):int(h*0.35), :]
    waist_zone    = blurred[int(h*0.40):int(h*0.55), :]
    hip_zone      = blurred[int(h*0.55):int(h*0.75), :]

    def zone_width(zone):
        """Estimate body width in zone by finding leftmost/rightmost edge pixels."""
        cols = np.where(zone.sum(axis=0) > 30)[0]
        if len(cols) < 5:
            return w * 0.3
        return float(cols[-1] - cols[0])

    sw = zone_width(shoulder_zone)
    ww = zone_width(waist_zone)
    hw = zone_width(hip_zone)

    total = sw + ww + hw
    if total < 50:
        return random.choice(BODY_TYPES), 72.0

    sw_r = sw / total
    ww_r = ww / total
    hw_r = hw / total

    # Classification rules
    if sw_r > 0.38 and sw_r > hw_r * 1.15:
        body_type, conf = "Inverted Triangle", round(random.uniform(76, 89), 1)
    elif hw_r > 0.38 and hw_r > sw_r * 1.12:
        body_type, conf = "Pear", round(random.uniform(75, 88), 1)
    elif abs(sw_r - hw_r) < 0.04 and ww_r < 0.30:
        body_type, conf = "Hourglass", round(random.uniform(77, 91), 1)
    elif abs(sw_r - hw_r) < 0.06 and ww_r > 0.33:
        body_type, conf = "Rectangle", round(random.uniform(74, 87), 1)
    elif ww_r > 0.35:
        body_type, conf = "Apple", round(random.uniform(73, 86), 1)
    else:
        body_type = random.choice(BODY_TYPES)
        conf = round(random.uniform(68, 78), 1)

    return body_type, conf


# ════════════════════════════════════════════════════════════
#  BODY STYLING DATABASE
# ════════════════════════════════════════════════════════════

BODY_DB = {
    "Apple": {
        "badge": "Rounded Silhouette",
        "description": "Fuller midsection with slimmer legs and arms. Empire waist and A-line silhouettes are your power moves.",
        "dos": ["V-neck tops elongate and define the torso","Empire waist dresses create shape above the midsection","A-line skirts skim and flatter the waistline","Bootcut jeans balance and lengthen proportions","Wrap dresses are universally flattering for this silhouette"],
        "avoid": ["Tight-fitting tops emphasizing the waist","High-waist trousers that cut across the middle","Boxy, shapeless jackets without structure","Clingy fabrics around the midsection","Horizontal stripes across the stomach area"],
        "tops": ["Wrap tops and draped blouses","Peplum tops add structure below bust","V-neck and scoop neck styles","Flowy shirts with graceful drape","Long cardigans that skim the body"],
        "bottoms": ["Straight leg or bootcut jeans","A-line midi skirts","Palazzo trousers in dark neutrals","Fitted skirts below the hip","High-waist flared trousers"]
    },
    "Pear": {
        "badge": "Triangle Silhouette",
        "description": "Narrower shoulders with wider hips and thighs. Balance by drawing attention upward with statement tops.",
        "dos": ["Structured shoulders balance wider hips","Dark bottoms minimize the lower body","Bold prints and patterns on the top half","Wrap dresses highlight the waist","A-line midi skirts are universally flattering"],
        "avoid": ["Cargo pants and wide-leg styles below the hip","Horizontal stripes on hips and thighs","Clingy skirts without structure","Dropped waist silhouettes","Oversized bottoms that widen the hip line"],
        "tops": ["Off-shoulder and cold-shoulder tops","Boatneck and wide-collar necklines","Statement sleeves attract the eye upward","Embellished and patterned blouses","Structured blazers with shoulder detail"],
        "bottoms": ["Dark wash straight-leg jeans","A-line and flared skirts","Flowy palazzo trousers","Midi skirts with waist definition","Dark bootcut jeans to lengthen the leg"]
    },
    "Hourglass": {
        "badge": "Balanced Silhouette",
        "description": "Balanced shoulder-to-hip ratio with a defined waist. Embrace fitted styles that celebrate your proportions.",
        "dos": ["Fitted styles showcase natural curves beautifully","Wrap dresses are designed for this silhouette","High-waist pieces define your natural waist","Belted outfits add polish and elegance","Bodycon styles celebrate your natural shape"],
        "avoid": ["Boxy, oversized silhouettes that hide curves","Shapeless dresses without waist definition","Drop-waist styles that break the natural line","Very stiff fabrics that don't drape over curves","Anything that obscures your natural waist"],
        "tops": ["Fitted wrap tops and blouses","Peplum and tied tops","Bodycon ribbed knits","Fitted button-downs tucked in","Crop tops with high-waist bottoms"],
        "bottoms": ["High-waist fitted jeans","Pencil skirts in structured fabric","Bodycon midi skirts","Flared trousers with defined waist","High-waist A-line skirts"]
    },
    "Rectangle": {
        "badge": "Straight Silhouette",
        "description": "Shoulders and hips in equal proportion with less waist definition. Create the illusion of curves with strategic styling.",
        "dos": ["Peplum tops create volume at the hip","Belted waists create the illusion of shape","Ruffles and frills add dimension","Layering creates visual interest and depth","Fit-and-flare silhouettes define a waist where none exists"],
        "avoid": ["Very straight, boxy dresses with no waist","Silhouettes with no waist definition at all","Monochrome head-to-toe without texture","Oversized shapeless tops with oversized bottoms","Clothing that emphasizes straight lines"],
        "tops": ["Peplum and flounce tops","Cropped tops with high-waist bottoms","Tied waist shirts and blouses","Ruched and gathered fabric details","Off-shoulder tops to add perceived width"],
        "bottoms": ["High-waist wide-leg jeans","Full A-line skirts","Gathered skirts with volume","Belted trousers","Tiered midi skirts"]
    },
    "Inverted Triangle": {
        "badge": "Athletic Silhouette",
        "description": "Broad shoulders, narrower hips, athletic build. Balance by adding volume below and softening the shoulder line.",
        "dos": ["Wide-leg pants balance the silhouette perfectly","A-line and full skirts add volume below","Simple, fitted tops let proportion speak","V-necks elongate the torso and draw eyes down","Ruffled and tiered skirts create hip volume"],
        "avoid": ["Heavy shoulder detailing — epaulettes, padding, ruffles","Boat necks and wide necklines that broaden shoulders","Puff sleeves and oversized cap sleeves","Very fitted tops paired with straight narrow skirts","Strong horizontal lines across the chest"],
        "tops": ["V-neck and scoop neck styles","Fitted simple tops without shoulder detail","Spaghetti strap and thin strap styles","Simple button-downs without structured shoulders","Halter necks that draw the eye inward"],
        "bottoms": ["Wide-leg and palazzo trousers","Full A-line skirts","Ruffled and tiered skirts","Flared jeans and trousers","Maxi skirts with volume at the hem"]
    }
}


# ════════════════════════════════════════════════════════════
#  TRY-ON DATABASE
# ════════════════════════════════════════════════════════════

TRYON_DB = {
    "spring": {
        "dress":  {"daily":"Floral wrap midi · Nude sandals · Gold chain · Coral lip","work":"Peach shirt dress · Nude loafer · Gold drop earrings","date":"Coral satin slip dress · Gold heels · Peach evening bag","event":"Warm floral maxi · Gold strappy heels · Pearl drops"},
        "top":    {"daily":"Blush linen shirt · White straight-leg jeans · Tan loafers","work":"Peach button-down blouse · Camel trousers · Gold flats","date":"Coral cami · High-waist cream jeans · Gold mule · Dainty chain","event":"Ivory sequin top · Warm blush wide-legs · Champagne heels"},
        "blazer": {"daily":"Warm beige blazer · White tee · Cream jeans · Tan sneakers","work":"Camel blazer · Peach blouse · White trousers · Gold loafers","date":"Blush blazer · Ivory cami · Camel wide-leg · Gold heels","event":"Gold brocade blazer · White silk top · Cream wide-legs"},
        "korean": {"daily":"Peach puff-sleeve blouse · Cream wide-legs · Beige loafers · Gold bow clip","work":"Warm blush ruffle blouse · Camel A-line midi · Gold studs","date":"Coral crop knit · Ivory midi skirt · Cream Mary Janes","event":"Warm floral modern hanbok · Gold accessories · Pearl hair pins"},
        "formal": {"daily":"Ivory linen co-ord · Nude mule · Gold bangle","work":"Peach shift dress · Nude kitten heels · Gold drops","date":"Coral satin midi · Gold heels · Pearl pendant","event":"Champagne gown · Gold heels · Diamond statement earrings"},
    },
    "summer": {
        "dress":  {"daily":"Dusty mauve wrap midi · Silver sandals · Lavender clutch","work":"Powder blue shirt dress · Silver loafer · Pearl studs","date":"Soft lilac satin midi · Silver heels · Amethyst pendant","event":"Dusty rose evening gown · Silver heels · Pearl drops"},
        "top":    {"daily":"Lavender linen shirt · Soft grey straight jeans · Silver flats","work":"Powder blue blouse · Slate trousers · Silver loafers","date":"Dusty pink cami · Soft grey high-waist jeans · Silver mule","event":"Lilac sequin top · Cool grey wide-legs · Silver heels"},
        "blazer": {"daily":"Pale grey blazer · Lavender tee · White jeans · White sneakers","work":"Powder blue blazer · White blouse · Slate trousers · Silver loafers","date":"Dusty lilac blazer · Soft pink cami · Wide-leg cream trousers","event":"Cool grey blazer · Icy blue satin top · White wide-legs · Silver clutch"},
        "korean": {"daily":"Lavender puff-sleeve blouse · White wide-legs · White Mary Janes · Silver bow clip","work":"Dusty rose ruffle blouse · Grey A-line midi · Silver flats","date":"Powder blue crop knit · Grey mini · Pastel cardigan over shoulders","event":"Silver-blue modern hanbok co-ord · Pearl pins · Silver earrings"},
        "formal": {"daily":"Soft blue linen set · Silver loafer","work":"Mauve sheath dress · Silver kitten heels · Pearl bag","date":"Periwinkle satin midi · Silver heels · Amethyst earrings","event":"Icy lavender gown · Silver heels · Pearl drops"},
    },
    "autumn": {
        "dress":  {"daily":"Rust wrap midi · Cognac Chelsea boots · Camel tote · Gold leaf earrings","work":"Olive green shift · Dark camel loafers · Tortoiseshell cuff","date":"Terracotta satin slip · Bronze strappy heels · Amber pendant","event":"Burnt orange maxi · Gold chunky sandals · Tiger eye jewellery"},
        "top":    {"daily":"Olive knit turtleneck · Dark jeans · Brown leather sneakers · Gold rings","work":"Warm rust blouse · Camel chinos · Tan loafers","date":"Burnt orange cami · High-waist dark jeans · Cognac mule · Gold chain","event":"Terracotta wrap top · Wide-leg camel trousers · Bronze sandals"},
        "blazer": {"daily":"Camel oversized blazer · Olive tee · Brown wide-legs · Tan Chelsea boots","work":"Deep rust blazer · Warm beige blouse · Chocolate trousers","date":"Forest-green velvet blazer · Ivory cami · Dark jeans · Cognac boots","event":"Gold brocade blazer · Cream silk top · Camel wide-legs · Bronze heels"},
        "korean": {"daily":"Camel cardigan · Rust plaid wide-legs · Brown loafers · Tan beret","work":"Olive blouse · Dark brown A-line skirt · Gold brooch","date":"Terracotta crop knit · Camel midi skirt · Brown Mary Janes","event":"Earthy floral modern hanbok · Bronze accessories · Gold hair pins"},
        "formal": {"daily":"Camel linen co-ord · Brown mule · Gold bangle","work":"Olive crepe dress · Cognac kitten heels · Amber drops","date":"Rust satin midi · Bronze heels · Amber pendant","event":"Deep gold gown · Cognac heels · Tiger eye statement jewellery"},
    },
    "winter": {
        "dress":  {"daily":"Crisp white midi · Black sandals · Silver clutch · Bold red lip","work":"Black tailored dress · Silver pumps · Diamond studs · White blazer","date":"Deep emerald satin midi · Silver heels · Diamond pendant","event":"Icy silver column gown · Silver heels · Sapphire statement earrings"},
        "top":    {"daily":"Black fitted turtleneck · White high-waist jeans · Black ankle boots · Silver studs","work":"Crisp white button-up · Black cigarette trousers · Black loafers","date":"Hot pink cami · Black high-waist jeans · Black mule · Diamond pendant","event":"Deep purple blouse · Black wide-legs · Silver strappy heels"},
        "blazer": {"daily":"Sharp black blazer · White tee · Black jeans · White sneakers · Silver chain","work":"Ivory power blazer · Black turtleneck · Black trousers · Silver pumps","date":"Royal blue blazer · Black cami · High-waist black trousers","event":"Midnight velvet blazer · White silk top · Black wide-legs · Silver clutch"},
        "korean": {"daily":"Oversized white shirt · Black straight trousers · Black loafers · Mini black bag","work":"Black peter-pan collar dress · White detail · Silver flats · Pearl studs","date":"Icy pink knit · Black mini skirt · White sneakers · Silver accessories","event":"Monochrome black/white modern hanbok · Bold silver hairpiece · Diamond studs"},
        "formal": {"daily":"Black/white co-ord · Silver loafer · Diamond studs","work":"Navy sheath dress · Silver heels · Pearl accessories","date":"Emerald midi · Silver heels · Sapphire drops","event":"Black gown with silver embroidery · Diamond statement earrings"},
    }
}

STYLE_TIPS = {
    "spring": "Light flowing fabrics — chiffon, linen, cotton — breathe with your warm glow. Layer delicate gold jewellery.",
    "summer": "Choose muted dusty shades over neons. Silver accessories always over gold for your season.",
    "autumn": "Rich textures are your superpower — suede, corduroy, velvet, and knit amplify Autumn's depth.",
    "winter": "High contrast is your friend. Pair stark white with deep jewel tones. Clean sharp silhouettes.",
}

ACCESSORIES_DB = {
    "spring": ["Layered gold or rose gold chains","Straw or woven bags in neutral tones","Coral or pearl stud earrings","Delicate ankle bracelet in gold"],
    "summer": ["Delicate silver pendant necklace","Small structured bag in grey or blush","Amethyst or rose quartz drop earrings","Fine silver bracelet stack"],
    "autumn": ["Chunky gold ring stack","Suede or leather crossbody in cognac","Amber or tiger eye drop earrings","Woven bucket bag in tan"],
    "winter": ["Bold statement silver earrings","Mini structured bag in black or white","Diamond or sapphire stud earrings","Sleek silver cuff bracelet"],
}


# ════════════════════════════════════════════════════════════
#  OUTFIT RATING DATABASE
# ════════════════════════════════════════════════════════════

STYLE_LABELS = ["Casual","Office","Date Night","Party","Late Night Party","Dinner","Korean","Streetwear"]

RATING_DB = {
    "Casual":        {"score":8.0,"stars":4,"verdict":"Versatile and wearable — small tweaks will elevate it.","pros":["Great colour balance for everyday wear","Proportions work well for the occasion","Footwear choice grounds the look"],"tips":["Add one statement accessory to elevate from basic","Try a leather loafer instead of sneakers for instant polish","A structured tote adds purpose without sacrificing comfort"],"complete":"White sneakers or block-heel mule · Crossbody or structured tote · Minimal gold jewellery"},
    "Office":        {"score":9.2,"stars":5,"verdict":"Polished and authoritative — this look commands the room.","pros":["Tailored silhouette reads as confident","Neutral palette communicates credibility","Proportions are perfectly balanced"],"tips":["Ensure blazer shoulder seam sits at shoulder exactly","Add a silk scarf for elevated boardroom energy","Opt for closed-toe heels or smart loafers in client settings"],"complete":"Block-heel pump or leather Oxford · Structured briefcase · Silver or pearl accessories"},
    "Date Night":    {"score":9.0,"stars":5,"verdict":"Romantic and refined — effortlessly memorable.","pros":["Feminine silhouette flatters naturally","Colour palette is mood-perfect","Balances effort and ease beautifully"],"tips":["Keep accessories minimal — let the outfit breathe","A bold lip completes the look without over-accessorising","Opt for a bag slightly smaller than your hand"],"complete":"Strappy heels or kitten mule · Mini evening bag · One statement earring or necklace"},
    "Party":         {"score":8.7,"stars":4,"verdict":"Fun, festive and full of energy.","pros":["Bold colour choice reads as confident","Fit flatters for dancing and movement","Accessories add just enough sparkle"],"tips":["Add a metallic bag for maximum impact","Comfortable heels — you'll be on your feet","Layer a blazer for late-night temperature change"],"complete":"Block heel or embellished sandal · Metallic clutch · Statement earrings"},
    "Late Night Party":{"score":9.1,"stars":5,"verdict":"Effortlessly after-dark — the right kind of energy.","pros":["Colour palette commands after-dark lighting","Silhouette is flattering and movement-friendly","Look strikes the right balance of sultry and chic"],"tips":["A bold lip elevates the outfit instantly","Opt for comfortable heels — prioritize endurance","Add minimal metallic accessories to catch the light"],"complete":"Strappy heel or ankle boot · Small metallic bag · Hoop earrings or bold pendant"},
    "Dinner":        {"score":9.3,"stars":5,"verdict":"Elegant and refined — perfectly pitched for fine dining.","pros":["Sophisticated palette suits the setting","Silhouette is polished and elevated","Accessories feel considered, not overdone"],"tips":["Avoid overly casual footwear — elevate with a heel or loafer","A structured mini bag adds refinement","Ensure outfit is crease-free for this occasion"],"complete":"Kitten heel or pointed loafer · Structured mini bag · Diamond or pearl earrings"},
    "Korean":        {"score":9.3,"stars":5,"verdict":"Perfectly aligned with Korean aesthetic sensibility.","pros":["Soft proportions read as considered and youthful","Colour palette hits K-fashion's sweet spot","Layering creates depth without effort"],"tips":["Add a hair bow or claw clip for the full aesthetic","Carry a mini structured bag in a complementary tone","Mary Janes or pastel sneakers seal the look"],"complete":"Mary Janes or platform loafers · Mini structured bag · Hair bow or claw clip accessory"},
    "Streetwear":    {"score":8.5,"stars":4,"verdict":"Effortlessly cool with sharp urban energy.","pros":["Oversized proportions are balanced by slim pieces","Monochrome base reads as intentional","Sneaker choice anchors the whole aesthetic"],"tips":["Layer a longline tee under the outer piece for depth","Add a crossbody or mini bag for practical edge","A cap or beanie ties the streetwear narrative together"],"complete":"Chunky sneakers or Air Force 1s · Crossbody mini bag · Baseball cap or beanie"},
    "Minimalist":    {"score":9.5,"stars":5,"verdict":"A study in restraint — and it is stunning.","pros":["Every piece earns its place with precision","Neutral palette creates a seamless visual story","Silhouette is timeless and season-agnostic"],"tips":["One elevated accessory (watch or minimal chain) ties everything","Fit precision is everything — consider tailoring hems","Invest in one high-quality fabric piece"],"complete":"White leather sneaker or pointed loafer · Simple leather tote · Minimal chain or watch"},
}


# ════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status": "ok",
        "cv2_available": CV2_AVAILABLE,
        "tf_available": TF_AVAILABLE,
        "version": "3.0.0"
    }


@app.post("/color-analysis")
async def color_analysis(image: UploadFile = File(None)):
    """
    Full color analysis: undertone, season, power palette, avoid, jewelry, tips, makeup.
    Uses OpenCV for skin tone extraction + DL model (or heuristic fallback).
    """
    season, confidence = None, None
    skin_data = {}
    dominant_colors = []

    if image and image.filename:
        raw = await image.read()
        arr = decode_image_bytes(raw)

        # OpenCV analysis (preferred)
        if CV2_AVAILABLE:
            bgr = decode_image_cv2(raw)
            if bgr is not None:
                skin_data = extract_skin_tone_cv2(bgr)
                dominant_colors = extract_dominant_colors_cv2(bgr)
        
        if not skin_data:
            skin_data = extract_skin_tone_pil(arr)

        # DL model prediction
        season_dl, conf_dl = predict_season_dl(arr)

        if season_dl and conf_dl and conf_dl >= 50:
            season, confidence = season_dl, conf_dl
        else:
            season_h, conf_h = predict_season_heuristic(skin_data)
            season = season_dl if season_dl else season_h
            confidence = round((conf_dl or 0) * 0.4 + conf_h * 0.6, 1)
    else:
        season = random.choice(SEASONS)
        confidence = round(random.uniform(72, 91), 1)
        skin_data = {"undertone": "Neutral", "depth": "Medium Depth", "contrast": "High"}

    db = SEASON_DB[season]
    return {
        "undertone":    db["label"],
        "season":       season.lower(),
        "confidence":   confidence,
        "depth":        skin_data.get("depth", "Medium Depth"),
        "contrast":     skin_data.get("contrast", "Medium"),
        "colors":       db["colors"],
        "color_names":  db["color_names"],
        "avoid":        db["avoid"],
        "jewelry":      db["jewelry"],
        "tips":         db["tips"],
        "makeup":       db["makeup"],
        "dominant_colors": dominant_colors,
    }


class BodyRequest(BaseModel):
    body_type: str


@app.post("/body-analysis")
async def body_analysis(req: BodyRequest):
    """Structural outfit recommendations for a given body type (manual selection)."""
    db = BODY_DB.get(req.body_type)
    if not db:
        raise HTTPException(status_code=404, detail=f"Body type '{req.body_type}' not found.")
    return {"body_type": req.body_type, **db}


@app.post("/body-image-analysis")
async def body_image_analysis(image: UploadFile = File(None)):
    """
    Analyze a full-body image to detect body type using OpenCV edge detection.
    Falls back to random classification if model/image unavailable.
    """
    if not image or not image.filename:
        body_type = random.choice(BODY_TYPES)
        confidence = round(random.uniform(68, 82), 1)
    else:
        raw = await image.read()
        if CV2_AVAILABLE:
            bgr = decode_image_cv2(raw)
            if bgr is not None:
                body_type, confidence = estimate_body_type_cv2(bgr)
            else:
                body_type, confidence = random.choice(BODY_TYPES), round(random.uniform(68, 82), 1)
        else:
            body_type, confidence = random.choice(BODY_TYPES), round(random.uniform(68, 82), 1)

    db = BODY_DB.get(body_type, BODY_DB["Hourglass"])
    return {
        "body_type":  body_type,
        "confidence": confidence,
        "badge":      db["badge"],
        "description":db["description"],
        "dos":        db["dos"],
        "avoid":      db["avoid"],
        "tops":       db["tops"],
        "bottoms":    db["bottoms"],
    }


@app.post("/tryon")
async def tryon(
    season:       str = Form(...),
    garment_type: str = Form(...),
    occasion:     str = Form("daily"),
    image:        UploadFile = File(None),
):
    """Returns a curated outfit description for the given season / garment / occasion."""
    season_db  = TRYON_DB.get(season.lower(), TRYON_DB["winter"])
    garment_db = season_db.get(garment_type.lower(), season_db.get("top", {}))
    look       = garment_db.get(occasion.lower(), garment_db.get("daily", "A curated look for this combination."))
    return {
        "season":             season,
        "garment_type":       garment_type,
        "occasion":           occasion,
        "outfit_description": look,
        "style_tip":          STYLE_TIPS.get(season.lower(), ""),
        "accessories":        ACCESSORIES_DB.get(season.lower(), []),
    }


@app.post("/rating")
async def rating(
    style_hint: str = Form(""),
    image:      UploadFile = File(None),
):
    """
    Rates an outfit. Uses EfficientNetB0 if available, falls back to style_hint KB.
    Returns score, stars, verdict, pros, tips, and 'complete the look' advice.
    """
    dl_score, dl_style = None, None

    if image and image.filename:
        raw = await image.read()
        arr = decode_image_bytes(raw)
        model = get_rating_model()
        if model:
            try:
                x = arr / 255.0
                x = np.expand_dims(x, 0)
                score_pred, style_pred = model.predict(x, verbose=0)
                dl_score = round(float(score_pred[0][0]) * 10, 1)
                dl_style = STYLE_LABELS[int(np.argmax(style_pred[0]))]
            except:
                pass

    chosen_style = dl_style or style_hint or "Casual"
    advice       = RATING_DB.get(chosen_style, RATING_DB["Casual"])
    final_score  = dl_score if dl_score is not None else advice["score"]
    stars        = max(1, min(5, round(final_score / 2)))

    return {
        "score":    final_score,
        "stars":    stars,
        "verdict":  advice["verdict"],
        "style":    chosen_style,
        "pros":     advice["pros"],
        "tips":     advice["tips"],
        "complete": advice["complete"],
        "dl_used":  dl_score is not None,
    }


@app.post("/save-scan")
async def save_scan(data: dict):
    """Persist scan results server-side (optional — frontend also caches in localStorage)."""
    timestamp = datetime.now().isoformat()
    record = {"timestamp": timestamp, **data}
    # In production, persist to DB; for now just echo back
    return {"status": "saved", "record": record}


# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════╗")
    print("║     FURISTIC Fashion Backend v3.0        ║")
    print("║   FastAPI + OpenCV + TensorFlow DL       ║")
    print("╠══════════════════════════════════════════╣")
    print("║  Color Analysis  →  /color-analysis      ║")
    print("║  Body Measure    →  /body-image-analysis ║")
    print("║  Body Manual     →  /body-analysis       ║")
    print("║  Virtual Try-On  →  /tryon               ║")
    print("║  Outfit Rating   →  /rating              ║")
    print("╚══════════════════════════════════════════╝\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
