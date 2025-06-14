from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import base64
import io
import os
import json
import google.generativeai as genai
from flask_cors import CORS
import timm
from datetime import datetime
import sqlite3
import uuid
import logging
import requests

# Configure logging  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
CORS(app, origins="*")

# Configure Gemini API
API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyCeGZiWJ6_Ynysbwt5-32VRStPTGs1Iwyw')
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Model configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'mushrooms_multiclass_best_model.pth'
CLASS_NAMES_FILE = 'mushroom_classes.json'
POISONOUS_MAPPING_FILE = 'poisonous_mapping.json'

# HUGGING FACE CONFIGURATION
HUGGINGFACE_REPO = "trandangduc0/appnam"
HF_BASE_URL = f"https://huggingface.co/{HUGGINGFACE_REPO}/resolve/main"

# Image preprocessing
mean, std, im_size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224
TRANSFORMS = transforms.Compose([
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# ================================================
# EMBEDDED DATA - KHÔNG CẦN FILES RIÊNG BIỆT
# ================================================

# Class names mapping (47 classes)
EMBEDDED_CLASS_NAMES = {
    0: "Agaricus bisporus",
    1: "Agaricus subrufescens",
    2: "Amanita bisporigera",
    3: "Amanita muscaria",
    4: "Amanita ocreata",
    5: "Amanita phalloides",
    6: "Amanita smithiana",
    7: "Amanita verna",
    8: "Amanita virosa",
    9: "Auricularia auricula-judae",
    10: "Boletus edulis",
    11: "Cantharellus cibarius",
    12: "Clitocybe dealbata",
    13: "Conocybe filaris",
    14: "Coprinus comatus",
    15: "Cordyceps sinensis",
    16: "Cortinarius rubellus",
    17: "Entoloma sinuatum",
    18: "Flammulina velutipes",
    19: "Galerina marginata",
    20: "Ganoderma lucidum",
    21: "Grifola frondosa",
    22: "Gyromitra esculenta",
    23: "Hericium erinaceus",
    24: "Hydnum repandum",
    25: "Hypholoma fasciculare",
    26: "Inocybe erubescens",
    27: "Lentinula edodes",
    28: "Lepiota brunneoincarnata",
    29: "Macrolepiota procera",
    30: "Morchella esculenta",
    31: "Omphalotus olearius",
    32: "Paxillus involutus",
    33: "Pholiota nameko",
    34: "Pleurotus citrinopileatus",
    35: "Pleurotus eryngii",
    36: "Pleurotus ostreatus",
    37: "Psilocybe semilanceata",
    38: "Rhodophyllus rhodopolius",
    39: "Russula emetica",
    40: "Russula virescens",
    41: "Scleroderma citrinum",
    42: "Suillus luteus",
    43: "Tremella fuciformis",
    44: "Tricholoma matsutake",
    45: "Truffles",
    46: "Tuber melanosporum"
}

# Poisonous mapping
EMBEDDED_POISONOUS_MAPPING = {
    "Agaricus bisporus": False,
    "Agaricus subrufescens": False,
    "Amanita bisporigera": True,
    "Amanita muscaria": True,
    "Amanita ocreata": True,
    "Amanita phalloides": True,
    "Amanita smithiana": True,
    "Amanita verna": True,
    "Amanita virosa": True,
    "Auricularia auricula-judae": False,
    "Boletus edulis": False,
    "Cantharellus cibarius": False,
    "Clitocybe dealbata": True,
    "Conocybe filaris": True,
    "Coprinus comatus": False,
    "Cordyceps sinensis": False,
    "Cortinarius rubellus": True,
    "Entoloma sinuatum": True,
    "Flammulina velutipes": False,
    "Galerina marginata": True,
    "Ganoderma lucidum": False,
    "Grifola frondosa": False,
    "Gyromitra esculenta": True,
    "Hericium erinaceus": False,
    "Hydnum repandum": False,
    "Hypholoma fasciculare": True,
    "Inocybe erubescens": True,
    "Lentinula edodes": False,
    "Lepiota brunneoincarnata": True,
    "Macrolepiota procera": False,
    "Morchella esculenta": False,
    "Omphalotus olearius": True,
    "Paxillus involutus": True,
    "Pholiota nameko": False,
    "Pleurotus citrinopileatus": False,
    "Pleurotus eryngii": False,
    "Pleurotus ostreatus": False,
    "Psilocybe semilanceata": True,
    "Rhodophyllus rhodopolius": True,
    "Russula emetica": True,
    "Russula virescens": False,
    "Scleroderma citrinum": True,
    "Suillus luteus": False,
    "Tremella fuciformis": False,
    "Tricholoma matsutake": False,
    "Truffles": False,
    "Tuber melanosporum": False
}

# Global variables
CLASS_NAMES = {}
POISONOUS_MAPPING = {}

def check_file_valid(file_path):
    """Kiểm tra file có hợp lệ không"""
    if not os.path.exists(file_path):
        return False
    
    try:
        file_size = os.path.getsize(file_path)
        logger.info(f"📏 File size: {file_size/1024/1024:.1f}MB")
        
        if file_path.endswith('.pth'):
            if file_size < 1000:
                logger.error("❌ Model file too small")
                return False
            
            with open(file_path, 'rb') as f:
                header = f.read(100)
                if header.startswith(b'<'):
                    logger.error("❌ File is HTML, not a model")
                    return False
                
                if header.startswith(b'PK') or header.startswith(b'\x80'):
                    logger.info("✅ Valid PyTorch model file")
                    return True
        
        elif file_path.endswith('.json'):
            if file_size < 10:
                return False
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json.load(f)
                logger.info("✅ Valid JSON file")
                return True
            except json.JSONDecodeError:
                logger.error("❌ Invalid JSON file")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error checking file {file_path}: {e}")
        return False

def download_file_from_huggingface(filename, local_path):
    """Download file từ Hugging Face"""
    try:
        url = f"{HF_BASE_URL}/{filename}"
        logger.info(f"🔄 Downloading {filename} from Hugging Face...")
        logger.info(f"📡 URL: {url}")
        
        response = requests.get(url, stream=True)
        
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if total_size > 1024*1024 and downloaded_size % (1024*1024) == 0:
                            progress = (downloaded_size / total_size) * 100 if total_size > 0 else 0
                            logger.info(f"📥 Downloaded: {progress:.1f}%")
            
            logger.info(f"💾 Download completed: {downloaded_size/1024/1024:.1f}MB")
            
            if check_file_valid(local_path):
                logger.info(f"✅ {filename} downloaded and verified successfully!")
                return True
            else:
                logger.error(f"❌ Downloaded {filename} is invalid")
                if os.path.exists(local_path):
                    os.remove(local_path)
                return False
                
        elif response.status_code == 404:
            logger.error(f"❌ File {filename} not found on Hugging Face (404)")
            return False
        else:
            logger.error(f"❌ HTTP Error {response.status_code} downloading {filename}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Download failed for {filename}: {e}")
        return False

def download_all_files():
    """Download tất cả files cần thiết từ Hugging Face"""
    files_to_download = [
        (MODEL_PATH, MODEL_PATH),
        (CLASS_NAMES_FILE, CLASS_NAMES_FILE),
        (POISONOUS_MAPPING_FILE, POISONOUS_MAPPING_FILE)
    ]
    
    success_count = 0
    
    for remote_name, local_path in files_to_download:
        if os.path.exists(local_path) and check_file_valid(local_path):
            logger.info(f"✅ {local_path} already exists and valid")
            success_count += 1
            continue
        
        if download_file_from_huggingface(remote_name, local_path):
            success_count += 1
        else:
            logger.warning(f"⚠️ Failed to download {remote_name}")
    
    logger.info(f"📊 Downloaded {success_count}/{len(files_to_download)} files successfully")
    return success_count >= 1

def load_class_names():
    """Load class names - ưu tiên embedded data"""
    global CLASS_NAMES
    
    logger.info("🔄 Loading class names...")
    
    # Sử dụng embedded data trước
    CLASS_NAMES = EMBEDDED_CLASS_NAMES.copy()
    logger.info(f"✅ Using embedded class names: {len(CLASS_NAMES)} classes")
    
    # Thử load từ file nếu có (để override nếu cần)
    if os.path.exists(CLASS_NAMES_FILE) and check_file_valid(CLASS_NAMES_FILE):
        try:
            with open(CLASS_NAMES_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Chuyển đổi format nếu cần (name: index -> index: name)
                if data and isinstance(list(data.values())[0], int):
                    # Format: {"Agaricus bisporus": 0} -> {0: "Agaricus bisporus"}
                    file_classes = {v: k for k, v in data.items()}
                    CLASS_NAMES.update(file_classes)
                    logger.info(f"✅ Updated with {len(file_classes)} classes from file")
                else:
                    # Format: {0: "Agaricus bisporus"}
                    file_classes = {int(k) if str(k).isdigit() else k: v for k, v in data.items()}
                    CLASS_NAMES.update(file_classes)
                    logger.info(f"✅ Updated with {len(file_classes)} classes from file")
        except Exception as e:
            logger.error(f"❌ Error loading class names from file: {e}")
    
    logger.info(f"📝 Final class count: {len(CLASS_NAMES)}")

def load_poisonous_mapping():
    """Load poisonous mapping - ưu tiên embedded data"""
    global POISONOUS_MAPPING
    
    logger.info("🔄 Loading poisonous mapping...")
    
    # Sử dụng embedded data
    POISONOUS_MAPPING = EMBEDDED_POISONOUS_MAPPING.copy()
    logger.info(f"✅ Using embedded poisonous mapping: {len(POISONOUS_MAPPING)} entries")
    
    # Thử load từ file nếu có (để override nếu cần)
    if os.path.exists(POISONOUS_MAPPING_FILE) and check_file_valid(POISONOUS_MAPPING_FILE):
        try:
            with open(POISONOUS_MAPPING_FILE, 'r', encoding='utf-8') as f:
                file_mapping = json.load(f)
                POISONOUS_MAPPING.update(file_mapping)
                logger.info(f"✅ Updated with mapping from file")
        except Exception as e:
            logger.error(f"❌ Error loading poisonous mapping from file: {e}")
    
    logger.info(f"📝 Final mapping count: {len(POISONOUS_MAPPING)}")

def load_model():
    """Load model với embedded data"""
    try:
        logger.info("🚀 Starting model loading process...")
        
        # Load embedded data trước
        load_class_names()
        load_poisonous_mapping()
        
        # Download model file từ Hugging Face nếu cần
        if not os.path.exists(MODEL_PATH) or not check_file_valid(MODEL_PATH):
            logger.info("🔄 Attempting to download model file...")
            download_file_from_huggingface(MODEL_PATH, MODEL_PATH)
        
        num_classes = len(CLASS_NAMES)
        logger.info(f"🔧 Creating model with {num_classes} classes")
        
        # Tạo model architecture
        model = timm.create_model("rexnet_150", pretrained=False, num_classes=num_classes)
        
        # Load trained weights nếu có
        if os.path.exists(MODEL_PATH) and check_file_valid(MODEL_PATH):
            try:
                logger.info(f"🔄 Loading weights from {MODEL_PATH}")
                
                checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                        logger.info("📦 Found model_state_dict in checkpoint")
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                        logger.info("📦 Found state_dict in checkpoint")
                    else:
                        state_dict = checkpoint
                        logger.info("📦 Using checkpoint as state_dict")
                else:
                    state_dict = checkpoint
                    logger.info("📦 Using checkpoint directly")
                
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    logger.warning(f"⚠️ Missing keys: {len(missing_keys)} keys")
                if unexpected_keys:
                    logger.warning(f"⚠️ Unexpected keys: {len(unexpected_keys)} keys")
                
                logger.info("✅ Loaded trained weights successfully")
                
            except Exception as e:
                logger.error(f"❌ Error loading weights: {e}")
                logger.info("📝 Using pretrained model instead")
                model = timm.create_model("rexnet_150", pretrained=True, num_classes=num_classes)
        else:
            logger.warning("⚠️ Model file not available, using pretrained model")
            model = timm.create_model("rexnet_150", pretrained=True, num_classes=num_classes)
        
        model.to(DEVICE)
        model.eval()
        logger.info(f"✅ Model ready on {DEVICE}")
        return model
        
    except Exception as e:
        logger.error(f"❌ Model loading failed completely: {e}")
        return None

def init_database():
    """Initialize database"""
    try:
        conn = sqlite3.connect('mushroom_history.db')
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS predictions
                     (id TEXT PRIMARY KEY,
                      timestamp TEXT,
                      predicted_class TEXT,
                      confidence REAL,
                      is_poisonous BOOLEAN,
                      image_data TEXT,
                      gemini_info TEXT)''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.error(f"❌ Database error: {e}")

# Load model on startup
model = load_model()

def preprocess_image(image_data):
    """Preprocess image"""
    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = TRANSFORMS(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        logger.error(f"❌ Image preprocessing error: {e}")
        return None

def predict_mushroom(image_tensor):
    """Predict mushroom"""
    try:
        if model is None:
            return None, 0.0
            
        with torch.no_grad():
            image_tensor = image_tensor.to(DEVICE)
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_index = predicted.item()
            confidence_score = confidence.item()
            predicted_class = CLASS_NAMES.get(predicted_index, f"Unknown_Class_{predicted_index}")
            
            return predicted_class, confidence_score
            
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return None, 0.0

def is_mushroom_poisonous(mushroom_name):
    """Check if mushroom is poisonous"""
    return POISONOUS_MAPPING.get(mushroom_name, False)

def get_mushroom_info_from_gemini(mushroom_name):
    """Get mushroom info from Gemini"""
    try:
        prompt = f"""
        Cung cấp thông tin về nấm: {mushroom_name}
        
        Trả lời JSON:
        {{
            "name": "Tên tiếng Việt",
            "scientific_name": "Tên khoa học", 
            "edibility": "Ăn được/Độc/Không rõ",
            "toxicity_level": "Mức độ độc",
            "appearance": "Mô tả hình dáng",
            "habitat": "Môi trường sống",
            "warnings": "Cảnh báo"
        }}
        """
        
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"❌ Gemini error: {e}")
        return json.dumps({
            "name": mushroom_name,
            "scientific_name": mushroom_name,
            "edibility": "Không rõ",
            "toxicity_level": "Không rõ", 
            "appearance": "Không có thông tin",
            "habitat": "Không có thông tin",
            "warnings": "Tham khảo chuyên gia"
        }, ensure_ascii=False)

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
            
        data = request.get_json()
        image_data = data.get('image', '')
        
        if not image_data:
            return jsonify({"error": "No image provided"}), 400
        
        # Preprocess
        image_tensor = preprocess_image(image_data)
        if image_tensor is None:
            return jsonify({"error": "Image processing failed"}), 400
        
        # Predict
        predicted_class, confidence = predict_mushroom(image_tensor)
        if predicted_class is None:
            return jsonify({"error": "Prediction failed"}), 500
        
        # Check toxicity
        is_poisonous = is_mushroom_poisonous(predicted_class)
        
        # Get Gemini info
        gemini_info = get_mushroom_info_from_gemini(predicted_class)
        
        result = {
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "is_poisonous": is_poisonous,
            "gemini_info": gemini_info,
            "status": "success"
        }
        
        logger.info(f"✅ Prediction: {predicted_class} ({confidence*100:.1f}%)")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Prediction endpoint error: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "num_classes": len(CLASS_NAMES),
        "embedded_classes": len(EMBEDDED_CLASS_NAMES),
        "embedded_mappings": len(EMBEDDED_POISONOUS_MAPPING),
        "model_file_exists": os.path.exists(MODEL_PATH),
        "model_file_valid": check_file_valid(MODEL_PATH) if os.path.exists(MODEL_PATH) else False,
        "huggingface_repo": HUGGINGFACE_REPO,
        "data_source": "embedded"
    })

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint"""
    return jsonify({
        "message": "🍄 Mushroom API working!",
        "model_status": "Loaded" if model else "Not loaded",
        "classes": len(CLASS_NAMES),
        "huggingface_repo": HUGGINGFACE_REPO,
        "data_embedded": True
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get all mushroom classes"""
    return jsonify({
        "classes": CLASS_NAMES,
        "count": len(CLASS_NAMES),
        "poisonous_mapping": POISONOUS_MAPPING
    })

@app.route('/redownload', methods=['POST'])
def redownload_files():
    """Force redownload model file từ Hugging Face"""
    try:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            logger.info(f"🗑️ Removed {MODEL_PATH}")
        
        success = download_file_from_huggingface(MODEL_PATH, MODEL_PATH)
        
        if success:
            global model
            model = load_model()
            
            return jsonify({
                "status": "success",
                "message": "Model file redownloaded and reloaded",
                "model_loaded": model is not None
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to redownload model file"
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Redownload error: {e}")
        return jsonify({"error": f"Redownload failed: {str(e)}"}), 500

if __name__ == '__main__':
    print("🍄" + "="*50)
    print("🍄 MUSHROOM RECOGNITION API")
    print("🍄 WITH EMBEDDED DATA")
    print("🍄" + "="*50)
    
    init_database()
    
    print(f"📱 Device: {DEVICE}")
    print(f"🤗 Hugging Face Repo: {HUGGINGFACE_REPO}")
    print(f"🔧 Model: {'✅ Loaded' if model else '❌ Failed'}")
    print(f"🗂️ Classes: {len(CLASS_NAMES)} (embedded: {len(EMBEDDED_CLASS_NAMES)})")
    print(f"🧪 Poisonous mappings: {len(POISONOUS_MAPPING)}")
    print(f"📁 Model file exists: {os.path.exists(MODEL_PATH)}")
    print(f"📄 Using embedded data: ✅")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
