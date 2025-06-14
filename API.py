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

# HUGGING FACE CONFIGURATION - THAY ĐỔI THÔNG TIN NÀY
HUGGINGFACE_REPO = "trandangduc0/appnam"  # Thay bằng repo của bạn
HF_BASE_URL = f"https://huggingface.co/{HUGGINGFACE_REPO}/resolve/main"

# Image preprocessing
mean, std, im_size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224
TRANSFORMS = transforms.Compose([
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Global variables
CLASS_NAMES = {}
POISONOUS_MAPPING = {}

def check_file_valid(file_path):
    """Kiểm tra file có hợp lệ không"""
    if not os.path.exists(file_path):
        return False
    
    try:
        # Kiểm tra kích thước file
        file_size = os.path.getsize(file_path)
        logger.info(f"📏 File size: {file_size/1024/1024:.1f}MB")
        
        if file_path.endswith('.pth'):
            if file_size < 1000:  # File quá nhỏ
                logger.error("❌ Model file too small")
                return False
            
            # Kiểm tra header file
            with open(file_path, 'rb') as f:
                header = f.read(100)
                # Nếu bắt đầu bằng < thì là HTML
                if header.startswith(b'<'):
                    logger.error("❌ File is HTML, not a model")
                    return False
                
                # PyTorch files thường bắt đầu với PK (zip format) hoặc pickle
                if header.startswith(b'PK') or header.startswith(b'\x80'):
                    logger.info("✅ Valid PyTorch model file")
                    return True
        
        elif file_path.endswith('.json'):
            if file_size < 10:  # JSON quá nhỏ
                return False
            
            # Thử parse JSON
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
                        
                        # Progress log cho file lớn
                        if total_size > 1024*1024 and downloaded_size % (1024*1024) == 0:
                            progress = (downloaded_size / total_size) * 100 if total_size > 0 else 0
                            logger.info(f"📥 Downloaded: {progress:.1f}%")
            
            logger.info(f"💾 Download completed: {downloaded_size/1024/1024:.1f}MB")
            
            # Verify file
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
            logger.error(f"🔍 Please check if the file exists at: {url}")
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
        # Kiểm tra file đã tồn tại và hợp lệ chưa
        if os.path.exists(local_path) and check_file_valid(local_path):
            logger.info(f"✅ {local_path} already exists and valid")
            success_count += 1
            continue
        
        # Download file
        if download_file_from_huggingface(remote_name, local_path):
            success_count += 1
        else:
            logger.warning(f"⚠️ Failed to download {remote_name}")
    
    logger.info(f"📊 Downloaded {success_count}/{len(files_to_download)} files successfully")
    return success_count >= 1  # Ít nhất phải có model file

def load_class_names():
    """Load class names"""
    global CLASS_NAMES
    
    if os.path.exists(CLASS_NAMES_FILE) and check_file_valid(CLASS_NAMES_FILE):
        try:
            with open(CLASS_NAMES_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                CLASS_NAMES = {int(k) if str(k).isdigit() else k: v for k, v in data.items()}
            logger.info(f"✅ Loaded {len(CLASS_NAMES)} classes from file")
            return
        except Exception as e:
            logger.error(f"❌ Error loading class names: {e}")
    
    # Fallback classes
    CLASS_NAMES = {
        0: "Agaricus bisporus",
        1: "Amanita muscaria", 
        2: "Boletus edulis",
        3: "Cantharellus cibarius",
        4: "Pleurotus ostreatus",
        5: "Shiitake",
        6: "Ganoderma lucidum",
        7: "Amanita phalloides",
        8: "Lactarius deliciosus",
        9: "Morchella esculenta"
    }
    logger.info(f"📝 Using fallback classes: {len(CLASS_NAMES)}")

def load_poisonous_mapping():
    """Load poisonous mapping"""
    global POISONOUS_MAPPING
    
    if os.path.exists(POISONOUS_MAPPING_FILE) and check_file_valid(POISONOUS_MAPPING_FILE):
        try:
            with open(POISONOUS_MAPPING_FILE, 'r', encoding='utf-8') as f:
                POISONOUS_MAPPING = json.load(f)
            logger.info(f"✅ Loaded poisonous mapping from file")
            return
        except Exception as e:
            logger.error(f"❌ Error loading poisonous mapping: {e}")
    
    # Fallback mapping
    POISONOUS_MAPPING = {
        "Agaricus bisporus": False,
        "Amanita muscaria": True,
        "Boletus edulis": False,
        "Cantharellus cibarius": False,
        "Pleurotus ostreatus": False,
        "Shiitake": False,
        "Ganoderma lucidum": False,
        "Amanita phalloides": True,
        "Lactarius deliciosus": False,
        "Morchella esculenta": False
    }
    logger.info(f"📝 Using fallback poisonous mapping")

def load_model():
    """Load model với Hugging Face integration"""
    try:
        logger.info("🚀 Starting model loading process...")
        
        # Download files từ Hugging Face nếu cần
        if not download_all_files():
            logger.error("❌ Critical files download failed")
        
        # Load class names và poisonous mapping
        load_class_names()
        load_poisonous_mapping()
        
        num_classes = len(CLASS_NAMES)
        logger.info(f"🔧 Creating model with {num_classes} classes")
        
        # Tạo model architecture
        model = timm.create_model("rexnet_150", pretrained=False, num_classes=num_classes)
        
        # Kiểm tra và load model weights
        if not os.path.exists(MODEL_PATH) or not check_file_valid(MODEL_PATH):
            logger.warning("⚠️ Model file not available, using pretrained model")
            model = timm.create_model("rexnet_150", pretrained=True, num_classes=num_classes)
            model.to(DEVICE)
            model.eval()
            return model
        
        # Load trained weights
        try:
            logger.info(f"🔄 Loading weights from {MODEL_PATH}")
            
            # Load checkpoint với error handling
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            
            # Xử lý format khác nhau
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
            
            # Load weights
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
        "model_file_exists": os.path.exists(MODEL_PATH),
        "model_file_valid": check_file_valid(MODEL_PATH) if os.path.exists(MODEL_PATH) else False,
        "huggingface_repo": HUGGINGFACE_REPO,
        "files_status": {
            "model": check_file_valid(MODEL_PATH) if os.path.exists(MODEL_PATH) else False,
            "classes": check_file_valid(CLASS_NAMES_FILE) if os.path.exists(CLASS_NAMES_FILE) else False,
            "poisonous_mapping": check_file_valid(POISONOUS_MAPPING_FILE) if os.path.exists(POISONOUS_MAPPING_FILE) else False
        }
    })

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint"""
    return jsonify({
        "message": "🍄 Mushroom API working!",
        "model_status": "Loaded" if model else "Not loaded",
        "classes": len(CLASS_NAMES),
        "huggingface_repo": HUGGINGFACE_REPO
    })

@app.route('/redownload', methods=['POST'])
def redownload_files():
    """Force redownload all files từ Hugging Face"""
    try:
        # Xóa files cũ
        files_to_remove = [MODEL_PATH, CLASS_NAMES_FILE, POISONOUS_MAPPING_FILE]
        for file_path in files_to_remove:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"🗑️ Removed {file_path}")
        
        # Download lại
        success = download_all_files()
        
        if success:
            # Reload model
            global model
            model = load_model()
            
            return jsonify({
                "status": "success",
                "message": "Files redownloaded and model reloaded",
                "model_loaded": model is not None
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to redownload files"
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Redownload error: {e}")
        return jsonify({"error": f"Redownload failed: {str(e)}"}), 500

if __name__ == '__main__':
    print("🍄" + "="*50)
    print("🍄 MUSHROOM RECOGNITION API")
    print("🍄 WITH HUGGING FACE INTEGRATION")
    print("🍄" + "="*50)
    
    init_database()
    
    print(f"📱 Device: {DEVICE}")
    print(f"🤗 Hugging Face Repo: {HUGGINGFACE_REPO}")
    print(f"🔧 Model: {'✅ Loaded' if model else '❌ Failed'}")
    print(f"🗂️ Classes: {len(CLASS_NAMES)}")
    print(f"📁 Model file exists: {os.path.exists(MODEL_PATH)}")
    print(f"📄 Model file valid: {check_file_valid(MODEL_PATH) if os.path.exists(MODEL_PATH) else 'N/A'}")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
