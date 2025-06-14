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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
CORS(app, origins="*")  # Allow all origins for development

# Configure Gemini API - SỬA: Dùng environment variable
API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyCeGZiWJ6_Ynysbwt5-32VRStPTGs1Iwyw')
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Model configuration - CHÍNH XÁC như trong training code
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'mushrooms_multiclass_best_model.pth'  # Đúng tên file từ training
CLASS_NAMES_FILE = 'mushroom_classes.json'  # File được tạo trong training
POISONOUS_MAPPING_FILE = 'poisonous_mapping.json'  # File mapping độc/không độc

# Image preprocessing - CHÍNH XÁC như trong training
mean, std, im_size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224
TRANSFORMS = transforms.Compose([
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Global variables
CLASS_NAMES = {}
POISONOUS_MAPPING = {}

# Initialize database
def init_database():
    """Khởi tạo database SQLite"""
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
        
        c.execute('''CREATE TABLE IF NOT EXISTS favorites
                     (id TEXT PRIMARY KEY,
                      mushroom_name TEXT,
                      scientific_name TEXT,
                      description TEXT,
                      added_date TEXT)''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Error initializing database: {e}")

def load_class_names():
    """Load class names từ file được tạo trong training"""
    global CLASS_NAMES
    
    try:
        # Thử load từ file training tạo ra
        if os.path.exists(CLASS_NAMES_FILE):
            with open(CLASS_NAMES_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert string keys to int if needed
                CLASS_NAMES = {int(k) if k.isdigit() else k: v for k, v in data.items()}
            logger.info(f"✅ Loaded {len(CLASS_NAMES)} class names from {CLASS_NAMES_FILE}")
        else:
            logger.warning(f"⚠️ Class names file not found: {CLASS_NAMES_FILE}")
            # Fallback: create demo classes for testing
            CLASS_NAMES = {
                0: "Agaricus bisporus",
                1: "Amanita muscaria", 
                2: "Boletus edulis",
                3: "Cantharellus cibarius",
                4: "Pleurotus ostreatus"
            }
            logger.info(f"📝 Using fallback classes: {len(CLASS_NAMES)} types")
            
    except Exception as e:
        logger.error(f"❌ Error loading class names: {e}")
        CLASS_NAMES = {0: "Unknown mushroom"}
    
    return CLASS_NAMES

def load_poisonous_mapping():
    """Load poisonous mapping từ file được tạo trong training"""
    global POISONOUS_MAPPING
    
    try:
        if os.path.exists(POISONOUS_MAPPING_FILE):
            with open(POISONOUS_MAPPING_FILE, 'r', encoding='utf-8') as f:
                POISONOUS_MAPPING = json.load(f)
            logger.info(f"✅ Loaded poisonous mapping for {len(POISONOUS_MAPPING)} species")
        else:
            logger.warning(f"⚠️ Poisonous mapping file not found: {POISONOUS_MAPPING_FILE}")
            # Fallback: default poisonous species
            default_poisonous = {
                'amanita bisporigera', 'amanita muscaria', 'amanita ocreata', 'amanita pantherina',
                'amanita phalloides', 'amanita virosa', 'amanita smithiana', 'amanita verna',
                'clitocybe dealbata', 'conocybe filaris', 'cortinarius rubellus', 'entoloma sinuatum',
                'galerina marginata', 'gyromitra esculenta', 'hypholoma fasciculare', 'inocybe erubescens',
                'lepiota brunneoincarnata', 'omphalotus olearius', 'paxillus involutus', 
                'psilocybe semilanceata', 'rhodophyllus rhodopolius', 'russula emetica', 'scleroderma citrinum'
            }
            
            # Create default mapping
            POISONOUS_MAPPING = {}
            for species_name in CLASS_NAMES.values() if isinstance(CLASS_NAMES, dict) else []:
                is_poisonous = any(poison in species_name.lower() for poison in default_poisonous)
                POISONOUS_MAPPING[species_name] = is_poisonous
                
    except Exception as e:
        logger.error(f"❌ Error loading poisonous mapping: {e}")
        POISONOUS_MAPPING = {}

def load_model():
    """Load model AI với architecture CHÍNH XÁC từ training"""
    try:
        # Load class names trước
        load_class_names()
        load_poisonous_mapping()
        
        if not CLASS_NAMES:
            logger.error("❌ No class names loaded - cannot create model")
            return None
            
        num_classes = len(CLASS_NAMES)
        logger.info(f"🔧 Creating model with {num_classes} classes")
        
        # Tạo model với architecture CHÍNH XÁC như training
        model = timm.create_model("rexnet_150", pretrained=False, num_classes=num_classes)
        
        # Kiểm tra file model tồn tại
        if os.path.exists(MODEL_PATH):
            # Load trained weights
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
            logger.info(f"✅ Loaded trained model from {MODEL_PATH}")
        else:
            logger.warning(f"⚠️ Model file not found: {MODEL_PATH}")
            logger.info("📝 Using pretrained model for demo (accuracy will be limited)")
            # Tạo model pretrained để demo
            model = timm.create_model("rexnet_150", pretrained=True, num_classes=num_classes)
        
        model.to(DEVICE)
        model.eval()
        
        logger.info(f"✅ Model loaded successfully on {DEVICE}")
        return model
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return None

# Load model on startup
model = load_model()

def preprocess_image(image_data):
    """Tiền xử lý ảnh cho model - CHÍNH XÁC như training"""
    try:
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply transforms - giống y hệt training
        image_tensor = TRANSFORMS(image).unsqueeze(0)
        logger.info(f"🖼️ Image preprocessed: {image_tensor.shape}")
        return image_tensor
    except Exception as e:
        logger.error(f"❌ Error preprocessing image: {e}")
        return None

def predict_mushroom(image_tensor):
    """Dự đoán loại nấm bằng model AI"""
    try:
        if model is None:
            logger.error("❌ Model not loaded")
            return None, 0.0
            
        with torch.no_grad():
            image_tensor = image_tensor.to(DEVICE)
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_index = predicted.item()
            confidence_score = confidence.item()
            
            # Lấy class name từ index
            predicted_class = CLASS_NAMES.get(predicted_index, f"Unknown_Class_{predicted_index}")
            
            logger.info(f"🔍 Predicted: {predicted_class} (index: {predicted_index}, confidence: {confidence_score:.3f})")
            
            return predicted_class, confidence_score
            
    except Exception as e:
        logger.error(f"❌ Error during prediction: {e}")
        return None, 0.0

def is_mushroom_poisonous(mushroom_name):
    """Xác định nấm có độc không dựa trên mapping từ training"""
    try:
        # Kiểm tra trong mapping từ training
        if mushroom_name in POISONOUS_MAPPING:
            return POISONOUS_MAPPING[mushroom_name]
        
        # Fallback: kiểm tra theo từ khóa
        mushroom_lower = mushroom_name.lower().strip()
        poisonous_keywords = [
            'amanita', 'galerina', 'gyromitra', 'hypholoma', 'omphalotus',
            'paxillus', 'russula emetica', 'scleroderma', 'clitocybe dealbata'
        ]
        
        for poisonous in poisonous_keywords:
            if poisonous in mushroom_lower:
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"❌ Error checking toxicity: {e}")
        return False

def get_mushroom_info_from_gemini(mushroom_name):
    """Lấy thông tin chi tiết về nấm từ Gemini AI"""
    try:
        prompt = f"""
        Cung cấp thông tin chi tiết về loại nấm: {mushroom_name}
        
        Vui lòng trả lời theo định dạng JSON sau:
        {{
            "name": "Tên tiếng Việt",
            "scientific_name": "Tên khoa học",
            "edibility": "Có ăn được không (Ăn được/Độc/Không rõ)",
            "toxicity_level": "Mức độ độc tính (Không độc/Nhẹ/Trung bình/Cao/Chết người)",
            "appearance": "Mô tả hình dáng và đặc điểm",
            "habitat": "Môi trường sống",
            "warnings": "Cảnh báo và lưu ý"
        }}
        
        Chỉ trả lời bằng JSON, không thêm text khác.
        """
        
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"❌ Error getting info from Gemini: {e}")
        return json.dumps({
            "name": mushroom_name,
            "scientific_name": mushroom_name,
            "edibility": "Không rõ",
            "toxicity_level": "Không rõ",
            "appearance": "Thông tin không có sẵn",
            "habitat": "Thông tin không có sẵn",
            "warnings": "Hãy tham khảo chuyên gia trước khi sử dụng"
        }, ensure_ascii=False)

def save_prediction_to_db(prediction_data):
    """Lưu kết quả dự đoán vào database"""
    try:
        conn = sqlite3.connect('mushroom_history.db')
        c = conn.cursor()
        
        prediction_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        c.execute('''INSERT INTO predictions 
                     (id, timestamp, predicted_class, confidence, is_poisonous, image_data, gemini_info)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (prediction_id, timestamp, prediction_data['predicted_class'],
                   prediction_data['confidence'], prediction_data['is_poisonous'],
                   prediction_data['image_data'], prediction_data['gemini_info']))
        
        conn.commit()
        conn.close()
        return prediction_id
    except Exception as e:
        logger.error(f"❌ Error saving to database: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint chính để nhận diện nấm"""
    try:
        if model is None:
            return jsonify({"error": "Model chưa được load"}), 500
            
        data = request.get_json()
        image_data = data.get('image', '')
        
        if not image_data:
            return jsonify({"error": "Không có hình ảnh được gửi"}), 400
        
        logger.info("🔍 Processing mushroom prediction...")
        
        # Preprocess image
        image_tensor = preprocess_image(image_data)
        if image_tensor is None:
            return jsonify({"error": "Không thể xử lý hình ảnh"}), 400
        
        # Predict mushroom
        predicted_class, confidence = predict_mushroom(image_tensor)
        if predicted_class is None:
            return jsonify({"error": "Không thể nhận diện nấm"}), 500
        
        # Check if poisonous
        is_poisonous = is_mushroom_poisonous(predicted_class)
        
        # Get detailed info from Gemini
        logger.info("🤖 Getting detailed info from Gemini...")
        gemini_info = get_mushroom_info_from_gemini(predicted_class)
        
        # Prepare response data
        result = {
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "is_poisonous": is_poisonous,
            "gemini_info": gemini_info,
            "status": "success"
        }
        
        # Save to database
        prediction_data = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "is_poisonous": is_poisonous,
            "image_data": image_data[:100] + "...",  # Lưu một phần để tiết kiệm dung lượng
            "gemini_info": gemini_info
        }
        prediction_id = save_prediction_to_db(prediction_data)
        result["prediction_id"] = prediction_id
        
        logger.info(f"✅ Prediction successful: {predicted_class} ({confidence*100:.1f}%)")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Error in prediction endpoint: {e}")
        return jsonify({"error": f"Lỗi server: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Kiểm tra tình trạng server"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "num_classes": len(CLASS_NAMES),
        "timestamp": datetime.now().isoformat(),
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH)
    })

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint đơn giản"""
    return jsonify({
        "message": "🍄 Mushroom API is working!",
        "server_time": datetime.now().isoformat(),
        "device": str(DEVICE),
        "model_status": "Loaded" if model else "Not loaded",
        "classes_loaded": len(CLASS_NAMES),
        "model_architecture": "rexnet_150"
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    """Lấy danh sách tất cả các class"""
    return jsonify({
        "classes": CLASS_NAMES,
        "total_classes": len(CLASS_NAMES),
        "poisonous_mapping": POISONOUS_MAPPING
    })

@app.route('/history', methods=['GET'])
def get_history():
    """Get prediction history"""
    try:
        conn = sqlite3.connect('mushroom_history.db')
        c = conn.cursor()
        
        c.execute('''SELECT id, timestamp, predicted_class, confidence, is_poisonous 
                     FROM predictions ORDER BY timestamp DESC LIMIT 50''')
        
        history = []
        for row in c.fetchall():
            history.append({
                "id": row[0],
                "timestamp": row[1],
                "predicted_class": row[2],
                "confidence": round(row[3] * 100, 2),
                "is_poisonous": bool(row[4])
            })
        
        conn.close()
        return jsonify({"history": history})
        
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return jsonify({"error": "Không thể lấy lịch sử"}), 500

# SỬA: Thêm port từ environment variable cho deployment
if __name__ == '__main__':
    print("🍄" + "="*50)
    print("🍄 MUSHROOM RECOGNITION API SERVER")
    print("🍄" + "="*50)
    
    # Initialize database
    init_database()
    
    print(f"📱 Device: {DEVICE}")
    print(f"🔧 Model Status: {'✅ Loaded' if model else '❌ Failed to load'}")
    print(f"🗂️ Classes: {len(CLASS_NAMES)} mushroom types")
    print(f"📁 Model Path: {MODEL_PATH}")
    print(f"📄 Model File Exists: {os.path.exists(MODEL_PATH)}")
    print("🍄" + "="*50)
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)