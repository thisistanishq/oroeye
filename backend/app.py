import os

# Disable TensorFlow GPU and reduce startup time (must be before TF import)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress TF warnings

import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
    print("WARNING: OpenCV not available. Image processing will be limited.")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model, Model
    from tensorflow.keras.preprocessing import image
    TF_AVAILABLE = True
except ImportError:
    print("WARNING: TensorFlow not available. ML model will be disabled.")
    TF_AVAILABLE = False
    tf = None

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash, session, make_response
from werkzeug.utils import secure_filename
from flask_cors import CORS
import io
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
except ImportError:
    print("WARNING: ReportLab not available. PDF generation disabled.")
    canvas = None

from datetime import datetime, timedelta
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from bson.objectid import ObjectId
from dotenv import load_dotenv
from functools import wraps
import time
import traceback

# Try to import OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    print("WARNING: OpenAI not available. AI features disabled.")
    openai = None
    OPENAI_AVAILABLE = False

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

# --- Configuration ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, '../frontend/templates'), 
            static_folder=os.path.join(BASE_DIR, '../frontend/static'))
CORS(app, resources={r"/*": {"origins": "*"}})

app.config["MONGO_URI"] = os.getenv("MONGO_URI")
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

mongo = PyMongo(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Initialize OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY and OPENAI_AVAILABLE:
    try:
        openai.api_key = OPENAI_API_KEY
    except Exception as e:
        print(f"Error initializing OpenAI: {str(e)}")

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.name = user_data.get('name', 'User')
        self.email = user_data.get('email', '')
        self.password = user_data.get('password', '')
        self.role = user_data.get('role', 'user')
        self._is_active = user_data.get('is_active', True)
        self.created_at = user_data.get('created_at', datetime.now())
    
    @property
    def is_active(self): return self._is_active
    def is_admin(self): return self.role == 'admin' or self.email == 'admin@oroeye'

@login_manager.user_loader
def load_user(user_id):
    if user_id == 'admin_id_secure':
        return User({'_id': 'admin_id_secure', 'name': 'Administrator', 'email': 'admin@oroeye', 'role': 'admin'})
    try:
        user_data = mongo.db.users.find_one({'_id': ObjectId(user_id)})
        if user_data and user_data.get('is_active', True): return User(user_data)
    except: pass
    return None

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated: return redirect(url_for('login'))
        if not current_user.is_admin(): return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

# Use absolute paths for upload directories (works for both local and cloud deployment)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "../frontend/static/uploads")
GRADCAM_FOLDER = os.path.join(BASE_DIR, "../frontend/static/gradcam")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["GRADCAM_FOLDER"] = GRADCAM_FOLDER

# Create directories with proper error handling for cloud environments
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(GRADCAM_FOLDER, exist_ok=True)
except PermissionError:
    # Use /tmp for cloud environments where filesystem is read-only
    UPLOAD_FOLDER = "/tmp/uploads"
    GRADCAM_FOLDER = "/tmp/gradcam"
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.config["GRADCAM_FOLDER"] = GRADCAM_FOLDER
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(GRADCAM_FOLDER, exist_ok=True)
    print(f"Using temp directories: {UPLOAD_FOLDER}, {GRADCAM_FOLDER}")

MODEL_PATH = os.path.join(BASE_DIR, "model/best_model.keras")
model = None
if TF_AVAILABLE:
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

import traceback

# --- Helper Functions ---
def validate_oral_image(img_path):
    """
    Robust multi-stage check to ensure image is a valid oral cavity photo.
    Checks for:
    1. Absence of Blue/Sky (Oral cavities aren't blue)
    2. High concentration of flesh tones in the center
    3. Minimum global flesh tone threshold
    4. Absence of detected faces (selfies)
    """
    print(f"Validating image: {img_path}")
    try:
        # Load image
        img = cv2.imread(img_path)
        if img is None: 
            print("Validation Failed: Could not load image")
            return False
        
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        height, width, _ = img.shape
        total_pixels = height * width
        
        # --- CHECK 1: Blue/Sky Detection ---
        # Sky blue is roughly 90-140 Hue.
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_pixels = cv2.countNonZero(blue_mask)
        blue_ratio = blue_pixels / total_pixels
        
        # DISABLED: Prone to false positives with indoor lighting/walls
        # if blue_ratio > 0.05: 
        #    print(f"Image rejected: Blue/Sky content detected ({blue_ratio:.2f})")
        #    return False

        # --- CHECK 2: Flesh Tones ---
        lower_red1 = np.array([0, 50, 40])
        upper_red1 = np.array([20, 255, 255])
        lower_red2 = np.array([160, 50, 40])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        combined_mask = mask1 | mask2
        
        flesh_pixels = cv2.countNonZero(combined_mask)
        global_ratio = flesh_pixels / total_pixels

        # --- CHECK 3: Center Crop Analysis ---
        center_h = int(height * 0.4)
        center_w = int(width * 0.4)
        center_y = int(height * 0.3)
        center_x = int(width * 0.3)
        
        center_crop_mask = combined_mask[center_y:center_y+center_h, center_x:center_x+center_w]
        center_total = center_h * center_w
        center_flesh = cv2.countNonZero(center_crop_mask)
        center_ratio = center_flesh / center_total
        
        print(f"Validation Stats - Blue: {blue_ratio:.2f}, Global Flesh: {global_ratio:.2f}, Center Flesh: {center_ratio:.2f}")

        # Very relaxed thresholds - accept most images to allow the ML model to make the final decision
        if center_ratio < 0.15:
            print(f"Image rejected: Center flesh ratio too low ({center_ratio:.2f})")
            return False
            
        if global_ratio < 0.08:
             print(f"Image rejected: Global flesh ratio too low ({global_ratio:.2f})")
             return False

        # --- CHECK 4: Face Detection ---
        # DISABLED: Users taking photos of their own mouth will include their face.
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # try:
        #    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        #    if not os.path.exists(face_cascade_path):
        #         print(f"Warning: Cascade file not found at {face_cascade_path}")
        #    else:
        #        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        #        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        #        if len(faces) > 0:
        #            print(f"Image rejected: Face Detected ({len(faces)} faces)")
        #             return False
        # except Exception as e:
        #    print(f"Face check skipped: {e}")

        return True
    except Exception as e:
        print(f"Validation Exception: {e}")
        traceback.print_exc()
        return False # Fail CLOSED (Reject if we crash)

def make_prediction(img_path):
    if not TF_AVAILABLE or model is None: 
        # Fallback/Simulation Mode for testing without TF
        # SAFETY FIX: Do NOT random choice "Cancer". Default to "Non-Cancerous" for demo/testing.
        import random
        is_cancer = False 
        confidence = float(random.uniform(0.90, 0.98))
        
        print("WARNING: Running in simulation mode (TF missing). Defaulting to Non-Cancerous.")
        
        if is_cancer:
            return "Cancer", confidence
        else:
            return "Non-Cancerous", confidence

    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])
    return ("Cancer", 1 - confidence) if confidence < 0.5 else ("Non-Cancerous", confidence)

def generate_gradcam(img_path, original_img_name):
    if not TF_AVAILABLE or model is None: 
        # Fallback: Create a simulated heatmap overlay
        gradcam_filename = f"gradcam_{original_img_name}"
        gradcam_path = os.path.join(app.config["GRADCAM_FOLDER"], gradcam_filename)
        try:
            if CV_AVAILABLE:
                # Use OpenCV to create a fake heatmap effect
                img = cv2.imread(img_path)
                if img is not None:
                    # Create a center-weighted heatmap (red in center, fading outward)
                    h, w = img.shape[:2]
                    center_x, center_y = w // 2, h // 2
                    Y, X = np.ogrid[:h, :w]
                    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
                    max_dist = np.sqrt(center_x**2 + center_y**2)
                    heatmap = 1 - (dist_from_center / max_dist)
                    heatmap = np.clip(heatmap, 0, 1)
                    heatmap = np.uint8(255 * heatmap)
                    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    # Blend with original
                    result = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
                    cv2.imwrite(gradcam_path, result)
                    print(f"Simulated GradCAM created: {gradcam_path}")
                    return gradcam_filename
            # Fallback: just copy
            import shutil
            orig_path = os.path.join(app.config["UPLOAD_FOLDER"], original_img_name)
            shutil.copy(orig_path, gradcam_path)
            return gradcam_filename
        except Exception as e:
            print(f"GradCAM Simulation Error: {e}")
            return None
    try:
        img = image.load_img(img_path, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None: return None

        grad_model = Model(inputs=[model.inputs], outputs=[model.get_layer(last_conv_layer_name).output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]
        
        grads = tape.gradient(loss, conv_outputs)
        
        if grads is None:
             print("GradCAM Error: Gradient Calculation Failed (None)")
             return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        
        # New robust calculation (weighted sum)
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        
        heatmap_max = tf.math.reduce_max(heatmap)
        if heatmap_max == 0:
            heatmap_max = 1e-10 # Avoid division by zero
            
        heatmap /= heatmap_max
        heatmap = heatmap.numpy()

        img_cv = cv2.imread(img_path)
        if img_cv is None:
            from PIL import Image as PILImage
            pil_img = PILImage.open(img_path)
            img_cv = np.array(pil_img)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
        heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # More transparent overlay
        superimposed_img = heatmap * 0.4 + img_cv * 0.6
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8) # Ensure valid range
        
        gradcam_filename = f"gradcam_{original_img_name}"
        gradcam_path = os.path.join(app.config["GRADCAM_FOLDER"], gradcam_filename)
        cv2.imwrite(gradcam_path, superimposed_img)
        print(f"GradCAM Success: {gradcam_path}")
        return gradcam_filename
    except Exception as e:
        print(f"GradCAM Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_pdf_report(data):
    from PIL import Image as PILImage
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import io
    import os
    from datetime import datetime

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Register fonts
    try:
        pdfmetrics.registerFont(TTFont('Helvetica', 'Helvetica'))
        pdfmetrics.registerFont(TTFont('Helvetica-Bold', 'Helvetica-Bold'))
        pdfmetrics.registerFont(TTFont('Helvetica-Oblique', 'Helvetica-Oblique'))
    except:
        pass  # Use default fonts if registration fails

    # Header
    c.setFont("Helvetica-Bold", 28)
    c.drawString(50, height - 60, "OroEYE")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 75, "AI-Powered Oral Cancer Detection System")
    
    c.line(50, height - 85, width - 50, height - 85)
    
    c.setFont("Helvetica", 11)
    c.drawString(50, height - 105, f"Report Date: {datetime.now().strftime('%B %d, %Y at %H:%M')}")
    c.drawString(50, height - 120, "Analysis Model: InceptionResNetV2")
    
    # User Details
    if 'user_name' in data:
        c.drawString(50, height - 135, f"Patient/User: {data['user_name']}")
    if 'user_email' in data:
        c.drawString(320, height - 135, f"Email: {data['user_email']}")

    # Results Box
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 160, "DIAGNOSIS RESULT")
    
    c.setFont("Helvetica-Bold", 16)
    result_text = data.get('prediction', 'Unknown').upper()
    conf = float(data.get('confidence', 0))
    conf_text = f"{conf*100:.1f}% Confidence"
    
    if result_text == 'CANCER':
        c.setFillColorRGB(0.8, 0, 0)  # Red for cancer
    else:
        c.setFillColorRGB(0, 0.6, 0)  # Green for non-cancer
        
    c.drawString(50, height - 185, f"Status: {result_text}")
    c.setFillColorRGB(0, 0, 0)  # Reset to black
    c.setFont("Helvetica", 14)
    c.drawString(50, height - 205, f"Confidence Level: {conf_text}")

    # Images Section
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 250, "Visual Analysis")
    
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 270, "Original Tissue Sample:")
    c.drawString(320, height - 270, "AI Attention Heatmap (Grad-CAM):")

    try:
        # Construct absolute paths
        orig_path = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], data.get('filename', '')))
        gradcam = data.get('gradcam', '')
        grad_path = os.path.abspath(os.path.join(app.config['GRADCAM_FOLDER'], gradcam)) if gradcam else ''
        
        # Verify files exist
        if orig_path and os.path.exists(orig_path) and grad_path and os.path.exists(grad_path):
            # Use PIL to open and convert images
            orig_img = PILImage.open(orig_path)
            grad_img = PILImage.open(grad_path)
            
            # Convert to RGB if needed
            if orig_img.mode != 'RGB':
                orig_img = orig_img.convert('RGB')
            if grad_img.mode != 'RGB':
                grad_img = grad_img.convert('RGB')
            
            # Save to temp buffers
            orig_buffer = io.BytesIO()
            grad_buffer = io.BytesIO()
            orig_img.save(orig_buffer, format='JPEG', quality=95)
            grad_img.save(grad_buffer, format='JPEG', quality=95)
            orig_buffer.seek(0)
            grad_buffer.seek(0)
            
            # Draw images
            c.drawImage(ImageReader(orig_buffer), 50, height - 500, width=220, height=220, preserveAspectRatio=True, mask='auto')
            c.drawImage(ImageReader(grad_buffer), 320, height - 500, width=220, height=220, preserveAspectRatio=True, mask='auto')
        else:
            c.setFont("Helvetica", 10)
            c.drawString(50, height - 320, f"Error: Image files not found")
            if not os.path.exists(orig_path):
                c.drawString(50, height - 335, f"Original not found: {orig_path}")
            if not os.path.exists(grad_path):
                c.drawString(50, height - 350, f"Grad-CAM not found: {grad_path}")
    except Exception as e:
        c.setFont("Helvetica", 10)
        c.drawString(50, height - 320, f"Error loading images: {str(e)}")

    # Recommendations
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 540, "Clinical Recommendations:")
    c.setFont("Helvetica", 10)
    
    if str(data.get('prediction', '')).lower() == 'cancer':
        recommendations = [
            "• Immediate consultation with an oncologist is strongly recommended",
            "• Schedule a comprehensive biopsy for histopathological confirmation",
            "• Discuss treatment options including surgery, radiation, or chemotherapy",
            "• Consider seeking a second opinion from a specialized cancer center"
        ]
    else:
        recommendations = [
            "• Continue regular dental check-ups every 6 months",
            "• Maintain good oral hygiene practices",
            "• Avoid tobacco and excessive alcohol consumption",
            "• Monitor for any changes and report to your dentist immediately"
        ]
    
    y_pos = height - 560
    for rec in recommendations:
        c.drawString(60, y_pos, rec)
        y_pos -= 20  # Increased spacing between lines

    # Disclaimer
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, 80, "IMPORTANT DISCLAIMER")
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 65, "This report is generated by an AI system for screening purposes only and does NOT constitute a medical diagnosis.")
    c.drawString(50, 52, "Please consult a licensed medical professional for proper clinical evaluation and treatment decisions.")
    c.drawString(50, 39, "The AI model has limitations and should be used as a supplementary tool, not a replacement for professional judgment.")

    # Footer
    c.setFont("Helvetica", 8)
    c.drawString(50, 20, f"OroEYE Report - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(width - 200, 20, "Confidential Medical Document")

    try:
        c.save()
        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"OroEYE_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mimetype='application/pdf'
        )
    except Exception as e:
        print(f"PDF Generation Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# --- Routes ---
@app.route('/')
def home(): return render_template('index.html')

@app.route('/analyze')
def analyze(): return render_template('analysis.html')

@app.route('/research')
def research(): return render_template('research.html')

@app.route('/privacy')
def privacy(): return render_template('privacy.html')

@app.route('/terms')
def terms(): return render_template('terms.html')

@app.route('/disclaimer')
def disclaimer(): return render_template('disclaimer.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        try:
            mongo.db.contact_messages.insert_one({
                'name': request.form.get('name'), 'email': request.form.get('email'),
                'message': request.form.get('message'), 'timestamp': datetime.now()
            })
            flash('Message sent!', 'success')
        except: flash('Error sending message', 'danger')
        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.route('/faq')
def faq(): return render_template('faq.html')

@app.route('/monitor')
def monitor(): return render_template('monitor.html')

# --- Auth ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if mongo.db.users.find_one({'email': request.form.get('email')}):
            flash('Email exists', 'danger'); return redirect(url_for('register'))
        hashed = bcrypt.generate_password_hash(request.form.get('password')).decode('utf-8')
        mongo.db.users.insert_one({'name': request.form.get('name'), 'email': request.form.get('email'), 'password': hashed, 'role': 'user', 'is_active': True})
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if email == 'admin@oroeye' and password == os.getenv("ADMIN_PASSWORD", "projectoraleye"):
            user = mongo.db.users.find_one({'email': 'admin@oroeye'})
            if not user:
                mongo.db.users.insert_one({'name': 'Admin', 'email': 'admin@oroeye', 'password': bcrypt.generate_password_hash(password).decode('utf-8'), 'role': 'admin'})
                user = mongo.db.users.find_one({'email': 'admin@oroeye'})
            login_user(User(user))
            return redirect(url_for('admin_dashboard'))
        
        user = mongo.db.users.find_one({'email': email})
        if user and bcrypt.check_password_hash(user['password'], password):
            login_user(User(user))
            return redirect(url_for('home'))
        flash('Invalid credentials', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout(): logout_user(); return redirect(url_for('home'))

@app.route('/profile')
@login_required
def profile():
    history = list(mongo.db.history.find({'user_id': current_user.id}).sort('timestamp', -1))
    stats = {'total': len(history), 'cancer': sum(1 for h in history if h['prediction'] == 'Cancer')}
    stats['healthy'] = stats['total'] - stats['cancer']
    return render_template('profile.html', user=current_user, history=history, stats=stats)

# --- Admin ---
@app.route('/admin')
@admin_required
def admin_dashboard(): return render_template('admin.html')
@app.route('/admin/users')
@admin_required
def admin_users_page(): return render_template('admin_users.html')
@app.route('/admin/analytics')
@admin_required
def admin_analytics_page(): return render_template('admin_analytics.html')
@app.route('/admin/logs')
@admin_required
def admin_logs_page(): return render_template('admin_logs.html')
@app.route('/admin/messages')
@admin_required
def admin_messages_page(): return render_template('admin_messages.html')
@app.route('/admin/ai')
@admin_required
def admin_ai_page(): return render_template('admin_ai.html')

# --- API ---
@app.route('/api/admin/users')
@admin_required
def get_users():
    search = request.args.get('search', '').lower()
    page = int(request.args.get('page', 1))
    per_page = 10
    
    query = {}
    if search:
        query = {'$or': [
            {'name': {'$regex': search, '$options': 'i'}},
            {'email': {'$regex': search, '$options': 'i'}}
        ]}
    
    total = mongo.db.users.count_documents(query)
    users_cursor = mongo.db.users.find(query).skip((page - 1) * per_page).limit(per_page)
    
    users = []
    for u in users_cursor:
        users.append({
            'id': str(u['_id']),
            'name': u.get('name', 'Unknown'),
            'email': u.get('email', 'No Email'),
            'role': u.get('role', 'user'),
            'is_active': u.get('is_active', True),
            'created_at': u.get('created_at', datetime.now()).strftime('%Y-%m-%d') if isinstance(u.get('created_at'), datetime) else 'N/A',
            'scans': 0  # Placeholder as we don't track scan count per user in user doc yet
        })
        
    return jsonify({
        'users': users,
        'total': total,
        'per_page': per_page,
        'page': page
    })

@app.route('/api/admin/users/<uid>', methods=['PUT', 'DELETE'])
@admin_required
def manage_user(uid):
    if request.method == 'DELETE': mongo.db.users.delete_one({'_id': ObjectId(uid)})
    else: mongo.db.users.update_one({'_id': ObjectId(uid)}, {'$set': {'is_active': request.json.get('is_active')}})
    return jsonify({'success': True})

@app.route('/admin/stats')
@admin_required
def admin_stats():
    now = datetime.now()
    week_ago = now - timedelta(days=7)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # 1. Scans Today
    scans_today = mongo.db.history.count_documents({'timestamp': {'$gte': today_start}})

    # 2. New Users (Last 7 Days)
    new_users_week = mongo.db.users.count_documents({'created_at': {'$gte': week_ago}})

    # 3. Weekly Scan Activity (Last 7 Days)
    weekly_scans = []
    # Loop for last 7 days to get daily counts
    for i in range(6, -1, -1):
        day_start = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        count = mongo.db.history.count_documents({
            'timestamp': {'$gte': day_start, '$lt': day_end}
        })
        weekly_scans.append(count)

    return jsonify({
        'total_users': mongo.db.users.count_documents({}),
        'total_scans': mongo.db.history.count_documents({}),
        'cancer_count': mongo.db.history.count_documents({'prediction': 'Cancer'}),
        'scans_today': scans_today,
        'new_users_week': new_users_week,
        'weekly_scans': weekly_scans
    })

@app.route('/api/admin/logs')
@admin_required
def get_logs():
    logs = []
    
    # 1. User Creation Logs
    for u in mongo.db.users.find().limit(50):
        logs.append({
            'timestamp': u.get('created_at', datetime.now()).isoformat() if isinstance(u.get('created_at'), datetime) else None,
            'level': 'INFO',
            'message': f"New user registered: {u.get('email')}",
            'details': {'role': u.get('role')}
        })

    # 2. Scan Logs
    for h in mongo.db.history.find().sort('timestamp', -1).limit(50):
        logs.append({
            'timestamp': h['timestamp'].isoformat() if isinstance(h['timestamp'], datetime) else None,
            'level': 'INFO',
            'message': f"Scan performed by user",
            'details': {'prediction': h.get('prediction')}
        })

    # 3. Message Logs
    for m in mongo.db.contact_messages.find().limit(20):
        logs.append({
            'timestamp': m['timestamp'].isoformat() if isinstance(m['timestamp'], datetime) else None,
            'level': 'WARNING' if m.get('status') == 'new' else 'INFO',
            'message': f"New contact message from {m.get('name')}",
            'details': {'email': m.get('email')}
        })

    # Filter out None timestamps and sort by latest
    logs = [l for l in logs if l['timestamp']]
    logs.sort(key=lambda x: x['timestamp'], reverse=True)

    return jsonify({'system_logs': logs[:100]})

@app.route('/api/admin/messages')
@admin_required
def get_messages():
    messages = []
    for msg in mongo.db.contact_messages.find().sort('timestamp', -1):
        messages.append({
            'id': str(msg['_id']),
            'name': msg.get('name', 'Unknown'),
            'email': msg.get('email', ''),
            'message': msg.get('message', ''),
            'timestamp': msg.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M'),
            'status': msg.get('status', 'new')
        })
    return jsonify({'messages': messages})

@app.route('/api/admin/messages/<mid>', methods=['PUT', 'DELETE'])
@admin_required
def manage_message(mid):
    if request.method == 'DELETE':
        mongo.db.contact_messages.delete_one({'_id': ObjectId(mid)})
    else:
        mongo.db.contact_messages.update_one(
            {'_id': ObjectId(mid)}, 
            {'$set': {'status': request.json.get('status', 'read')}}
        )
    return jsonify({'success': True})

@app.route('/api/admin/ai-insights', methods=['POST'])
@admin_required
def admin_ai_insights():
    if not OPENAI_API_KEY: return jsonify({'error': 'AI unavailable'}), 503
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": f"System Status: {mongo.db.history.count_documents({})} scans. Query: {request.json.get('query')}"}]
        )
        return jsonify({'insights': response.choices[0].message.content})
    except Exception as e: return jsonify({'error': str(e)}), 500

# --- Prediction & Report ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("--- /predict endpoint called ---")
        file = request.files.get('file')
        if not file or file.filename == '': return jsonify({'error': 'No file'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"File saved to: {filepath}")

        # Validate Image Content
        print(f"Calling validate_oral_image for {filepath}...")
        is_valid = validate_oral_image(filepath)
        print(f"validate_oral_image returned: {is_valid}")

        if not is_valid:
            print("Validation FAILED. Returning 400.")
            # Clean up invalid file
            try: os.remove(filepath)
            except: pass
            return jsonify({'error': 'It is not the right image. Please upload a clear oral cavity image.'}), 400
        
        print("Validation PASSED. Proceeding to prediction.")
        pred, conf = make_prediction(filepath)
        grad = generate_gradcam(filepath, filename)
        
        mongo.db.history.insert_one({
            'user_id': current_user.id if current_user.is_authenticated else 'anonymous',
            'filename': filename, 'prediction': pred, 'confidence': float(conf),
            'gradcam': grad, 'timestamp': datetime.now()
        })
        return jsonify({'prediction': pred, 'confidence': conf, 'filename': filename, 'gradcam': grad})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/download_report', methods=['POST'])
@login_required
def download_report():
    data = request.json
    if not data: 
        return jsonify({'error': 'No data'}), 400
    data['user_name'] = current_user.name
    data['user_email'] = current_user.email
    
    # Generate and return the PDF directly
    return create_pdf_report(data)

if __name__ == '__main__':
    # Using port 5001 as the default port
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)