from flask import Flask, render_template, send_from_directory, jsonify, request, redirect, url_for, send_file, flash
import pandas as pd
import sys
import qrcode
import os
import shutil
import random
import string
import json
from difflib import SequenceMatcher
from cryptography.fernet import Fernet
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from werkzeug.utils import secure_filename
import threading
import platform
from tempfile import NamedTemporaryFile
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import subprocess
import traceback
from dotenv import load_dotenv
import time
import atexit
import atexit
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import win32print
import base64
from functools import lru_cache
import concurrent.futures

load_dotenv()

app = Flask(__name__, static_folder='static')
BASE_DIR = os.getenv('BASE_DIR', os.path.dirname(os.path.abspath(__file__)))

# For persistent storage (Render disk volume)
DATA_DIR = os.path.join(BASE_DIR, "data")  # Parent directory for persistent storage
os.makedirs(DATA_DIR, exist_ok=True)  # Ensure it exists

# Persistent storage paths (will be created on disk volume)
QR_FOLDER = os.path.join(DATA_DIR, "qr_codes")
DATASET_DIR = os.path.join(DATA_DIR, "participant_list") 
TEMPLATE_DIR = os.path.join(DATA_DIR, "id_templates")  
CONFIG_PATH = os.path.join(DATA_DIR, "active_config.json")

# For static assets (committed to Git)
STATIC_DIR = os.path.join(BASE_DIR, "static")
STATIC_TEMPLATES = os.path.join(STATIC_DIR, "id_templates")  
STATIC_QR_CODES = os.path.join(STATIC_DIR, "qr_codes")  

# Flask templates (for HTML files)
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Ensure all required directories exist at startup
required_dirs = [
    QR_FOLDER,
    DATASET_DIR, 
    TEMPLATE_DIR,  
    STATIC_TEMPLATES,
    STATIC_QR_CODES,
    TEMPLATES_DIR
]

for directory in required_dirs:
    os.makedirs(directory, exist_ok=True)

app.secret_key = os.getenv('SECRET_KEY')

with open("qr_key.key", "rb") as f:
    fernet = Fernet(f.read())


def check_upload_availability():
    """Check if persistent storage is ready for uploads"""
    while not os.path.exists('/opt/render/project/src/data/.ready'):
        time.sleep(1)

# Run this check when app starts
threading.Thread(target=check_upload_availability, daemon=True).start()

@app.route("/upload_status")
def upload_status():
    return jsonify({
        "ready": os.path.exists(os.path.join(DATA_DIR, ".ready")),
        "message": "System ready for uploads" if os.path.exists(os.path.join(DATA_DIR, ".ready")) 
    else "Initializing storage..."
    })

def load_active_config():
    if not os.path.exists(CONFIG_PATH):
        return {}  # Return empty dict for first run
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def get_active_dataset():
    config = load_active_config()
    dataset_path = os.path.join(DATASET_DIR, config["active_dataset"])
    return pd.read_csv(dataset_path), config["active_template"]

def get_template_path(filename):
    # Check persistent storage first
    persistent_path = os.path.join(TEMPLATE_DIR, filename)
    if os.path.exists(persistent_path):
        return persistent_path
        
    # Fall back to static files
    static_path = os.path.join(STATIC_TEMPLATES, filename)
    if os.path.exists(static_path):
        return static_path
        
    raise FileNotFoundError(f"Template {filename} not found")

def get_template_preview_url(template_filename):
    """Returns URL for template from either persistent storage or static files"""
    # Check persistent storage first
    persistent_path = os.path.join(TEMPLATE_DIR, template_filename)
    if os.path.exists(persistent_path):
        return url_for('serve_persistent_template', filename=template_filename)
    
    # Fall back to static files
    static_path = os.path.join(app.static_folder, 'id_templates', template_filename)
    if os.path.exists(static_path):
        return url_for('static', filename=f'id_templates/{template_filename}')
    
    # Default fallback
    return url_for('static', filename='id_templates/default_template.png')

app.jinja_env.globals.update(get_template_preview_url=get_template_preview_url)

@app.route("/first-run")
def first_run_setup():
    # Ensure all required directories exist
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    os.makedirs(QR_FOLDER, exist_ok=True)
    
    # Get existing files
    try:
        dataset_files = sorted(os.listdir(DATASET_DIR), reverse=True)
    except FileNotFoundError:
        dataset_files = []
    
    try:
        template_files = sorted(os.listdir(TEMPLATE_DIR), reverse=True)
    except FileNotFoundError:
        template_files = []
    
    return render_template("first_run.html",
        dataset_files=dataset_files,
        template_files=template_files)

@app.route('/persistent_templates/<filename>')
def serve_persistent_template(filename):
    """Special route for serving templates from persistent storage"""
    return send_from_directory(TEMPLATE_DIR, filename)

def clean_directory(dir_path, exclude_files=None, max_age_days=7):
    """Clean directory with age-based deletion"""
    exclude_files = exclude_files or set()
    now = time.time()
    for filename in os.listdir(dir_path):
        if filename in exclude_files:
            continue
        file_path = os.path.join(dir_path, filename)
        try:
            file_age = (now - os.path.getmtime(file_path)) / (24 * 3600)
            if file_age > max_age_days:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        except Exception as e:
            app.logger.error(f"Cleanup failed for {file_path}: {str(e)}")

def cleanup_old_templates(current_template):
    """
    Remove old template files while preserving the current active template
    and default template.
    """
    try:
        for filename in os.listdir(TEMPLATE_DIR):
            file_path = os.path.join(TEMPLATE_DIR, filename)
            # Skip if: current template, default template, or not an image file
            if (filename == current_template or 
                filename == 'default_template.png' or
                not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))):
                continue
                
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    app.logger.info(f"Cleaned up old template: {filename}")
            except Exception as e:
                app.logger.error(f"Failed to delete old template {filename}: {str(e)}")
                
    except Exception as e:
        app.logger.error(f"Template cleanup failed: {str(e)}")

def update_encrypted_qr_column(df):
    """Ensure all rows have valid encrypted QR data"""
    for index, row in df.iterrows():
        try:
            # Test if existing QR data is valid
            if 'EncryptedQR' in df.columns:
                fernet.decrypt(row['EncryptedQR'].encode()).decode()
        except:
            # Regenerate if invalid
            df.at[index, 'EncryptedQR'] = fernet.encrypt(
                f"id={str(row['ID']).zfill(3)};name={row['Name']};position={row['Position']};company={row['Company']}".encode()
            ).decode()
    return df

def generate_qrs(df):
    """Generate QR codes for all rows in dataframe, cleaning old ones"""
    current_ids = {f"{row['ID']}.png" for _, row in df.iterrows()}
    
    # Clean directory keeping only current IDs
    clean_directory(QR_FOLDER, exclude_files=current_ids)
    
    # Generate new QR codes
    for _, row in df.iterrows():
        filename = f"{row['ID']}.png"
        file_path = os.path.join(QR_FOLDER, filename)
        
        if not os.path.exists(file_path):
            try:
                qr_img = qrcode.make(row['EncryptedQR'])
                qr_img.save(file_path)
                # Optimize image memory usage
                with Image.open(file_path) as img:
                    img.save(file_path, optimize=True, quality=85)
            except Exception as e:
                app.logger.error(f"Failed to generate QR for ID {row['ID']}: {str(e)}")

def safe_float(val, default):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def scheduled_cleanup():
    """Regular cleanup of orphaned files"""
    with app.app_context():
        try:
            config = load_active_config()
            active_files = set()
            
            if config.get("active_dataset"):
                try:
                    df = pd.read_csv(os.path.join(DATASET_DIR, config["active_dataset"]))
                    active_files.update(f"{row['ID']}.png" for _, row in df.iterrows())
                except Exception as e:
                    app.logger.error(f"Cleanup dataset read failed: {str(e)}")
            
            # Clean QR folder
            clean_directory(QR_FOLDER, exclude_files=active_files)
            
            # Clean templates
            if config.get("active_template"):
                cleanup_old_templates(config["active_template"])
                
        except Exception as e:
            app.logger.error(f"Scheduled cleanup failed: {str(e)}")

# Initialize scheduler
# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(func=scheduled_cleanup, trigger="interval", hours=6)

try:
    scheduler.start()
    app.logger.info("Scheduler started successfully")
except Exception as e:
    app.logger.error(f"Failed to start scheduler: {str(e)}")
    # Depending on your application, you might want to:
    # 1. Exit the application
    # 2. Disable scheduled tasks
    # 3. Fall back to manual cleanup
    sys.exit(1)  # If this is critical functionality

# Ensure scheduler shuts down cleanly
atexit.register(lambda: scheduler.shutdown())

@app.route("/activate", methods=["POST"])
def activate():
    try:
        # Check if this is a first-run setup
        is_first_run = request.form.get("triggeredBy") == "first_run"
        
        # Ensure directories exist
        os.makedirs(DATASET_DIR, exist_ok=True)
        os.makedirs(TEMPLATE_DIR, exist_ok=True)
        os.makedirs(QR_FOLDER, exist_ok=True)

        dataset_file = request.files.get("dataset_file")
        template_file = request.files.get("template_file")
        existing_dataset = request.form.get("existing_dataset")
        existing_template = request.form.get("existing_template")
        config = load_active_config()
        
        # Process dataset selection
        if existing_dataset:
            # Use existing dataset if selected
            config["active_dataset"] = existing_dataset
        elif dataset_file and dataset_file.filename:
            # Process new dataset upload
            random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
            dataset_filename = f"{int(time.time())}_{random_str}_{secure_filename(dataset_file.filename)}"
            dataset_path = os.path.join(DATASET_DIR, dataset_filename)
            
            dataset_file.save(dataset_path)
            config["active_dataset"] = dataset_filename
            
            try:
                df = pd.read_csv(dataset_path)
                required_cols = {'ID', 'Name', 'Position', 'Company'}
                if not required_cols.issubset(df.columns):
                    os.remove(dataset_path)
                    if is_first_run:
                        flash("Dataset missing required columns: ID, Name, Position, Company", "error")
                        return redirect(url_for("first_run_setup"))
                    return jsonify({
                        "error": "Dataset missing required columns",
                        "details": "Required columns: ID, Name, Position, Company",
                        "received_columns": list(df.columns)
                    }), 400
                
                if df['ID'].duplicated().any():
                    os.remove(dataset_path)
                    if is_first_run:
                        flash("Duplicate IDs found in dataset - each ID must be unique", "error")
                        return redirect(url_for("first_run_setup"))
                    return jsonify({
                        "error": "Duplicate IDs found in dataset",
                        "details": "Each ID must be unique"
                    }), 400
                
                regenerate_qr = 'EncryptedQR' not in df.columns
                if not regenerate_qr:
                    try:
                        sample_row = df.iloc[0]
                        decrypted = fernet.decrypt(sample_row['EncryptedQR'].encode()).decode()
                        regenerate_qr = not all(
                            f"{k}={v}" in decrypted 
                            for k, v in sample_row[['ID', 'Name', 'Position', 'Company']].items()
                        )
                    except:
                        regenerate_qr = True
                
                if regenerate_qr:
                    df['EncryptedQR'] = df.apply(lambda row: fernet.encrypt(
                        f"id={str(row['ID']).zfill(3)};name={row['Name']};position={row['Position']};company={row['Company']}".encode()
                    ).decode(), axis=1)
                    df.to_csv(dataset_path, index=False)
                
                # Clear ALL old QR codes before generating new ones
                clean_directory(QR_FOLDER)
                generate_qrs(df)
                        
            except pd.errors.EmptyDataError:
                os.remove(dataset_path)
                if is_first_run:
                    flash("The dataset file is empty", "error")
                    return redirect(url_for("first_run_setup"))
                return jsonify({"error": "Dataset file is empty"}), 400
            except Exception as e:
                if os.path.exists(dataset_path):
                    os.remove(dataset_path)
                app.logger.error(f"Dataset processing failed: {str(e)}", exc_info=True)
                if is_first_run:
                    flash(f"Dataset processing failed: {str(e)}", "error")
                    return redirect(url_for("first_run_setup"))
                return jsonify({
                    "error": "Dataset processing failed",
                    "details": str(e)
                }), 400

        # Process template selection
        if existing_template:
            # Use existing template if selected
            config["active_template"] = existing_template
        elif template_file and template_file.filename:
            # Process new template upload
            random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
            template_filename = f"{int(time.time())}_{random_str}_{secure_filename(template_file.filename)}"
            template_path = os.path.join(TEMPLATE_DIR, template_filename)
            
            try:
                allowed_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
                file_ext = os.path.splitext(template_file.filename.lower())[1]
                if file_ext not in allowed_extensions:
                    if is_first_run:
                        flash(f"Invalid template file type. Allowed types: {', '.join(allowed_extensions)}", "error")
                        return redirect(url_for("first_run_setup"))
                    return jsonify({
                        "error": "Invalid template file type",
                        "details": f"Allowed types: {', '.join(allowed_extensions)}"
                    }), 400
                
                template_file.save(template_path)
                config["active_template"] = template_filename
                
                try:
                    with Image.open(template_path) as img:
                        img.verify()
                        width, height = img.size
                        if width < 500 or height < 500:
                            os.remove(template_path)
                            if is_first_run:
                                flash("Image dimensions too small (minimum 500x500 pixels)", "error")
                                return redirect(url_for("first_run_setup"))
                            return jsonify({
                                "error": "Image dimensions too small",
                                "details": "Minimum size: 500x500 pixels"
                            }), 400
                except Exception as e:
                    os.remove(template_path)
                    if is_first_run:
                        flash(f"Invalid image file: {str(e)}", "error")
                        return redirect(url_for("first_run_setup"))
                    return jsonify({
                        "error": "Invalid image file",
                        "details": str(e)
                    }), 400
                
                # Clean up old templates after successful upload
                cleanup_old_templates(template_filename)
                    
            except Exception as e:
                if os.path.exists(template_path):
                    os.remove(template_path)
                app.logger.error(f"Template processing failed: {str(e)}", exc_info=True)
                if is_first_run:
                    flash(f"Template upload failed: {str(e)}", "error")
                    return redirect(url_for("first_run_setup"))
                return jsonify({
                    "error": "Template upload failed",
                    "details": str(e)
                }), 400

        # Validate we have both dataset and template
        if not config.get("active_dataset"):
            if is_first_run:
                flash("Please select or upload a dataset", "error")
                return redirect(url_for("first_run_setup"))
            return jsonify({"error": "No dataset selected or uploaded"}), 400
            
        if not config.get("active_template"):
            if is_first_run:
                flash("Please select or upload a template", "error")
                return redirect(url_for("first_run_setup"))
            return jsonify({"error": "No template selected or uploaded"}), 400

        # Save configuration with backup
        try:
            if os.path.exists(CONFIG_PATH):
                backup_path = f"{CONFIG_PATH}.bak.{int(time.time())}"
                shutil.copy2(CONFIG_PATH, backup_path)
            
            with open(CONFIG_PATH, "w") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            app.logger.error(f"Config save failed: {str(e)}", exc_info=True)
            if is_first_run:
                flash("Failed to save configuration", "error")
                return redirect(url_for("first_run_setup"))
            return jsonify({
                "error": "Failed to save configuration",
                "details": str(e)
            }), 500

        # Return appropriate response
        if is_first_run or request.form.get("triggeredBy") == "manual":
            return redirect(url_for("index", success="true", _=int(time.time())))
        return jsonify({
            "success": True,
            "message": "Activation successful",
            "active_template": config.get("active_template"),
            "active_dataset": config.get("active_dataset"),
            "timestamp": int(time.time())
        })

    except Exception as e:
        app.logger.error(f"Activation failed: {str(e)}", exc_info=True)
        if request.form.get("triggeredBy") == "first_run":
            flash(f"Setup failed: {str(e)}", "error")
            return redirect(url_for("first_run_setup"))
        return jsonify({
            "error": "Internal server error during activation",
            "details": str(e)
        }), 500

@app.route("/")
def index():
    try:
        config = load_active_config()
        success = request.args.get("success", "false") == "true"
        cache_buster = request.args.get("_", str(int(time.time())))
        
        # First-run detection with more robust checks
        if not config:
            config = {"active_dataset": None, "active_template": None}
            
        required_files_exist = (
            config.get("active_dataset") and 
            config.get("active_template") and
            os.path.exists(os.path.join(DATASET_DIR, config["active_dataset"])) and
            os.path.exists(os.path.join(TEMPLATE_DIR, config["active_template"]))
        )

        if not required_files_exist:
            return redirect(url_for("first_run_setup"))
            
        # Get available files with error handling
        try:
            dataset_files = [f for f in sorted(os.listdir(DATASET_DIR), reverse=True) 
                          if f.endswith('.csv')]
            template_files = [f for f in sorted(os.listdir(TEMPLATE_DIR), reverse=True)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        except FileNotFoundError:
            os.makedirs(DATASET_DIR, exist_ok=True)
            os.makedirs(TEMPLATE_DIR, exist_ok=True)
            dataset_files = []
            template_files = []

        # Validate and repair config if needed
        config_changed = False
        
        # Check dataset
        if not os.path.exists(os.path.join(DATASET_DIR, config["active_dataset"])):
            flash("Active dataset not found", "error")
            config["active_dataset"] = dataset_files[0] if dataset_files else None
            config_changed = True
            
        # Check template
        if not os.path.exists(os.path.join(TEMPLATE_DIR, config["active_template"])):
            flash("Active template not found", "error")
            config["active_template"] = template_files[0] if template_files else None
            config_changed = True
            
        # Save config if changed
        if config_changed:
            try:
                with open(CONFIG_PATH, "w") as f:
                    json.dump(config, f)
            except Exception as e:
                app.logger.error(f"Failed to save config: {str(e)}")
                flash("Failed to save configuration", "error")

        # Final verification before rendering
        if not config.get("active_dataset") or not config.get("active_template"):
            return redirect(url_for("first_run_setup"))
            
        # Load dataset with error handling
        try:
            dataset_path = os.path.join(DATASET_DIR, config["active_dataset"])
            df = pd.read_csv(dataset_path)
            
            # Validate required columns
            required_columns = {'ID', 'Name', 'Position', 'Company'}
            if not required_columns.issubset(df.columns):
                flash("Dataset missing required columns", "error")
                return redirect(url_for("first_run_setup"))
                
            update_encrypted_qr_column(df)
            generate_qrs(df)
            
        except Exception as e:
            app.logger.error(f"Dataset processing failed: {str(e)}")
            flash("Failed to process dataset", "error")
            return redirect(url_for("first_run_setup"))

        return render_template(
            "index.html",
            config=config,
            files=sorted(os.listdir(QR_FOLDER), reverse=True) if os.path.exists(QR_FOLDER) else [],
            success=success,
            active_template=config.get("active_template"),
            active_dataset=config.get("active_dataset"),
            dataset_files=dataset_files,
            template_files=template_files,
            cache_buster=cache_buster
        )
        
    except Exception as e:
        app.logger.error(f"Index route failed: {str(e)}", exc_info=True)
        return render_template("error.html", error="System error - please try again"), 500

@app.route("/download/<filename>")
def download_qr(filename):
    return send_from_directory(QR_FOLDER, filename, as_attachment=True)

def parse_qr_content(qr_content):
    try:
        decrypted = fernet.decrypt(qr_content.encode()).decode()
    except:
        return None
    try:
        if '=' in decrypted and ';' in decrypted:
            parts = dict(item.split('=', 1) for item in decrypted.split(';') if '=' in item)
            return parts.get('id', '').strip()
    except:
        return None
    return None

def find_employee_by_id(id_value):
    df, _ = get_active_dataset()
    exact_match = df[df['ID'].astype(str) == id_value]
    if not exact_match.empty:
        return exact_match.iloc[0]
    padded_id = id_value.zfill(3)
    padded_match = df[df['ID'].astype(str) == padded_id]
    if not padded_match.empty:
        return padded_match.iloc[0]
    stripped_id = id_value.lstrip('0')
    if stripped_id:
        stripped_match = df[df['ID'].astype(str) == stripped_id]
        if not stripped_match.empty:
            return stripped_match.iloc[0]
    best_match = None
    best_score = 0
    for _, row in df.iterrows():
        score = SequenceMatcher(None, str(row['ID']), id_value).ratio()
        if score > best_score and score > 0.8:
            best_score = score
            best_match = row
    return best_match

@app.route("/get_data", methods=["POST"])
def get_data():
    data = request.get_json()
    qr_content = data.get("qr_content", "")
    if not qr_content:
        return {"error": "No QR content provided"}, 400
    id_value = parse_qr_content(qr_content)
    if not id_value:
        return {"error": "Could not decrypt or parse ID", "qr_content": qr_content}, 400
    employee = find_employee_by_id(id_value)
    if employee is None:
        return {"error": f"Employee with ID '{id_value}' not found", "searched_id": id_value}, 404
    return {
        "ID": str(employee['ID']),
        "Name": str(employee['Name']),
        "Position": str(employee['Position']),
        "Company": str(employee['Company']),
        "qr_content": qr_content,
        "parsed_id": id_value
    }

@app.route("/employees", methods=["GET"])
def get_employees():
    df, _ = get_active_dataset()
    return jsonify(df.to_dict('records'))

@app.route("/print")
def print_page():
    name = request.args.get("id_name", "")
    position = request.args.get("id_position", "")
    company = request.args.get("id_company", "")
    template_filename = request.args.get("template", "")
    if not template_filename:
        return "Missing template filename", 400
    template_url = url_for('static', filename=f"id_templates/{template_filename}")
    paper_size = request.args.get("paper_size", "A4")
    custom_width = request.args.get("custom_width", "")
    custom_height = request.args.get("custom_height", "")
    return render_template(
        "print.html",
        name=name,
        position=position,
        company=company,
        template_url=template_url,
        paper_size=paper_size,
        custom_width=custom_width,
        custom_height=custom_height,
    )

@app.route("/scan_camera")
def scan_camera():
    return render_template("camera_scanner.html")

@app.route("/preview_qr/<employee_id>")
def preview_qr(employee_id):
    df, template_filename = get_active_dataset()
    normalized_id = str(employee_id).zfill(3)
    employee = df[df['ID'].astype(str).str.zfill(3) == normalized_id]
    if employee.empty:
        return {"error": "Employee not found"}, 404
    emp = employee.iloc[0]
    encrypted_data = emp['EncryptedQR']
    qr_img = qrcode.make(encrypted_data).resize((150, 150))
    template_path = os.path.join(TEMPLATE_DIR, template_filename)
    if not os.path.exists(template_path):
        return {"error": "Template not found"}, 500
    id_template = Image.open(template_path).convert("RGBA")
    id_template.paste(qr_img, (400, 100))
    draw = ImageDraw.Draw(id_template)
    try:
        font_path = os.path.join(BASE_DIR, "static", "fonts", "arial.ttf")
        font = ImageFont.truetype(font_path, 18)
    except:
        font = ImageFont.load_default()
    draw.text((50, 100), f"Name: {emp['Name']}", fill="black", font=font)
    draw.text((50, 140), f"Position: {emp['Position']}", fill="black", font=font)
    draw.text((50, 180), f"Company: {emp['Company']}", fill="black", font=font)
    buffer = BytesIO()
    id_template.save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(buffer, mimetype="image/png")

@app.route("/print_id", methods=["POST"])
def print_id():
    data = request.get_json()
    employee_id = data.get("id", "")
    df, template_filename = get_active_dataset()
    normalized_id = str(employee_id).zfill(3)
    employee = df[df['ID'].astype(str).str.zfill(3) == normalized_id]
    if employee.empty:
        return {"error": "Employee not found"}, 404
    emp = employee.iloc[0]
    template_path = os.path.join(TEMPLATE_DIR, template_filename)
    if not os.path.exists(template_path):
        return {"error": "Template not found"}, 500
    
    try:
        # Generate the ID image
        id_template = Image.open(template_path).convert("RGBA").resize((1013, 638))
        qr_img = qrcode.make(emp['EncryptedQR']).resize((150, 150))
        id_template.paste(qr_img, (400, 80))
        draw = ImageDraw.Draw(id_template)
        
        try:
            font_path = os.path.join(BASE_DIR, "static", "fonts", "arial.ttf")
            font = ImageFont.truetype(font_path, 24)
        except:
            font = ImageFont.load_default()
            
        fields = [
            ("Name", emp.get("Name", "N/A"), (50, 40)),
            ("Position", emp.get("Position", "N/A"), (50, 80)),
            ("Company", emp.get("Company", "N/A"), (50, 120)),
        ]
        
        for label, value, (x, y) in fields:
            text = f"{label}: {value}"
            draw.text((x, y), text, fill="black", font=font)
            text_width = draw.textlength(text, font=font)
            underline_y = y + font.size + 2
            draw.line((x, underline_y, x + text_width, underline_y), fill="black", width=1)

        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"id_{employee_id}_{timestamp}.pdf"
        output_path = os.path.join(DATA_DIR, output_filename)

        # Create PDF directly
        c = canvas.Canvas(output_path, pagesize=(1013, 638))
        img_buffer = BytesIO()
        id_template.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        c.drawImage(ImageReader(img_buffer), 0, 0, width=1013, height=638)
        c.save()

        # Improved printer detection
        printer_detected = False
        printer_name = None
        
        if platform.system() == "Windows":
            try:
                printers = win32print.EnumPrinters(win32print.PRINTER_ENUM_LOCAL | win32print.PRINTER_ENUM_CONNECTIONS)
                if printers:
                    printer_detected = True
                    printer_name = printers[0][2]  # Default to first printer
            except Exception as e:
                app.logger.error(f"Windows printer detection failed: {str(e)}")
        else:
            try:
                # Try CUPS first
                result = subprocess.run(["lpstat", "-p"], capture_output=True, text=True)
                if "enabled" in result.stdout.lower():
                    printer_detected = True
                    # Extract printer name from output
                    lines = result.stdout.split('\n')
                    if lines and ' ' in lines[0]:
                        printer_name = lines[0].split(' ')[1]
            except FileNotFoundError:
                try:
                    # Fall back to LPR
                    result = subprocess.run(["lpstat", "-a"], capture_output=True, text=True)
                    if result.stdout.strip():
                        printer_detected = True
                except:
                    pass

        if not printer_detected:
            return {
                "success": False,
                "error": "No printer detected",
                "download_url": url_for('download_file', filename=output_filename, _external=True),
                "message": "Document saved and available for download"
            }, 503

        # Improved printing function
        print_success = False
        print_error = None
        
        try:
            if platform.system() == "Windows":
                if printer_name:
                    win32print.SetDefaultPrinter(printer_name)
                os.startfile(output_path, "print")
                print_success = True
            else:
                print_cmd = ["lp"]
                if printer_name:
                    print_cmd.extend(["-d", printer_name])
                print_cmd.append(output_path)
                
                try:
                    subprocess.run(print_cmd, check=True, timeout=10)
                    print_success = True
                except subprocess.TimeoutExpired:
                    print_error = "Print job timed out"
                except subprocess.CalledProcessError as e:
                    print_error = f"Print command failed with code {e.returncode}"
        except Exception as e:
            print_error = str(e)

        if not print_success:
            return {
                "success": False,
                "error": f"Printing failed: {print_error or 'Unknown error'}",
                "download_url": url_for('download_file', filename=output_filename, _external=True),
                "message": "Document saved and available for download"
            }

        return {
            "success": True,
            "message": f"Print job sent to {printer_name or 'default printer'} for ID: {employee_id}",
            "download_url": url_for('download_file', filename=output_filename, _external=True)
        }
        
    except Exception as e:
        app.logger.error(f"Printing failed: {str(e)}", exc_info=True)
        return {"error": f"Unexpected error during printing: {str(e)}"}, 500

@app.route("/print_image_direct", methods=["POST"])
def print_image_direct():
    import base64
    try:
        data = request.get_json()
        image_data = data.get("image_base64", "")
        paper_size = data.get("paper_size", "custom")
        custom_width = safe_float(data.get("custom_width"), 3.375)
        custom_height = safe_float(data.get("custom_height"), 2.125)
        redirect_to_preview = data.get("redirect", False)

        if not image_data.startswith("data:image/png;base64,"):
            return {"error": "Invalid image data format"}, 400

        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"print_{timestamp}.png"
        output_path = os.path.join(DATA_DIR, output_filename)

        # Process image
        base64_data = image_data.split(",")[1]
        img_bytes = base64.b64decode(base64_data)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        
        # Calculate dimensions and save
        dpi = 300
        paper_sizes = {
            "a4": (8.27, 11.69), "letter": (8.5, 11), "legal": (8.5, 14),
            "a3": (11.7, 16.5), "a5": (5.8, 8.3), "a6": (4.1, 5.8),
            "a7": (2.9, 3.9), "b5": (7.2, 10.1), "b4": (9.8, 13.9),
            "4x6": (4, 6), "5x7": (5, 7), "5x8": (5, 8), "9x13": (3.5, 5)
        }
        
        if paper_size.lower() != "custom":
            custom_width, custom_height = paper_sizes.get(paper_size.lower(), (3.375, 2.125))

        img.resize((int(custom_width * dpi), int(custom_height * dpi)), Image.LANCZOS).save(output_path, dpi=(dpi, dpi))

        if redirect_to_preview:
            return jsonify({
                "success": True,
                "preview_url": url_for('show_preview', session_id=timestamp, w=custom_width, h=custom_height)
            })

        # Printer detection (same improved version as above)
        printer_detected = False
        printer_name = None
        
        if platform.system() == "Windows":
            try:
                printers = win32print.EnumPrinters(win32print.PRINTER_ENUM_LOCAL | win32print.PRINTER_ENUM_CONNECTIONS)
                if printers:
                    printer_detected = True
                    printer_name = printers[0][2]
            except Exception as e:
                app.logger.error(f"Windows printer detection failed: {str(e)}")
        else:
            try:
                result = subprocess.run(["lpstat", "-p"], capture_output=True, text=True)
                if "enabled" in result.stdout.lower():
                    printer_detected = True
                    lines = result.stdout.split('\n')
                    if lines and ' ' in lines[0]:
                        printer_name = lines[0].split(' ')[1]
            except FileNotFoundError:
                try:
                    result = subprocess.run(["lpstat", "-a"], capture_output=True, text=True)
                    if result.stdout.strip():
                        printer_detected = True
                except:
                    pass

        if not printer_detected:
            return {
                "success": False,
                "error": "No printer detected",
                "download_url": url_for('download_file', filename=output_filename, _external=True),
                "message": "Document saved and available for download"
            }, 503

        # Printing (same improved version as above)
        print_success = False
        print_error = None
        
        try:
            if platform.system() == "Windows":
                if printer_name:
                    win32print.SetDefaultPrinter(printer_name)
                subprocess.run(["mspaint", "/pt", output_path], check=True, timeout=10)
                print_success = True
            else:
                print_cmd = ["lp"]
                if printer_name:
                    print_cmd.extend(["-d", printer_name])
                print_cmd.append(output_path)
                
                try:
                    subprocess.run(print_cmd, check=True, timeout=10)
                    print_success = True
                except subprocess.TimeoutExpired:
                    print_error = "Print job timed out"
                except subprocess.CalledProcessError as e:
                    print_error = f"Print command failed with code {e.returncode}"
        except Exception as e:
            print_error = str(e)

        if not print_success:
            return {
                "success": False,
                "error": f"Printing failed: {print_error or 'Unknown error'}",
                "download_url": url_for('download_file', filename=output_filename, _external=True),
                "message": "Document saved and available for download"
            }

        return {
            "success": True,
            "message": f"Print job sent to {printer_name or 'default printer'}",
            "download_url": url_for('download_file', filename=output_filename, _external=True)
        }
        
    except Exception as e:
        app.logger.error(f"Direct print failed: {str(e)}", exc_info=True)
        return {"error": f"Unexpected error: {str(e)}"}, 500
    
@app.route("/show_preview/<session_id>")
def show_preview(session_id):
    w = request.args.get("w", "8.27")
    h = request.args.get("h", "11.69")
    return render_template("preview.html", session_id=session_id, w=w, h=h)