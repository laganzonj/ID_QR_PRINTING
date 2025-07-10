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
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import base64
from functools import lru_cache
import concurrent.futures
import glob
from typing import Dict, Tuple, Optional, Union
import gc
import resource
import psutil

# Memory utilities will be imported after Flask app initialization

if platform.system() == "Windows":
    import win32print

# Load environment variables
load_dotenv()

# Memory optimization settings
import os
os.environ['OMP_NUM_THREADS'] = '1'  # For numpy/pandas
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['PYTHONHASHSEED'] = '0'  # Deterministic hashing
os.environ['MALLOC_TRIM_THRESHOLD_'] = '100000'  # More aggressive memory trimming

# Configure pandas for low memory usage
import pandas as pd
pd.set_option('mode.chained_assignment', None)
pd.set_option('compute.use_bottleneck', False)
pd.set_option('compute.use_numexpr', False)

# Initialize Flask app
app = Flask(__name__, static_folder='static')
BASE_DIR = os.getenv('BASE_DIR', os.path.dirname(os.path.abspath(__file__)))

# Import memory utilities after Flask app initialization
try:
    from memory_utils import MemoryMonitor, memory_limit_check, optimize_pandas_memory, set_memory_limits, emergency_cleanup
    MEMORY_UTILS_AVAILABLE = True

    # Initialize memory optimization
    optimize_pandas_memory()
    set_memory_limits()

except ImportError:
    MEMORY_UTILS_AVAILABLE = False
    print("Memory utilities not available, using basic memory management")

    # Fallback memory limit setting
    try:
        if platform.system() != "Windows":
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            memory_limit = int(os.getenv('MEMORY_LIMIT_MB', '180')) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, hard))
            print(f"Memory limit set to {memory_limit // (1024*1024)}MB")
    except Exception as e:
        print(f"Could not set memory limit: {e}")

# Configuration constants
CONFIG = {
    'DATA_DIR': os.path.join(BASE_DIR, "data"),
    'QR_FOLDER': 'qr_codes',
    'DATASET_DIR': 'participant_list',
    'TEMPLATE_DIR': 'id_templates',
    'CONFIG_FILE': 'active_config.json',
    'MAX_DATASET_SIZE_MB': 10,
    'MAX_TEMPLATE_SIZE_MB': 5,
    'CLEANUP_INTERVAL_HOURS': 6,
    'FILE_RETENTION_DAYS': 7,
    'DEFAULT_TEMPLATE': 'default_template.png'
}

# Initialize paths
DATA_DIR = os.path.join(BASE_DIR, CONFIG['DATA_DIR'])
os.makedirs(DATA_DIR, exist_ok=True)

QR_FOLDER = os.path.join(DATA_DIR, CONFIG['QR_FOLDER'])
DATASET_DIR = os.path.join(DATA_DIR, CONFIG['DATASET_DIR'])
TEMPLATE_DIR = os.path.join(DATA_DIR, CONFIG['TEMPLATE_DIR'])
CONFIG_PATH = os.path.join(DATA_DIR, CONFIG['CONFIG_FILE'])

STATIC_DIR = os.path.join(BASE_DIR, "static")
STATIC_TEMPLATES = os.path.join(STATIC_DIR, CONFIG['TEMPLATE_DIR'])
STATIC_QR_CODES = os.path.join(STATIC_DIR, CONFIG['QR_FOLDER'])
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Ensure required directories exist
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

# Security configuration
ALLOWED_EXTENSIONS = {
    'dataset': {'csv'},
    'template': {'png', 'jpg', 'jpeg', 'webp'}
}

# Initialize encryption
with open("qr_key.key", "rb") as f:
    fernet = Fernet(f.read())

# Global state for print jobs
print_jobs: Dict[str, dict] = {}

import logging
logging.basicConfig(level=logging.INFO)
app.logger.addHandler(logging.StreamHandler())

# Memory monitoring middleware
@app.before_request
def monitor_memory():
    """Monitor memory usage before each request"""
    try:
        mem = psutil.virtual_memory()
        if mem.percent > 90:
            app.logger.warning(f"High memory usage: {mem.percent}%")
            gc.collect()  # Force garbage collection

        # Skip memory-intensive operations if memory is critically low
        if mem.percent > 95:
            if request.endpoint in ['activate', 'get_active_dataset', 'generate_qrs']:
                return json_response(False, "System memory critically low. Please try again later.", status=503)
    except Exception as e:
        app.logger.error(f"Memory monitoring failed: {e}")

@app.after_request
def cleanup_after_request(response):
    """Cleanup after each request"""
    try:
        # Force garbage collection for memory-intensive endpoints
        if request.endpoint in ['activate', 'print_id', 'print_image_direct']:
            gc.collect()
    except Exception as e:
        app.logger.error(f"Post-request cleanup failed: {e}")
    return response

# --------------------------
# Utility Functions
# --------------------------

def retry_on_memory_error(max_retries=3, delay=1.0):
    """Decorator to retry operations on memory errors with exponential backoff"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except MemoryError as e:
                    if attempt == max_retries - 1:
                        app.logger.error(f"Function {func.__name__} failed after {max_retries} attempts: {e}")
                        raise

                    app.logger.warning(f"Memory error in {func.__name__}, attempt {attempt + 1}/{max_retries}")
                    gc.collect()  # Force garbage collection
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
                except Exception as e:
                    app.logger.error(f"Non-memory error in {func.__name__}: {e}")
                    raise
            return None
        return wrapper
    return decorator

def safe_operation(operation_name: str):
    """Context manager for safe operations with automatic cleanup"""
    class SafeOperationContext:
        def __init__(self, name):
            self.name = name
            self.start_memory = None

        def __enter__(self):
            try:
                self.start_memory = psutil.virtual_memory().percent
                app.logger.info(f"Starting {self.name} (Memory: {self.start_memory:.1f}%)")
            except:
                pass
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            try:
                gc.collect()
                end_memory = psutil.virtual_memory().percent
                app.logger.info(f"Completed {self.name} (Memory: {end_memory:.1f}%)")

                if exc_type is MemoryError:
                    app.logger.error(f"Memory error in {self.name}")
                    return False  # Don't suppress the exception
                elif exc_type is not None:
                    app.logger.error(f"Error in {self.name}: {exc_val}")

            except Exception as cleanup_error:
                app.logger.error(f"Cleanup error for {self.name}: {cleanup_error}")

    return SafeOperationContext(operation_name)

def json_response(success: bool, message: str, data: Optional[dict] = None, status: int = 200) -> Tuple[dict, int]:
    """Standardized JSON response format."""
    return {
        'success': success,
        'message': message,
        'data': data or {},
        'timestamp': datetime.now().isoformat()
    }, status

def allowed_file(filename: str, file_type: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS[file_type]

def atomic_write(filepath: str, data: str) -> None:
    """Atomic file write operation."""
    temp_path = f"{filepath}.tmp"
    with open(temp_path, 'w') as f:
        f.write(data)
    os.replace(temp_path, filepath)

def safe_float(val: Union[str, float, None], default: float) -> float:
    """Safely convert to float with default fallback."""
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default

def validate_template(image_path: str) -> bool:
    """Validate template image meets requirements."""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            return img.size >= (500, 500)
    except Exception:
        return False

# --------------------------
# Core Application Functions
# --------------------------

@app.route("/system_status")
def system_status():
    """Endpoint to check system resource usage"""
    import psutil
    return jsonify({
        "memory": psutil.virtual_memory()._asdict(),
        "cpu": psutil.cpu_percent(),
        "disk": psutil.disk_usage('/')._asdict()
    })

@app.route("/health")
def health_check():
    """Simple health check endpoint"""
    try:
        # Basic system checks
        pd.read_csv(os.path.join(DATASET_DIR, load_active_config().get("active_dataset", "")), nrows=1)
        return "OK", 200
    except Exception as e:
        return str(e), 500

@app.route("/profile")
def profile():
    """Simple memory profile endpoint"""
    try:
        mem = psutil.virtual_memory()
        process = psutil.Process()
        return jsonify({
            "system_memory": {
                "total": f"{mem.total / (1024**3):.2f} GB",
                "available": f"{mem.available / (1024**3):.2f} GB",
                "percent": f"{mem.percent}%"
            },
            "process_memory": {
                "rss": f"{process.memory_info().rss / (1024**2):.2f} MB",
                "vms": f"{process.memory_info().vms / (1024**2):.2f} MB",
                "percent": f"{process.memory_percent():.2f}%"
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def check_upload_availability() -> None:
    """Check if persistent storage is ready for uploads."""
    while not os.path.exists('/opt/render/project/src/data/.ready'):
        time.sleep(1)

def load_active_config() -> dict:
    """Load active configuration with validation."""
    if not os.path.exists(CONFIG_PATH):
        return {}
    
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    
    if not validate_config(config):
        app.logger.warning("Invalid config detected, resetting to defaults")
        return {}
    
    return config

def validate_config(config: dict) -> bool:
    """Validate configuration structure and file existence."""
    required_keys = ['active_dataset', 'active_template']
    if not all(k in config for k in required_keys):
        return False
    
    if not os.path.exists(os.path.join(DATASET_DIR, config['active_dataset'])):
        return False
    
    if not os.path.exists(os.path.join(TEMPLATE_DIR, config['active_template'])):
        return False
    
    return True

# Memory-efficient dataset loading with streaming
@lru_cache(maxsize=8)  # Reduced cache size
def get_active_dataset() -> Tuple[pd.DataFrame, str]:
    """Get active dataset with memory-efficient loading and strict limits"""
    config = load_active_config()
    dataset_path = os.path.join(DATASET_DIR, config["active_dataset"])

    # Check file size first
    file_size = os.path.getsize(dataset_path)
    if file_size > 2 * 1024 * 1024:  # 2MB limit
        raise ValueError(f"Dataset too large: {file_size / (1024*1024):.1f}MB (max 2MB)")

    try:
        # Use optimized dtypes and small chunks
        df = pd.read_csv(
            dataset_path,
            dtype={
                'ID': 'string',
                'Name': 'string',
                'Position': 'string',
                'Company': 'string',
                'EncryptedQR': 'string'
            },
            engine='c',
            low_memory=True
        )

        # Limit number of rows for memory safety
        if len(df) > 1000:
            app.logger.warning(f"Dataset has {len(df)} rows, limiting to 1000")
            df = df.head(1000)

        return df, config["active_template"]

    except Exception as e:
        app.logger.error(f"Failed to load dataset: {e}")
        raise

def get_template_path(filename: str) -> str:
    """Locate template file with fallback logic."""
    # Check persistent storage first
    persistent_path = os.path.join(TEMPLATE_DIR, filename)
    if os.path.exists(persistent_path):
        return persistent_path
        
    # Fall back to static files
    static_path = os.path.join(STATIC_TEMPLATES, filename)
    if os.path.exists(static_path):
        return static_path
        
    raise FileNotFoundError(f"Template {filename} not found")

def get_template_preview_url(template_filename: str) -> str:
    """Generate URL for template preview."""
    persistent_path = os.path.join(TEMPLATE_DIR, template_filename)
    if os.path.exists(persistent_path):
        return url_for('serve_persistent_template', filename=template_filename)
    
    static_path = os.path.join(app.static_folder, 'id_templates', template_filename)
    if os.path.exists(static_path):
        return url_for('static', filename=f'id_templates/{template_filename}')
    
    return url_for('static', filename=f'id_templates/{CONFIG["DEFAULT_TEMPLATE"]}')

app.jinja_env.globals.update(get_template_preview_url=get_template_preview_url)

# --------------------------
# File Management Functions
# --------------------------

def clean_directory(dir_path: str, exclude_files: Optional[set] = None, max_age_days: int = CONFIG['FILE_RETENTION_DAYS']) -> None:
    """Clean directory with age-based deletion."""
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

def cleanup_old_templates(current_template: str) -> None:
    """Remove old template files while preserving current and default."""
    try:
        for filename in os.listdir(TEMPLATE_DIR):
            file_path = os.path.join(TEMPLATE_DIR, filename)
            if (filename == current_template or 
                filename == CONFIG['DEFAULT_TEMPLATE'] or
                not filename.lower().endswith(tuple(ALLOWED_EXTENSIONS['template']))):
                continue
                
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    app.logger.info(f"Cleaned up old template: {filename}")
            except Exception as e:
                app.logger.error(f"Failed to delete old template {filename}: {str(e)}")
    except Exception as e:
        app.logger.error(f"Template cleanup failed: {str(e)}")

def cleanup_temp_files() -> None:
    """Clean up temporary files."""
    temp_files = glob.glob(os.path.join(DATA_DIR, '*.tmp'))
    for file in temp_files:
        try:
            os.unlink(file)
        except Exception as e:
            app.logger.error(f"Failed to delete temp file {file}: {str(e)}")

# Memory limit already set above - removing duplicate


# --------------------------
# QR Code Functions
# --------------------------

def update_encrypted_qr_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all rows have valid encrypted QR data."""
    for index, row in df.iterrows():
        try:
            if 'EncryptedQR' in df.columns:
                fernet.decrypt(row['EncryptedQR'].encode()).decode()
        except:
            # Regenerate if invalid
            df.at[index, 'EncryptedQR'] = fernet.encrypt(
                f"id={str(row['ID']).zfill(3)};name={row['Name']};position={row['Position']};company={row['Company']}".encode()
            ).decode()
    return df

def generate_single_qr(row: pd.Series, qr_folder: str) -> None:
    """Generate QR code for a single row."""
    filename = f"{row['ID']}.png"
    file_path = os.path.join(qr_folder, filename)
    
    try:
        qr_img = qrcode.make(row['EncryptedQR'])
        qr_img.save(file_path)
        with Image.open(file_path) as img:
            img.save(file_path, optimize=True, quality=85)
    except Exception as e:
        app.logger.error(f"Failed to generate QR for ID {row['ID']}: {str(e)}")

def generate_qrs(df: pd.DataFrame) -> None:
    """Generate QR codes with ultra-conservative memory management"""
    try:
        # Check memory before starting
        if not check_memory_available(50):
            app.logger.warning("Insufficient memory for QR generation, skipping")
            return

        current_ids = {f"{row['ID']}.png" for _, row in df.iterrows()}

        # Clean directory only if memory allows
        if check_memory_available(30):
            clean_directory(QR_FOLDER, exclude_files=current_ids)

        # Process one at a time with aggressive memory management
        total_rows = len(df)
        for i, (_, row) in enumerate(df.iterrows()):
            try:
                # Check memory before each QR generation
                if not check_memory_available(20):
                    app.logger.warning(f"Low memory, stopping QR generation at {i}/{total_rows}")
                    break

                generate_single_qr(row, QR_FOLDER)

                # Aggressive cleanup every 3 QRs
                if i % 3 == 0:
                    gc.collect()
                    time.sleep(0.1)  # Brief pause

            except Exception as e:
                app.logger.error(f"Failed to generate QR for ID {row['ID']}: {e}")
                continue

        # Final cleanup
        gc.collect()

    except Exception as e:
        app.logger.error(f"QR generation failed: {e}")
        raise

# --------------------------
# Scheduled Tasks
# --------------------------

def scheduled_cleanup() -> None:
    """Regular cleanup of orphaned files."""
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
            
            clean_directory(QR_FOLDER, exclude_files=active_files)
            
            if config.get("active_template"):
                cleanup_old_templates(config["active_template"])
                
            cleanup_temp_files()
                
        except Exception as e:
            app.logger.error(f"Scheduled cleanup failed: {str(e)}")

if os.getenv("ENABLE_SCHEDULER", "true").lower() == "true":
    try:
        scheduler = BackgroundScheduler(
            job_defaults={
                'misfire_grace_time': 300,
                'coalesce': True,
                'max_instances': 1
            },
            timezone='UTC',
            daemon=False
        )
        scheduler.add_job(func=scheduled_cleanup, trigger="interval", hours=CONFIG['CLEANUP_INTERVAL_HOURS'])
        scheduler.start()
        app.logger.info("Scheduler started successfully")
        atexit.register(lambda: scheduler.shutdown())
    except Exception as e:
        app.logger.error(f"Failed to start scheduler: {str(e)}")

# --------------------------
# Route Handlers
# --------------------------

@app.route("/upload_status")
def upload_status() -> Tuple[dict, int]:
    """Check storage readiness."""
    return jsonify({
        "ready": os.path.exists(os.path.join(DATA_DIR, ".ready")),
        "message": "System ready for uploads" if os.path.exists(os.path.join(DATA_DIR, ".ready")) 
                else "Initializing storage..."
    })

@app.route("/first-run")
def first_run_setup():
    """Initial setup page."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    os.makedirs(QR_FOLDER, exist_ok=True)
    
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
def serve_persistent_template(filename: str):
    """Serve templates from persistent storage."""
    return send_from_directory(TEMPLATE_DIR, filename)

@app.route("/activate", methods=["POST"])
def activate() -> Union[Tuple[dict, int], redirect]:
    """Activate new configuration with enhanced memory optimizations."""
    try:
        # Initial setup with memory check
        if not check_memory_available(150):  # Need at least 150MB free
            return handle_activation_error(
                "System memory low. Please try again later.",
                request.form.get("triggeredBy") == "first_run"
            )

        is_first_run = request.form.get("triggeredBy") == "first_run"
        os.makedirs(DATASET_DIR, exist_ok=True)
        os.makedirs(TEMPLATE_DIR, exist_ok=True)
        os.makedirs(QR_FOLDER, exist_ok=True)

        dataset_file = request.files.get("dataset_file")
        template_file = request.files.get("template_file")
        existing_dataset = request.form.get("existing_dataset")
        existing_template = request.form.get("existing_template")
        config = load_active_config()
        
        # Process dataset with strict memory controls
        if existing_dataset:
            config["active_dataset"] = existing_dataset
        elif dataset_file and dataset_file.filename:
            if not allowed_file(dataset_file.filename, 'dataset'):
                return handle_activation_error(
                    "Invalid file type. Only CSV files are allowed.",
                    is_first_run
                )
                
            # Check memory again right before processing
            if not check_memory_available(120):
                return handle_activation_error(
                    "System memory low during processing. Try a smaller dataset.",
                    is_first_run
                )
                
            random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
            dataset_filename = f"{int(time.time())}_{random_str}_{secure_filename(dataset_file.filename)}"
            dataset_path = os.path.join(DATASET_DIR, dataset_filename)
            
            try:
                # Save file first with size validation
                dataset_file.save(dataset_path)
                
                # Enforce strict size limit (1.5MB for extra safety)
                file_size = os.path.getsize(dataset_path) / (1024 * 1024)
                if file_size > 1.5:
                    os.remove(dataset_path)
                    return handle_activation_error(
                        f"Dataset exceeds 1.5MB limit (was {file_size:.2f}MB)",
                        is_first_run
                    )
                
                # Process in ultra-conservative chunks
                df = process_dataset(dataset_path)
                config["active_dataset"] = dataset_filename
                
                # Clean QR folder with memory awareness
                if check_memory_available(80):
                    clean_directory(QR_FOLDER)
                else:
                    app.logger.warning("Skipping QR folder cleanup due to low memory")
                
                # Generate QR codes with strict memory monitoring
                if not check_memory_available(80):
                    raise MemoryError("Insufficient memory for QR generation")
                
                generate_qrs(df)
                
                # Aggressive cleanup
                del df
                gc.collect()
                time.sleep(0.5)  # Allow memory to settle
                
            except MemoryError:
                if os.path.exists(dataset_path):
                    os.remove(dataset_path)
                return handle_activation_error(
                    "System ran out of memory during processing. Try a smaller file or wait a moment.",
                    is_first_run
                )
            except Exception as e:
                if os.path.exists(dataset_path):
                    os.remove(dataset_path)
                app.logger.error(f"Dataset processing error: {str(e)}", exc_info=True)
                return handle_activation_error(
                    f"Dataset processing failed: {str(e)}",
                    is_first_run
                )

        # Process template with size checks
        if existing_template:
            config["active_template"] = existing_template
        elif template_file and template_file.filename:
            if not allowed_file(template_file.filename, 'template'):
                return handle_activation_error(
                    f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS['template'])}",
                    is_first_run
                )
                
            # Check file size (4MB max for extra safety)
            if len(template_file.read()) > 4 * 1024 * 1024:
                return handle_activation_error(
                    "Template image exceeds 4MB size limit",
                    is_first_run
                )
            template_file.seek(0)
                
            random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
            template_filename = f"{int(time.time())}_{random_str}_{secure_filename(template_file.filename)}"
            template_path = os.path.join(TEMPLATE_DIR, template_filename)
            
            try:
                template_file.save(template_path)
                if not validate_template(template_path):
                    os.remove(template_path)
                    return handle_activation_error(
                        "Image dimensions too small (minimum 500x500 pixels) or invalid image file",
                        is_first_run
                    )
                config["active_template"] = template_filename
                
                # Cleanup old templates if memory allows
                if check_memory_available(50):
                    cleanup_old_templates(template_filename)
                else:
                    app.logger.warning("Skipping template cleanup due to low memory")
                    
            except Exception as e:
                if os.path.exists(template_path):
                    os.remove(template_path)
                return handle_activation_error(
                    f"Template processing failed: {str(e)}", 
                    is_first_run
                )

        # Validate configuration
        if not config.get("active_dataset"):
            return handle_activation_error("No dataset selected or uploaded", is_first_run)
        if not config.get("active_template"):
            return handle_activation_error("No template selected or uploaded", is_first_run)

        # Save configuration with atomic write
        try:
            if os.path.exists(CONFIG_PATH):
                backup_path = f"{CONFIG_PATH}.bak.{int(time.time())}"
                shutil.copy2(CONFIG_PATH, backup_path)
            
            atomic_write(CONFIG_PATH, json.dumps(config, indent=2))
            
            # Force cleanup
            gc.collect()
            time.sleep(0.2)  # Brief pause after file operations
            
        except Exception as e:
            return handle_activation_error(
                f"Failed to save configuration: {str(e)}",
                is_first_run
            )

        # Return response with cache buster
        cache_buster = int(time.time())
        if is_first_run or request.form.get("triggeredBy") == "manual":
            return redirect(url_for("index", success="true", _=cache_buster))
        return json_response(True, "Activation successful", {
            "active_template": config.get("active_template"),
            "active_dataset": config.get("active_dataset"),
            "memory_status": f"{psutil.virtual_memory().available / (1024 * 1024):.1f}MB available"
        })
        
    except MemoryError as e:
        app.logger.error(f"Memory error in activation: {str(e)}", exc_info=True)
        return handle_activation_error(
            "System ran out of memory. Please try again with smaller files.",
            is_first_run
        )
    except Exception as e:
        app.logger.error(f"Unexpected error in activation: {str(e)}", exc_info=True)
        return handle_activation_error(
            f"System error: {str(e)}",
            is_first_run
        )


def check_memory_available(min_mb: int = 100) -> bool:
    """Check if at least min_mb MB of memory is available"""
    try:
        mem = psutil.virtual_memory()
        available_mb = mem.available / (1024 * 1024)
        app.logger.info(f"Available memory: {available_mb:.2f}MB")
        return available_mb >= min_mb
    except Exception as e:
        app.logger.error(f"Memory check failed: {str(e)}")
        return True  # Fallback to assuming memory is available

def handle_activation_error(message: str, is_first_run: bool) -> Union[Tuple[dict, int], redirect]:
    """Handle activation errors consistently."""
    app.logger.error(f"Activation error: {message}")
    if is_first_run:
        flash(message, "error")
        return redirect(url_for("first_run_setup"))
    return json_response(False, message, status=400)

def process_dataset(dataset_path: str) -> pd.DataFrame:
    """Process dataset with ultra-conservative memory management"""
    temp_path = None

    try:
        # Strict file size check (1.5MB max for extra safety)
        file_size = os.path.getsize(dataset_path) / (1024 * 1024)
        if file_size > 1.5:
            os.remove(dataset_path)
            raise ValueError(f"Dataset exceeds 1.5MB limit (was {file_size:.2f}MB)")

        # Check available memory before processing
        if not check_memory_available(100):
            os.remove(dataset_path)
            raise MemoryError("Insufficient memory to process dataset")

        temp_path = f"{dataset_path}.tmp"

        # Process in ultra-small chunks
        try:
            chunks = pd.read_csv(
                dataset_path,
                chunksize=25,  # Even smaller chunks
                dtype={
                    'ID': 'string',
                    'Name': 'string',
                    'Position': 'string',
                    'Company': 'string'
                },
                engine='c',
                low_memory=True
            )

            first_chunk = True
            processed_rows = 0
            max_rows = 500  # Limit total rows

            for chunk in chunks:
                # Memory check before each chunk
                if not check_memory_available(50):
                    app.logger.warning("Low memory during processing, stopping early")
                    break

                # Validate first chunk
                if first_chunk:
                    required_cols = {'ID', 'Name', 'Position', 'Company'}
                    if not required_cols.issubset(chunk.columns):
                        raise ValueError("Missing required columns: " + str(required_cols - set(chunk.columns)))
                    first_chunk = False

                # Limit total rows
                if processed_rows + len(chunk) > max_rows:
                    chunk = chunk.head(max_rows - processed_rows)

                # Process chunk with memory monitoring
                if 'EncryptedQR' not in chunk.columns:
                    chunk['EncryptedQR'] = chunk.apply(generate_qr_row, axis=1)

                # Append to temp file
                chunk.to_csv(temp_path, mode='a', header=not os.path.exists(temp_path), index=False)
                processed_rows += len(chunk)

                # Aggressive cleanup
                del chunk
                gc.collect()
                time.sleep(0.1)  # Brief pause

                if processed_rows >= max_rows:
                    break

            # Replace original with processed file
            if os.path.exists(temp_path):
                os.replace(temp_path, dataset_path)

            # Return minimal sample for validation
            return pd.read_csv(dataset_path, nrows=10, dtype='string')

        except pd.errors.EmptyDataError:
            raise ValueError("Dataset file is empty or corrupted")
        except pd.errors.ParserError as e:
            raise ValueError(f"Dataset parsing failed: {str(e)}")

    except Exception as e:
        # Comprehensive cleanup
        for path in [temp_path, dataset_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

        # Force memory cleanup
        gc.collect()

        if isinstance(e, (MemoryError, ValueError)):
            raise
        else:
            raise ValueError(f"Dataset processing failed: {str(e)}")

def generate_qr_row(row):
    return fernet.encrypt(
        f"id={str(row['ID']).zfill(3)};name={row['Name']};position={row['Position']};company={row['Company']}".encode()
    ).decode()

@app.route("/")
def index():
    """Main application interface."""
    try:
        config = load_active_config()
        success = request.args.get("success", "false") == "true"
        cache_buster = request.args.get("_", str(int(time.time())))

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

        try:
            dataset_files = [
                f for f in sorted(os.listdir(DATASET_DIR), reverse=True) 
                if f.endswith('.csv')
            ]
            template_files = [
                f for f in sorted(os.listdir(TEMPLATE_DIR), reverse=True)
                if f.lower().endswith(tuple(ALLOWED_EXTENSIONS['template']))
            ]
        except FileNotFoundError:
            os.makedirs(DATASET_DIR, exist_ok=True)
            os.makedirs(TEMPLATE_DIR, exist_ok=True)
            dataset_files = []
            template_files = []

        # Validate and repair config
        config_changed = False

        if not os.path.exists(os.path.join(DATASET_DIR, config["active_dataset"])):
            flash("Active dataset not found", "error")
            config["active_dataset"] = dataset_files[0] if dataset_files else None
            config_changed = True

        if not os.path.exists(os.path.join(TEMPLATE_DIR, config["active_template"])):
            flash("Active template not found", "error")
            config["active_template"] = template_files[0] if template_files else None
            config_changed = True

        if config_changed:
            try:
                atomic_write(CONFIG_PATH, json.dumps(config))
            except Exception as e:
                app.logger.error(f"Failed to save config: {str(e)}")
                flash("Failed to save configuration", "error")

        if not config.get("active_dataset") or not config.get("active_template"):
            return redirect(url_for("first_run_setup"))

        # Load dataset with memory-efficient processing
        try:
            # Check memory before loading dataset
            if not check_memory_available(80):
                flash("System memory low. Please wait and try again.", "warning")
                return redirect(url_for("first_run_setup"))

            dataset_path = os.path.join(DATASET_DIR, config["active_dataset"])

            # Load with optimized settings
            df = pd.read_csv(
                dataset_path,
                dtype={
                    'ID': 'string',
                    'Name': 'string',
                    'Position': 'string',
                    'Company': 'string',
                    'EncryptedQR': 'string'
                },
                engine='c',
                low_memory=True
            )

            required_columns = {'ID', 'Name', 'Position', 'Company'}
            if not required_columns.issubset(df.columns):
                flash("Dataset missing required columns", "error")
                return redirect(url_for("first_run_setup"))

            # Limit dataset size for memory safety
            if len(df) > 1000:
                app.logger.warning(f"Dataset has {len(df)} rows, limiting to 1000")
                df = df.head(1000)
                flash(f"Dataset limited to 1000 rows for memory safety", "warning")

            # Process QR codes only if needed and memory allows
            if check_memory_available(60):
                update_encrypted_qr_column(df)
                generate_qrs(df)
            else:
                app.logger.warning("Skipping QR generation due to low memory")
                flash("QR codes will be generated on demand due to memory constraints", "info")

        except MemoryError:
            app.logger.error("Memory error loading dataset")
            flash("Dataset too large for available memory", "error")
            return redirect(url_for("first_run_setup"))
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
def download_qr(filename: str):
    """Download QR code image."""
    return send_from_directory(QR_FOLDER, filename, as_attachment=True)

def parse_qr_content(qr_content: str) -> Optional[str]:
    """Decrypt and parse QR content."""
    try:
        decrypted = fernet.decrypt(qr_content.encode()).decode()
        if '=' in decrypted and ';' in decrypted:
            parts = dict(item.split('=', 1) for item in decrypted.split(';') if '=' in item)
            return parts.get('id', '').strip()
    except:
        return None
    return None

def find_employee_by_id(id_value: str) -> Optional[pd.Series]:
    """Find employee by ID with memory-efficient flexible matching."""
    try:
        df, _ = get_active_dataset()

        # Convert ID column to string once for efficiency
        id_series = df['ID'].astype(str)

        # Try exact match first
        exact_mask = id_series == id_value
        if exact_mask.any():
            return df[exact_mask].iloc[0]

        # Try padded match (e.g. "1" -> "001")
        padded_id = id_value.zfill(3)
        padded_mask = id_series == padded_id
        if padded_mask.any():
            return df[padded_mask].iloc[0]

        # Try stripped match (e.g. "001" -> "1")
        stripped_id = id_value.lstrip('0')
        if stripped_id:
            stripped_mask = id_series == stripped_id
            if stripped_mask.any():
                return df[stripped_mask].iloc[0]

        # Memory-efficient fuzzy matching (limit to first 100 rows)
        search_df = df.head(100) if len(df) > 100 else df
        best_match = None
        best_score = 0

        for _, row in search_df.iterrows():
            score = SequenceMatcher(None, str(row['ID']), id_value).ratio()
            if score > best_score and score > 0.8:
                best_score = score
                best_match = row

        return best_match

    except Exception as e:
        app.logger.error(f"Employee search failed: {e}")
        return None

@app.route("/get_data", methods=["POST"])
def get_data() -> Tuple[dict, int]:
    """Process QR code data."""
    data = request.get_json()
    qr_content = data.get("qr_content", "")
    
    if not qr_content:
        return json_response(False, "No QR content provided", status=400)
        
    id_value = parse_qr_content(qr_content)
    if not id_value:
        return json_response(False, "Could not decrypt or parse ID", {"qr_content": qr_content}, 400)
        
    employee = find_employee_by_id(id_value)
    if employee is None:
        return json_response(False, f"Employee with ID '{id_value}' not found", {"searched_id": id_value}, 404)
        
    return json_response(True, "Employee data retrieved", {
        "ID": str(employee['ID']),
        "Name": str(employee['Name']),
        "Position": str(employee['Position']),
        "Company": str(employee['Company']),
        "qr_content": qr_content,
        "parsed_id": id_value
    })

@app.route("/employees", methods=["GET"])
def get_employees() -> Tuple[dict, int]:
    """Get all employee data."""
    df, _ = get_active_dataset()
    return json_response(True, "Employee data retrieved", {"employees": df.to_dict('records')})

@app.route("/print")
def print_page():
    """Print preview page."""
    name = request.args.get("id_name", "")
    position = request.args.get("id_position", "")
    company = request.args.get("id_company", "")
    template_filename = request.args.get("template", "")
    
    if not template_filename:
        return json_response(False, "Missing template filename", status=400)
        
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
    """Camera scanning interface."""
    return render_template("camera_scanner.html")

@app.route("/preview_qr/<employee_id>")
def preview_qr(employee_id: str):
    """Preview QR code for an employee."""
    df, template_filename = get_active_dataset()
    normalized_id = str(employee_id).zfill(3)
    employee = df[df['ID'].astype(str).str.zfill(3) == normalized_id]
    
    if employee.empty:
        return json_response(False, "Employee not found", status=404)
        
    emp = employee.iloc[0]
    template_path = os.path.join(TEMPLATE_DIR, template_filename)
    
    if not os.path.exists(template_path):
        return json_response(False, "Template not found", status=500)
    
    try:
        # Generate ID image
        id_template = Image.open(template_path).convert("RGBA")
        qr_img = qrcode.make(emp['EncryptedQR']).resize((150, 150))
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
        
    except Exception as e:
        app.logger.error(f"Preview generation failed: {str(e)}")
        return json_response(False, f"Preview generation failed: {str(e)}", status=500)

def track_print_job(job_id: str, filename: str) -> None:
    """Track print job status."""
    print_jobs[job_id] = {
        'filename': filename,
        'status': 'queued',
        'timestamp': datetime.now().isoformat()
    }

@app.route('/print_status/<job_id>')
def print_status(job_id: str) -> Tuple[dict, int]:
    """Check print job status."""
    job = print_jobs.get(job_id, {})
    return json_response(bool(job), "Print job status", {"job": job})

def generate_id_image(employee: pd.Series, template_path: str) -> Image.Image:
    """Generate ID card image."""
    id_template = Image.open(template_path).convert("RGBA").resize((1013, 638))
    qr_img = qrcode.make(employee['EncryptedQR']).resize((150, 150))
    id_template.paste(qr_img, (400, 80))
    
    draw = ImageDraw.Draw(id_template)
    try:
        font_path = os.path.join(BASE_DIR, "static", "fonts", "arial.ttf")
        font = ImageFont.truetype(font_path, 24)
    except:
        font = ImageFont.load_default()
        
    fields = [
        ("Name", employee.get("Name", "N/A"), (50, 40)),
        ("Position", employee.get("Position", "N/A"), (50, 80)),
        ("Company", employee.get("Company", "N/A"), (50, 120)),
    ]
    
    for label, value, (x, y) in fields:
        text = f"{label}: {value}"
        draw.text((x, y), text, fill="black", font=font)
        text_width = draw.textlength(text, font=font)
        underline_y = y + font.size + 2
        draw.line((x, underline_y, x + text_width, underline_y), fill="black", width=1)
    
    return id_template

@app.route("/print_id", methods=["POST"])
def print_id() -> Tuple[dict, int]:
    """Print ID card."""
    data = request.get_json()
    employee_id = data.get("id", "")
    
    df, template_filename = get_active_dataset()
    normalized_id = str(employee_id).zfill(3)
    employee = df[df['ID'].astype(str).str.zfill(3) == normalized_id]
    
    if employee.empty:
        return json_response(False, "Employee not found", status=404)
        
    emp = employee.iloc[0]
    template_path = os.path.join(TEMPLATE_DIR, template_filename)
    
    if not os.path.exists(template_path):
        return json_response(False, "Template not found", status=500)
    
    try:
        # Generate ID image
        id_template = generate_id_image(emp, template_path)
        
        # Create PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"id_{employee_id}_{timestamp}.pdf"
        output_path = os.path.join(DATA_DIR, output_filename)
        
        c = canvas.Canvas(output_path, pagesize=(1013, 638))
        img_buffer = BytesIO()
        id_template.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        c.drawImage(ImageReader(img_buffer), 0, 0, width=1013, height=638)
        c.save()
        
        # Track print job
        job_id = f"job_{timestamp}"
        track_print_job(job_id, output_filename)
        
        # Handle printing
        result = handle_printing(output_path, output_filename, f"ID: {employee_id}")
        print_jobs[job_id]['status'] = 'completed' if result[0]['success'] else 'failed'
        
        return result
        
    except Exception as e:
        app.logger.error(f"Printing failed: {str(e)}", exc_info=True)
        return json_response(False, f"Unexpected error during printing: {str(e)}", status=500)

@app.route("/print_image_direct", methods=["POST"])
def print_image_direct() -> Tuple[dict, int]:
    """Direct image printing."""
    try:
        data = request.get_json()
        image_data = data.get("image_base64", "")
        paper_size = data.get("paper_size", "custom")
        custom_width = safe_float(data.get("custom_width"), 3.375)
        custom_height = safe_float(data.get("custom_height"), 2.125)
        redirect_to_preview = data.get("redirect", False)

        if not image_data.startswith("data:image/png;base64,"):
            return json_response(False, "Invalid image data format", status=400)

        # Process image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"print_{timestamp}.png"
        output_path = os.path.join(DATA_DIR, output_filename)
        
        base64_data = image_data.split(",")[1]
        img_bytes = base64.b64decode(base64_data)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        
        # Handle paper sizes
        paper_sizes = {
            "a4": (8.27, 11.69), "letter": (8.5, 11), "legal": (8.5, 14),
            "a3": (11.7, 16.5), "a5": (5.8, 8.3), "a6": (4.1, 5.8),
            "a7": (2.9, 3.9), "b5": (7.2, 10.1), "b4": (9.8, 13.9),
            "4x6": (4, 6), "5x7": (5, 7), "5x8": (5, 8), "9x13": (3.5, 5)
        }
        
        if paper_size.lower() != "custom":
            custom_width, custom_height = paper_sizes.get(paper_size.lower(), (3.375, 2.125))

        # Save image with proper DPI
        dpi = 300
        img.resize((int(custom_width * dpi), int(custom_height * dpi)), Image.LANCZOS).save(output_path, dpi=(dpi, dpi))

        if redirect_to_preview:
            return json_response(True, "Preview generated", {
                "preview_url": url_for('show_preview', session_id=timestamp, w=custom_width, h=custom_height)
            })

        # Track print job
        job_id = f"img_{timestamp}"
        track_print_job(job_id, output_filename)
        
        # Handle printing
        result = handle_printing(output_path, output_filename, "image")
        print_jobs[job_id]['status'] = 'completed' if result[0]['success'] else 'failed'
        
        return result
        
    except Exception as e:
        app.logger.error(f"Direct print failed: {str(e)}", exc_info=True)
        return json_response(False, f"Unexpected error: {str(e)}", status=500)

def detect_printers() -> Tuple[bool, str, str]:
    """Enhanced cross-platform printer detection with multiple fallbacks"""
    printer_detected = False
    printer_name = None
    print_error = None

    system = platform.system()

    if system == "Windows":
        try:
            # Try win32print first
            if 'win32print' in sys.modules:
                printers = win32print.EnumPrinters(win32print.PRINTER_ENUM_LOCAL | win32print.PRINTER_ENUM_CONNECTIONS)
                if printers:
                    printer_detected = True
                    printer_name = printers[0][2]
                    return printer_detected, printer_name, print_error
        except Exception as e:
            app.logger.warning(f"win32print failed: {e}")

        # Fallback to PowerShell
        try:
            result = subprocess.run([
                "powershell", "-Command",
                "Get-Printer | Where-Object {$_.PrinterStatus -eq 'Normal'} | Select-Object -First 1 -ExpandProperty Name"
            ], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                printer_detected = True
                printer_name = result.stdout.strip()
                return printer_detected, printer_name, print_error
        except Exception as e:
            app.logger.warning(f"PowerShell printer detection failed: {e}")

        # Final fallback to wmic
        try:
            result = subprocess.run([
                "wmic", "printer", "where", "WorkOffline=FALSE", "get", "Name", "/format:list"
            ], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Name=') and line.strip() != 'Name=':
                        printer_detected = True
                        printer_name = line.split('=', 1)[1].strip()
                        break
        except Exception as e:
            print_error = f"All Windows printer detection methods failed: {e}"

    elif system == "Darwin":  # macOS
        try:
            # Use lpstat for macOS
            result = subprocess.run(["lpstat", "-p"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'idle' in line.lower() or 'enabled' in line.lower():
                        printer_detected = True
                        parts = line.split()
                        if len(parts) > 1:
                            printer_name = parts[1]
                        break
        except Exception as e:
            app.logger.warning(f"macOS lpstat failed: {e}")

        # Fallback to system_profiler
        if not printer_detected:
            try:
                result = subprocess.run([
                    "system_profiler", "SPPrintersDataType", "-xml"
                ], capture_output=True, text=True, timeout=10)
                if result.returncode == 0 and "printer" in result.stdout.lower():
                    printer_detected = True
                    printer_name = "default"
            except Exception as e:
                print_error = f"macOS printer detection failed: {e}"

    else:  # Linux and other Unix-like systems
        # Try lpstat first
        try:
            result = subprocess.run(["lpstat", "-p"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'enabled' in line.lower() or 'idle' in line.lower():
                        printer_detected = True
                        parts = line.split()
                        if len(parts) > 1:
                            printer_name = parts[1]
                        break
        except FileNotFoundError:
            app.logger.warning("lpstat not found")
        except Exception as e:
            app.logger.warning(f"lpstat failed: {e}")

        # Fallback to lpstat -a
        if not printer_detected:
            try:
                result = subprocess.run(["lpstat", "-a"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    lines = result.stdout.strip().split('\n')
                    if lines and lines[0]:
                        printer_detected = True
                        printer_name = lines[0].split()[0]
            except Exception as e:
                app.logger.warning(f"lpstat -a failed: {e}")

        # Try lpadmin as final fallback
        if not printer_detected:
            try:
                result = subprocess.run(["lpadmin", "-p"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    printer_detected = True
                    printer_name = "default"
            except Exception as e:
                print_error = f"Linux printer detection failed: {e}"

    return printer_detected, printer_name, print_error

def handle_printing(file_path: str, output_filename: str, job_description: str = "") -> Tuple[dict, int]:
    """Enhanced cross-platform printing with robust fallbacks"""

    # Detect printers with enhanced detection
    printer_detected, printer_name, print_error = detect_printers()

    if not printer_detected:
        return json_response(False, print_error or "No printer detected", {
            "download_url": url_for('download_file', filename=output_filename, _external=True),
            "message": "Document saved and available for download"
        }, 503)

    # Enhanced printing implementation
    print_success = False
    system = platform.system()

    try:
        if system == "Windows":
            # Try multiple Windows printing methods
            try:
                # Method 1: win32print (if available)
                if 'win32print' in sys.modules and printer_name:
                    win32print.SetDefaultPrinter(printer_name)
                    os.startfile(file_path, "print")
                    print_success = True
            except Exception as e:
                app.logger.warning(f"win32print failed: {e}")

            # Method 2: PowerShell printing
            if not print_success:
                try:
                    ps_cmd = f'Start-Process -FilePath "{file_path}" -Verb Print -WindowStyle Hidden'
                    subprocess.run(["powershell", "-Command", ps_cmd], timeout=15, check=True)
                    print_success = True
                except Exception as e:
                    app.logger.warning(f"PowerShell printing failed: {e}")

            # Method 3: Direct file association
            if not print_success:
                try:
                    os.startfile(file_path, "print")
                    print_success = True
                except Exception as e:
                    print_error = f"Windows printing failed: {e}"

        elif system == "Darwin":  # macOS
            # Try lp command for macOS
            try:
                print_cmd = ["lp"]
                if printer_name and printer_name != "default":
                    print_cmd.extend(["-d", printer_name])
                print_cmd.append(file_path)

                subprocess.run(print_cmd, check=True, timeout=15)
                print_success = True
            except subprocess.CalledProcessError as e:
                print_error = f"macOS lp command failed with code {e.returncode}"
            except subprocess.TimeoutExpired:
                print_error = "macOS print job timed out"
            except Exception as e:
                print_error = f"macOS printing failed: {e}"

        else:  # Linux and other Unix-like systems
            # Try multiple Linux printing methods
            print_commands = [
                ["lp"] + (["-d", printer_name] if printer_name else []) + [file_path],
                ["lpr"] + (["-P", printer_name] if printer_name else []) + [file_path]
            ]

            for cmd in print_commands:
                try:
                    subprocess.run(cmd, check=True, timeout=15)
                    print_success = True
                    break
                except FileNotFoundError:
                    continue
                except subprocess.CalledProcessError as e:
                    app.logger.warning(f"Print command {cmd[0]} failed with code {e.returncode}")
                    continue
                except subprocess.TimeoutExpired:
                    app.logger.warning(f"Print command {cmd[0]} timed out")
                    continue
                except Exception as e:
                    app.logger.warning(f"Print command {cmd[0]} failed: {e}")
                    continue

            if not print_success:
                print_error = "All Linux printing methods failed"

    except Exception as e:
        print_error = f"Unexpected printing error: {e}"
        app.logger.error(f"Printing failed: {print_error}")

    if not print_success:
        return json_response(False, print_error or "Unknown printing error", {
            "download_url": url_for('download_file', filename=output_filename, _external=True),
            "message": "Document saved and available for download"
        })

    return json_response(True, f"Print job sent to {printer_name or 'default printer'} for {job_description}", {
        "download_url": url_for('download_file', filename=output_filename, _external=True)
    })
    
@app.route("/show_preview/<session_id>")
def show_preview(session_id: str):
    """Show print preview."""
    w = request.args.get("w", "8.27")
    h = request.args.get("h", "11.69")
    return render_template("preview.html", session_id=session_id, w=w, h=h)

@app.route("/download_file/<filename>")
def download_file(filename: str):
    """Download generated files (PDFs, images, etc.)"""
    try:
        return send_from_directory(DATA_DIR, filename, as_attachment=True)
    except FileNotFoundError:
        return json_response(False, "File not found", status=404)

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return render_template("error.html", error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    app.logger.error(f"500 Error: {str(error)}\n{traceback.format_exc()}")
    return render_template("error.html", error="Internal server error"), 500

@app.errorhandler(MemoryError)
def memory_error(error):
    """Handle memory errors gracefully."""
    app.logger.error(f"Memory Error: {str(error)}")
    gc.collect()  # Force garbage collection
    return render_template("error.html", error="System memory low. Please try again with smaller files."), 503
