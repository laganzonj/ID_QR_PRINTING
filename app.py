from flask import Flask, render_template, send_from_directory, jsonify, request, redirect, url_for, send_file
import pandas as pd
import qrcode
import os
import re
import json
from difflib import SequenceMatcher
from cryptography.fernet import Fernet
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from werkzeug.utils import secure_filename
import threading
import webview
import platform
from tempfile import NamedTemporaryFile
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import subprocess
import traceback
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
BASE_DIR = os.getenv('BASE_DIR', os.path.dirname(os.path.abspath(__file__)))

QR_FOLDER = os.path.join(BASE_DIR, "static", "qr_codes")
DATASET_DIR = os.path.join(BASE_DIR, "participant_list")
TEMPLATE_DIR = os.path.join(BASE_DIR, "static", "id_templates")
CONFIG_PATH = os.path.join(BASE_DIR, "active_config.json")

os.makedirs(QR_FOLDER, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

app.secret_key = os.getenv('SECRET_KEY')

with open("qr_key.key", "rb") as f:
    fernet = Fernet(f.read())

def load_active_config():
    if not os.path.exists(CONFIG_PATH):
        default_config = {
            "active_dataset": "sample_employees_dataset.csv",
            "active_template": "template1.png"
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(default_config, f)
        return default_config
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def get_active_dataset():
    config = load_active_config()
    dataset_path = os.path.join(DATASET_DIR, config["active_dataset"])
    return pd.read_csv(dataset_path), config["active_template"]

def update_encrypted_qr_column(df):
    if 'EncryptedQR' not in df.columns:
        df['EncryptedQR'] = df.apply(lambda row: fernet.encrypt(
            f"id={str(row['ID']).zfill(3)};name={row['Name']};position={row['Position']};company={row['Company']}".encode()
        ).decode(), axis=1)
        config = load_active_config()
        dataset_path = os.path.join(DATASET_DIR, config["active_dataset"])
        df.to_csv(dataset_path, index=False)

def generate_qrs(df):
    existing_ids = set()
    for _, row in df.iterrows():
        filename = f"{row['ID']}.png"
        file_path = os.path.join(QR_FOLDER, filename)
        existing_ids.add(filename)
        if not os.path.exists(file_path):
            qr_img = qrcode.make(row['EncryptedQR'])
            qr_img.save(file_path)
    for f in os.listdir(QR_FOLDER):
        if f not in existing_ids:
            os.remove(os.path.join(QR_FOLDER, f))

def safe_float(val, default):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

@app.route("/activate", methods=["POST"])
def activate():
    import time
    dataset_file = request.files.get("dataset_file")
    template_file = request.files.get("template_file")
    config = load_active_config()
    timestamp = str(int(time.time()))
    if dataset_file:
        dataset_filename = timestamp + "_" + secure_filename(dataset_file.filename)
        dataset_path = os.path.join(DATASET_DIR, dataset_filename)
        dataset_file.save(dataset_path)
        config["active_dataset"] = dataset_filename
    else:
        dataset_path = os.path.join(DATASET_DIR, config["active_dataset"])
    if template_file:
        template_filename = timestamp + "_" + secure_filename(template_file.filename)
        template_path = os.path.join(TEMPLATE_DIR, template_filename)
        template_file.save(template_path)
        config["active_template"] = template_filename
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f)
    df = pd.read_csv(dataset_path)
    required_cols = {'ID', 'Name', 'Position', 'Company'}
    if not required_cols.issubset(df.columns):
        return "Dataset is missing required columns: ID, Name, Position, Company", 400
    if 'EncryptedQR' not in df.columns:
        df['EncryptedQR'] = df.apply(lambda row: fernet.encrypt(
            f"id={str(row['ID']).zfill(3)};name={row['Name']};position={row['Position']};company={row['Company']}".encode()
        ).decode(), axis=1)
        df.to_csv(dataset_path, index=False)
    if dataset_file:
        for f in os.listdir(QR_FOLDER):
            os.remove(os.path.join(QR_FOLDER, f))
        generate_qrs(df)
    if request.form.get("triggeredBy") == "manual":
        return redirect(url_for("index", success="true"))
    return redirect(url_for("index"))

@app.route("/")
def index():
    config = load_active_config()
    dataset_path = os.path.join(DATASET_DIR, config["active_dataset"])
    df = pd.read_csv(dataset_path)
    update_encrypted_qr_column(df)
    generate_qrs(df)
    success = request.args.get("success", "false") == "true"
    return render_template(
        "index.html",
        files=os.listdir(QR_FOLDER),
        success=success,
        active_template=config["active_template"],
        active_dataset=config["active_dataset"]
    )

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
    with NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
        id_template.save(tmp_img.name)
        img_path = tmp_img.name
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf_path = tmp_pdf.name
        c = canvas.Canvas(pdf_path, pagesize=(1013, 638))
        c.drawImage(ImageReader(img_path), 0, 0, width=1013, height=638)
        c.save()
    try:
        if platform.system() == "Windows":
            os.startfile(pdf_path, "print")
        else:
            subprocess.run(["lp", pdf_path])
    except Exception as e:
        return {"error": f"Failed to print: {str(e)}"}, 500
    return {"success": True, "message": f"Print job sent for ID: {employee_id}"}

@app.route("/print_image_direct", methods=["POST"])
def print_image_direct():
    try:
        data = request.get_json()
        image_data = data.get("image_base64", "")
        paper_size = data.get("paper_size", "custom")
        custom_width = safe_float(data.get("custom_width"), 3.375)
        custom_height = safe_float(data.get("custom_height"), 2.125)
        redirect_to_preview = data.get("redirect", False)

        if not image_data.startswith("data:image/png;base64,"):
            return {"error": "Invalid image data format"}, 400

        import base64
        import io
        import uuid
        from tempfile import NamedTemporaryFile
        from PIL import Image

        dpi = 300

        paper_sizes = {
            "a4": (8.27, 11.69),
            "letter": (8.5, 11),
            "legal": (8.5, 14),
            "a3": (11.7, 16.5),
            "a5": (5.8, 8.3),
            "a6": (4.1, 5.8),
            "a7": (2.9, 3.9),
            "b5": (7.2, 10.1),
            "b4": (9.8, 13.9),
            "4x6": (4, 6),
            "5x7": (5, 7),
            "5x8": (5, 8),
            "9x13": (3.5, 5)
        }

        if paper_size.lower() != "custom":
            custom_width, custom_height = paper_sizes.get(paper_size.lower(), (3.375, 2.125))

        # Final paper pixel dimensions
        paper_width_px = int(custom_width * dpi)
        paper_height_px = int(custom_height * dpi)

        # Decode and open image
        base64_data = image_data.split(",")[1]
        img_bytes = base64.b64decode(base64_data)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Scale image to fill the entire paper size (stretch)
        resized_img = img.resize((paper_width_px, paper_height_px), Image.LANCZOS)

        from flask import jsonify, url_for

        if redirect_to_preview:
            # Save resized image to static folder
            session_id = str(uuid.uuid4())
            file_path = f"static/preview_{session_id}.png"
            resized_img.save(file_path, dpi=(dpi, dpi))
            return jsonify({
                "success": True,
                "preview_url": url_for('show_preview', session_id=session_id, w=custom_width, h=custom_height)
            })

        # Save for direct printing
        with NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            resized_img.save(tmp_img.name, dpi=(dpi, dpi))
            tmp_img_path = tmp_img.name

        import platform
        import subprocess
        if platform.system() == "Windows":
            subprocess.run(["mspaint", "/pt", tmp_img_path], check=True)
        elif platform.system() == "Darwin":
            subprocess.run(["lp", tmp_img_path], check=True)
        else:
            subprocess.run(["lpr", tmp_img_path], check=True)

        return {"success": True, "message": "Image sent to printer"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Unexpected error: {str(e)}"}, 500

@app.route("/show_preview/<session_id>")
def show_preview(session_id):
    w = request.args.get("w", "8.27")
    h = request.args.get("h", "11.69")
    return render_template("preview.html", session_id=session_id, w=w, h=h)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host="0.0.0.0", port=port)


