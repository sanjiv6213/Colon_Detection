from flask import Flask, render_template, request, send_file, flash, redirect
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from PIL import Image
import io

app = Flask(__name__)
app.secret_key = "secret"

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    return image

def segment_colon_tissue(image):
    colon_mask = np.zeros_like(image)
    return colon_mask

def measure_thickness(image):
    thickness = np.random.uniform(10,50)
    return thickness

def generate_pdf_report(thickness, tissue_image):
    filename = 'report.pdf'
    c = canvas.Canvas(filename, pagesize=letter)
    c.drawString(100, 750, "Colon Thickness Report")
    c.drawString(100, 700, "Thickness of Colon Tissue: {} units".format(thickness))

    # Convert NumPy array to PIL image
    pil_image = Image.fromarray(cv2.cvtColor(tissue_image, cv2.COLOR_BGR2RGB))

    # Save PIL image to bytes
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)

    # Add the tissue image to the PDF
    tissue_image_width = 3 * inch  # Set the width of the image (adjust as needed)
    tissue_image_height = tissue_image_width * tissue_image.shape[0] / tissue_image.shape[1]
    c.drawImage(ImageReader(buffer), 100, 600, width=tissue_image_width, height=tissue_image_height)

    c.save()
    return filename

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/upload1', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        preprocessed_image = preprocess_image(filepath)
        colon_mask = segment_colon_tissue(preprocessed_image)
        thickness = measure_thickness(colon_mask)
        report_filename = generate_pdf_report(thickness, preprocessed_image)
        return send_file(report_filename, as_attachment=True)
    else:
        flash('Allowed file types are png, jpg, jpeg')
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True, port=3000)
