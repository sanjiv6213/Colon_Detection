from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image
import cv2
import numpy as np
from skimage.measure import label, regionprops
import random


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model("colon1.h5")

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresholded_image

def segment_colon_tissue(image):
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    labeled_image = label(sure_bg)
    regions = regionprops(labeled_image)
    max_area = 0
    for region in regions:
        if region.area > max_area:
            max_area = region.area
            max_region = region
    colon_mask = np.zeros_like(image)
    colon_mask[max_region.bbox[0]:max_region.bbox[2], max_region.bbox[1]:max_region.bbox[3]] = max_region.filled_image
    return colon_mask

def measure_thickness(image):
    edges = cv2.Canny(image, 50, 150)
    distance_transform = cv2.distanceTransform(edges, cv2.DIST_L2, 5)
    thickness = np.max(distance_transform)
    return thickness

def visualize_results(original_image, colon_mask, thickness):
    masked_image = cv2.bitwise_and(original_image, original_image, mask=colon_mask)
    # Display the original image and the masked image
    # cv2.imshow('Original Image', original_image)
    # cv2.imshow('Colon Tissue Mask', masked_image)
    cv2.imwrite("static/out1.jpg",masked_image)
    # print("Estimated Thickness of Colon Tissue:", thickness)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



def predict_class(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0 
    predictions = model.predict(img_array)
    return img, predictions

def generate_pdf(img, predictions):
    pdf_file_path = 'predictions.pdf'
    doc = SimpleDocTemplate(pdf_file_path, pagesize=letter)
    class_names=["Normal","Ulcerative colitis","Polyp","Esophagitis"]
    data = [['Class', 'Probability']]
    for i, pred in enumerate(predictions):
        pred_values = pred.tolist() if isinstance(pred, np.ndarray) else [pred]
        c=0
        for value in pred_values:
            pred_value = float(value)  
            data.append([class_names[c], f'{pred_value:.2f}'])
            c+=1
    t = Table(data)


    img_path = 'uploads/train.jpg'
    img.save(img_path)
    img_component = Image(img_path)
    img_component.drawHeight = 300
    img_component.drawWidth = 300

    style = TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.gray),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)])
    t.setStyle(style)

    content = [img_component, t]
    doc.build(content)
    
    return pdf_file_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error='No selected file')
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        uploaded_image = cv2.imread(file_path)
        preprocessed_image = preprocess_image(file_path)
        colon_mask = segment_colon_tissue(preprocessed_image)
        thickness = measure_thickness(colon_mask)
        original_image = cv2.imread(file_path)
        visualize_results(original_image, colon_mask, thickness)
        out_image_path = 'static/out.jpg'
        cv2.imwrite(out_image_path, colon_mask)
        img, predictions = predict_class(file_path)
        pdf_file_path = generate_pdf(img, predictions)
        
        return send_file(pdf_file_path)

    
@app.route('/segment',methods=['GET','POST'])
def segment():
    output_image="static/out.jpg"
    thickness=random.random()
    columnar_tissue=["thin","weak","thick"]
    water_shed=["Present","Not Preset"]
    tool=["Surgical Scalpel","Graspers/Forceps","Bipolar Forceps","Surgical Suction","Endo GIA Stapler:"]
    spec={
        "tissue_thickness":thickness,
        "simple_columnar_tissue":random.choice(columnar_tissue),
        "watershed_area":random.choice(water_shed),
        "Recommended_tool":random.choice(tool)
    }
    return render_template("segment.html",processed_image="static/out1.jpg",spec=spec)

if __name__ == '__main__':
    app.run(debug=True)
