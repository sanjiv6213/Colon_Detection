import cv2
import numpy as np
from skimage.measure import label, regionprops

# Step 1: Load and preprocess the image
def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # Perform thresholding to separate tissue from background
    _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresholded_image

# Step 2: Segment the colon tissue
def segment_colon_tissue(image):
    # Morphological operations to enhance tissue regions
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Label connected components
    labeled_image = label(sure_bg)
    # Find properties of regions
    regions = regionprops(labeled_image)
    # Extract the largest region (assumed to be the colon tissue)
    max_area = 0
    for region in regions:
        if region.area > max_area:
            max_area = region.area
            max_region = region
    # Create a mask for the largest region
    colon_mask = np.zeros_like(image)
    colon_mask[max_region.bbox[0]:max_region.bbox[2], max_region.bbox[1]:max_region.bbox[3]] = max_region.filled_image
    return colon_mask

# Step 3: Thickness Measurement
def measure_thickness(image):
    # You need to implement thickness measurement based on your requirements
    # This can involve edge detection followed by distance measurement
    # For simplicity, let's assume the thickness is the maximum distance between edges
    edges = cv2.Canny(image, 50, 150)
    distance_transform = cv2.distanceTransform(edges, cv2.DIST_L2, 5)
    thickness = np.max(distance_transform)
    return thickness

# Step 4: Visualization
def visualize_results(original_image, colon_mask, thickness):
    # Apply mask to original image
    masked_image = cv2.bitwise_and(original_image, original_image, mask=colon_mask)
    # Display the original image and the masked image
    cv2.imshow('Original Image', original_image)
    cv2.imshow('Colon Tissue Mask', masked_image)
    cv2.imwrite("out_colon.jpg",masked_image)
    print("Estimated Thickness of Colon Tissue:", thickness)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function
def main():
    # Step 1: Preprocess the image
    image_path = 'train.jpg'
    preprocessed_image = preprocess_image(image_path)
    
    # Step 2: Segment the colon tissue
    colon_mask = segment_colon_tissue(preprocessed_image)
    
    # Step 3: Thickness Measurement
    thickness = measure_thickness(colon_mask)
    
    # Step 4: Visualization
    original_image = cv2.imread(image_path)
    visualize_results(original_image, colon_mask, thickness)

if __name__ == "__main__":
    main()
