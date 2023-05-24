import cv2
import os

def otsu_binarization(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur with a smaller kernel size to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(blurred)

    # Apply Otsu's Binarization
    ret, binary = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary




label_dir = '/Users/lu992/Documents/day5 endothelial/stained'
binarized_label_dir = '/Users/lu992/Documents/day5 endothelial/stained_binarized'

# Create the output directory if it doesn't exist
os.makedirs(binarized_label_dir, exist_ok=True)



# Process label_dir
for file_name in os.listdir(label_dir):
    # Check if the file is an image
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
        input_path = os.path.join(label_dir, file_name)
        output_path = os.path.join(binarized_label_dir, file_name)

        # Apply Otsu's Binarization
        binary_image = otsu_binarization(input_path)

        # Save the binary image
        cv2.imwrite(output_path, binary_image)

