
import cv2
import numpy as np
import os
import traceback

def validate_oral_image(img_path):
    print(f"Testing image: {img_path}")
    try:
        # Load image
        img = cv2.imread(img_path)
        if img is None: 
            print("Failed to load image")
            return False
        
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        height, width, _ = img.shape
        total_pixels = height * width
        
        # --- CHECK 1: Blue/Sky Detection ---
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_pixels = cv2.countNonZero(blue_mask)
        blue_ratio = blue_pixels / total_pixels
        print(f"Blue/Sky pixel ratio: {blue_ratio:.4f}")
        
        if blue_ratio > 0.05: 
            print("REJECTED (Blue Sky Detected > 5%)")
            # return False # Continue for debugging

        # --- CHECK 2: Flesh Tones (Strict) ---
        lower_red1 = np.array([0, 50, 40])
        upper_red1 = np.array([20, 255, 255])
        lower_red2 = np.array([160, 50, 40])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        combined_mask = mask1 | mask2
        
        flesh_pixels = cv2.countNonZero(combined_mask)
        global_ratio = flesh_pixels / total_pixels
        print(f"Global Flesh pixel ratio: {global_ratio:.4f}")

        # --- CHECK 3: Center Crop Analysis ---
        center_h = int(height * 0.4)
        center_w = int(width * 0.4)
        center_y = int(height * 0.3)
        center_x = int(width * 0.3)
        
        center_crop_mask = combined_mask[center_y:center_y+center_h, center_x:center_x+center_w]
        center_total = center_h * center_w
        center_flesh = cv2.countNonZero(center_crop_mask)
        center_ratio = center_flesh / center_total
        print(f"Center Flesh pixel ratio: {center_ratio:.4f}")
        
        if center_ratio < 0.50:
            print("REJECTED (Center Flesh < 50%)")
            
        if global_ratio < 0.30:
             print("REJECTED (Global Flesh < 30%)")

        # --- CHECK 4: Face Detection ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        if not os.path.exists(face_cascade_path):
             print(f"Warning: Cascade file not found at {face_cascade_path}")
        else:
            face_cascade = cv2.CascadeClassifier(face_cascade_path)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            print(f"Faces detected: {len(faces)}")
            if len(faces) > 0:
                print("REJECTED (Face Detected)")
                return False

        print("ACCEPTED")
        return True
    except Exception as e:
        print(f"Validation Error: {e}")
        traceback.print_exc()
        return True


# Test with the specific images
img_path_1 = "/Users/tanishq/.gemini/antigravity/brain/a4bbae1e-0fb8-483b-92a1-8923705c5f88/uploaded_media_0_1769622986432.png"
img_path_2 = "/Users/tanishq/.gemini/antigravity/brain/a4bbae1e-0fb8-483b-92a1-8923705c5f88/uploaded_media_1_1769622986432.png"

print("--- Testing Image 1 ---")
validate_oral_image(img_path_1)
print("\n--- Testing Image 2 ---")



img_path_5 = "/Users/tanishq/.gemini/antigravity/brain/a4bbae1e-0fb8-483b-92a1-8923705c5f88/uploaded_media_1769627127086.png"
print("\n--- Testing Image 5 (Religious Symbols) ---")
validate_oral_image(img_path_5)
