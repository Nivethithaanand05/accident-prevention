import os
import pandas as pd
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# ✅ Update the path below to where you extracted the folder
image_folder = r'D:\path\to\number_plate_dataset\number_plate_images'  # <-- CHANGE THIS

# List of image files
images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

# Extract text from images
data = []
for img in images[:5]:  # Preview only first 5 images for df.head()
    result = reader.readtext(os.path.join(image_folder, img), detail=0)
    plate_text = result[0] if result else "Not detected"
    data.append({'filename': img, 'plate_text': plate_text})

# Create DataFrame
df_plates = pd.DataFrame(data)
print(df_plates.head())
