from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Step 1: Load the image
image_path = r"C:\Users\hairi\OneDrive\IDP\monai_project\gait_project_backup\data_onecolumn.jpg"  # Replace with your actual file path
image = Image.open(image_path)

# Step 2: Convert to grayscale
gray_image = image.convert("L")  # "L" mode is for grayscale

# Step 3: Convert to NumPy array
gray_array = np.array(gray_image)

# Step 4: Apply jet colormap using OpenCV
heatmap = cv2.applyColorMap(gray_array, cv2.COLORMAP_JET)

# Step 5: Convert from BGR (OpenCV) to RGB (for matplotlib)
heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# Step 6: Display the heatmap
plt.figure(figsize=(10, 6))
plt.imshow(heatmap_rgb)
plt.axis("off")
plt.title("Heatmap using Jet Colormap")
plt.tight_layout()
plt.show()