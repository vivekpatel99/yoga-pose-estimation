import os

import cv2
import imageio.v2 as imageio

# --- Configuration ---
image_folder = "assets"  # The folder containing your images
output_gif = "assets/demo.gif"  # The name of the output GIF file
duration = 2  # Duration for each frame in seconds (e.g., 0.5 = half a second)


# Get a list of all image files in the folder
filenames = [
    f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".jpeg"))
]


# Create a list to hold the image data
images = []

# Read each image file and add it to the list
print("Reading image files...")
for filename in filenames:
    file_path = os.path.join(image_folder, filename)
    img = cv2.cvtColor(cv2.imread(file_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    resize_image = cv2.resize(img, (640, 480))
    images.append(resize_image)

if not images:
    print("No images found to create a GIF.")
    exit()

# The `duration` parameter for imageio.mimsave is in seconds PER FRAME.
# To get a total duration for the GIF, we divide it by the number of images.
duration_per_frame = duration / len(images)

# Save the images as a GIF
print(f"Creating GIF: {output_gif}")
imageio.mimsave(output_gif, images, fps=1)

print("GIF created successfully! 🎉")
