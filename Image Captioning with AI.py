import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
import matplotlib.pyplot as plt

# Step 1: Load pre-trained model, feature extractor, and tokenizer
print("Loading pre-trained model...")
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
print("Model loaded successfully.")

# Step 2: Load and preprocess the image
image_path = "your_image.jpg"  # Replace with your image file path
try:
    image = Image.open(image_path)
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

print("Preprocessing the image...")
pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

# Step 3: Generate caption
print("Generating caption...")
try:
    caption_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(caption_ids[0], skip_special_tokens=True)
except Exception as e:
    print(f"Error generating caption: {e}")
    caption = "Unable to generate caption."

# Step 4: Display the image with the caption
print(f"Caption: {caption}")
plt.imshow(image)
plt.title(caption)
plt.axis('off')
plt.show()
