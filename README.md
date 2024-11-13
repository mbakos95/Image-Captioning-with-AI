
<!DOCTYPE html>
<html>
<head>
    <title>Image Captioning Project</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f9f9f9;
            color: #333;
        }
        h1 {
            color: #4CAF50;
        }
        h2 {
            color: #333;
            border-bottom: 2px solid #4CAF50;
        }
        code {
            background-color: #eaeaea;
            padding: 5px;
            border-radius: 3px;
        }
        .code-block {
            background-color: #f4f4f4;
            border: 1px solid #ddd;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <h1>Image Captioning with AI</h1>
    <h2>Overview</h2>
    <p>This project generates captions for images using a pre-trained Vision Transformer (ViT) and GPT-2 model from the Hugging Face Transformers library.</p>
    
    <h2>Steps</h2>
    <ol>
        <li>Install required libraries.</li>
        <li>Load the pre-trained model and tokenizer.</li>
        <li>Preprocess the image.</li>
        <li>Generate a caption.</li>
        <li>Display the image with its caption.</li>
    </ol>

    <h2>Requirements</h2>
    <ul>
        <li>Python 3.8 or higher</li>
        <li>Libraries: <code>torch</code>, <code>transformers</code>, <code>pillow</code>, <code>matplotlib</code></li>
    </ul>

    <h2>Installation</h2>
    <p>Run the following command to install dependencies:</p>
    <div class="code-block">
        <code>pip install torch transformers pillow matplotlib</code>
    </div>

    <h2>Code</h2>
    <div class="code-block">
        <pre>
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
import matplotlib.pyplot as plt

# Step 1: Load pre-trained model, feature extractor, and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Step 2: Load and preprocess the image
image = Image.open("your_image.jpg")
pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

# Step 3: Generate caption
caption_ids = model.generate(pixel_values, max_length=16, num_beams=4)
caption = tokenizer.decode(caption_ids[0], skip_special_tokens=True)

# Step 4: Display the image with the caption
plt.imshow(image)
plt.title(caption)
plt.axis('off')
plt.show()
        </pre>
    </div>

    <h2>Example Output</h2>
    <p>For an image of a dog playing in a park, the generated caption might be:</p>
    <blockquote>"a dog playing in the grass."</blockquote>
</body>
</html>
