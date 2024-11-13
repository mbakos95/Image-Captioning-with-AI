
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Captioning with AI</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f9f9f9;
            color: #333;
            line-height: 1.6;
        }
        h1 {
            color: #4CAF50;
            text-align: center;
        }
        h2 {
            color: #4CAF50;
            margin-top: 20px;
        }
        p, ul, ol, pre, code {
            margin-bottom: 15px;
        }
        pre {
            background: #f4f4f4;
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        code {
            background: #eee;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: Consolas, monospace;
        }
        .footer {
            margin-top: 40px;
            text-align: center;
            color: #555;
        }
    </style>
</head>
<body>
    <h1>Image Captioning with AI</h1>
    <h2>Overview</h2>
    <p>This project generates captions for images using a pre-trained Vision Transformer (ViT) and GPT-2 model from the Hugging Face Transformers library. It integrates computer vision and natural language processing to describe images with meaningful captions.</p>
    
    <h2>Steps</h2>
    <ol>
        <li>Install required libraries.</li>
        <li>Load the pre-trained model, tokenizer, and feature extractor.</li>
        <li>Preprocess the input image.</li>
        <li>Generate a caption for the image.</li>
        <li>Display the image with the generated caption.</li>
    </ol>

    <h2>Requirements</h2>
    <ul>
        <li>Python 3.8 or higher</li>
        <li>Required libraries: <code>torch</code>, <code>transformers</code>, <code>pillow</code>, <code>matplotlib</code></li>
    </ul>

    <h2>Installation</h2>
    <p>Install the necessary libraries by running the following command:</p>
    <pre><code>pip install torch transformers pillow matplotlib</code></pre>

    <h2>Code</h2>
    <pre><code>
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
    </code></pre>

    <h2>Example Output</h2>
    <p>If the image is of a dog playing in a park, the generated caption might be:</p>
    <blockquote>"a dog playing in the grass."</blockquote>

    <div class="footer">
        <p>Created as part of an AI project using Hugging Face Transformers.</p>
    </div>
</body>
</html>
