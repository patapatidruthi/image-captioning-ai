import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from PIL import Image
import random
import string
import pickle

# Step 1: Generate a random image (for demonstration purposes)
def generate_random_image(width=299, height=299):
    """Generates a random image of the specified size."""
    random_data = np.random.rand(height, width, 3) * 255  # Random pixel values
    random_image = Image.fromarray(random_data.astype('uint8'))  # Convert to an image
    return random_image

# Step 2: Extract features using pre-trained InceptionV3
def extract_features(img):
    """Extract features from an image using InceptionV3."""
    # Load the pre-trained InceptionV3 model (no top layers)
    inception_model = InceptionV3(include_top=False, weights='imagenet')
    inception_model.trainable = False  # Freeze the layers

    # Preprocess the image to the required format
    img = img.resize((299, 299))  # Resize to 299x299 for InceptionV3
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image

    # Extract features
    features = inception_model.predict(img_array)
    features = features.reshape((features.shape[0], -1))  # Flatten the features
    return features

# Step 3: Load pre-trained captioning model and tokenizer
def load_model_and_tokenizer():
    """Load the pre-trained caption generation model and tokenizer."""
    # Assuming a pre-trained model and tokenizer are saved locally
    # For demonstration, we will generate random captions
    model = load_model('models/image_captioning_model.h5')  # Replace with your pre-trained model
    with open('models/tokenizer.pkl', 'rb') as f:  # Load the tokenizer
        tokenizer = pickle.load(f)
    return model, tokenizer

# Step 4: Generate random caption (this is a placeholder for the actual caption generation)
def generate_random_caption():
    """Generate a random caption (placeholder for actual model predictions)."""
    # For demonstration, generate a random caption using a list of sample words
    sample_words = ['cat', 'dog', 'sun', 'sky', 'bird', 'tree', 'mountain', 'sea', 'river', 'forest']
    random_caption = " ".join(random.sample(sample_words, 5))  # Random 5 words caption
    return random_caption

# Step 5: Generate caption for the input image
def generate_caption_for_image(img):
    """Generate a caption for the image."""
    # Extract features from the image
    features = extract_features(img)

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Generate a random caption (for demonstration)
    caption = generate_random_caption()

    return caption

# Main function to execute the image captioning process
def main():
    # Step 1: Generate a random image
    random_image = generate_random_image()

    # Step 2: Generate a random caption for this image
    caption = generate_caption_for_image(random_image)

    # Step 3: Display the image and the generated caption
    plt.imshow(random_image)
    plt.title(caption)
    plt.axis('off')  # Hide axis
    plt.show()

    print("Generated Caption:", caption)

if __name__ == "__main__":
    main()
