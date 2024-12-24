# image-captioning-ai
An image captioning system using CNN (InceptionV3) for feature extraction and LSTM for generating human-readable captions.
## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- Pillow

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/image-captioning-ai.git
    cd image-captioning-ai
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Preprocess the data (features extraction from images and tokenization of captions):
    ```bash
    python src/preprocess.py
    ```

4. Train the model (modify the training script as necessary for your dataset):
    ```bash
    python src/train_model.py  # You may need to create this script based on your training process
    ```

5. Generate captions:
    ```bash
    python src/image_captioning.py
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

