# Nudibranch Species ID App

This project creates an end-to-end application for identifying nudibranch species from images. The app uses TensorFlow for model training and TensorFlow.js for browser-based inference.

🪸 **Nudibranch ID Tech Stack**
This application combines:
-  **Model**: Pre-trained MobileNetV2 for feature extraction
-  **Backend**: Python with TensorFlow and TensorFlow Hub
-  **Frontend**: Clean HTML/JS interface with real-time processing
-  **Deployment**: Local server with seamless browser integration
-  **Data**: Built-in database of common nudibranch species

## Project Structure

This is a simplified version of the nudibranch identification app that doesn't require one collect or provide a dataset of images. It uses a pre-trained model to extract features from uploaded images and suggests possible nudibranch species matches.

## How It Works

1. This app uses a pre-trained MobileNetV2 model from TensorFlow Hub to extract features from uploaded images.
2. It includes a built-in database of 10 common nudibranch species with descriptions, habitats, and visual features.
3. For demonstration purposes, it currently uses a simplified matching algorithm that shows random species as potential matches.
4. In a real application, the feature vectors would be compared against a database of known nudibranch images to find the closest matches.

## Features

- Simple web interface for uploading nudibranch images
- No need to collect or train on a dataset of images
- Displays potential matches with descriptions and characteristics
- Works entirely on your local computer (no data is sent to external servers)

## Requirements

- Python 3.6+
- TensorFlow 2.5+
- TensorFlow Hub
- Pillow
- NumPy
- Requests

## Installation and Usage

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python3 nudibranch_identifier.py
   ```

3. The application will open in your web browser. You can:
   - Upload an image by clicking on the dropzone or dragging and dropping an image
   - Click "Identify Species" to process the image
   - View potential matches with descriptions and characteristics

## Note

This is a demonstration app and the species identification is currently simulated. In a full implementation, the app would use actual feature comparison against known nudibranch images to find the most similar species.

To build a more accurate nudibranch identification system, you would need:
1. A large dataset of labeled nudibranch images
2. Training a specialized model or fine-tuning a pre-trained model on this dataset
3. More sophisticated feature matching algorithms
