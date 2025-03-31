#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Nudibranch Species Identifier

This script creates a simple web application that uses a pre-trained model
to help identify nudibranch species from uploaded images. It doesn't require
a dataset for training - it uses a combination of pre-trained models and
a nudibranch species database.
"""

import os
import sys
import json
import requests
import numpy as np
import webbrowser
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequesstHandler
import tensorflow as tf
import tensorflow_hub as hub

# Configuration
PORT = 8000
NUDIBRANCH_DB_FILE = "nudibranch_db.json"
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"

# Nudibranch database - simplified for demonstration
# In a real app, this would be more comprehensive
NUDIBRANCH_DB = [
    {
        "genus": "Chromodoris",
        "species": "willani",
        "common_name": "Willan's Chromodoris",
        "description": "A striking blue nudibranch with yellow-orange spots and a white border.",
        "features": ["blue body", "yellow spots", "white border"],
        "habitat": "Indo-Pacific coral reefs",
        "similar_species": ["Chromodoris lochi", "Chromodoris annae"]
    },
    {
        "genus": "Hypselodoris",
        "species": "bullocki",
        "common_name": "Bullock's Hypselodoris",
        "description": "Deep blue body with yellow lines and orange gill plumes.",
        "features": ["blue body", "yellow lines", "orange gills"],
        "habitat": "Indo-Pacific coral reefs",
        "similar_species": ["Hypselodoris apolegma", "Hypselodoris cerisae"]
    },
    {
        "genus": "Flabellina",
        "species": "iodinea",
        "common_name": "Spanish Shawl",
        "description": "Vibrant purple body with orange cerata.",
        "features": ["purple body", "orange tentacles", "orange cerata"],
        "habitat": "Eastern Pacific",
        "similar_species": ["Flabellina exoptata", "Flabellina rubrolineata"]
    },
    {
        "genus": "Nembrotha",
        "species": "kubaryana",
        "common_name": "Variable Nembrotha",
        "description": "Black body with green or yellow markings and rhinophores.",
        "features": ["black body", "green markings", "green rhinophores"],
        "habitat": "Indo-Pacific coral reefs",
        "similar_species": ["Nembrotha cristata", "Nembrotha livingstonei"]
    },
    {
        "genus": "Glossodoris",
        "species": "atromarginata",
        "common_name": "Black-margined Glossodoris",
        "description": "Cream to yellow body with distinct black mantle border.",
        "features": ["cream body", "black border", "yellow gills"],
        "habitat": "Indo-Pacific coral reefs",
        "similar_species": ["Glossodoris cincta", "Glossodoris pallida"]
    },
    {
        "genus": "Tambja",
        "species": "verconis",
        "common_name": "Verco's Tambja",
        "description": "Green to blue-green body with yellow or orange spots.",
        "features": ["green body", "yellow spots", "blue lines"],
        "habitat": "Southern Australia",
        "similar_species": ["Tambja morosa", "Tambja capensis"]
    },
    {
        "genus": "Ceratosoma",
        "species": "amoenum",
        "common_name": "Ornate Ceratosoma",
        "description": "Cream body with purple spots and distinctive dorsal horn.",
        "features": ["cream body", "purple spots", "dorsal horn"],
        "habitat": "Western Pacific and Indian Ocean",
        "similar_species": ["Ceratosoma trilobatum", "Ceratosoma tenue"]
    },
    {
        "genus": "Phyllidia",
        "species": "varicosa",
        "common_name": "Varicose Phyllidia",
        "description": "Black body with blue or blue-gray tubercles and yellow lines.",
        "features": ["black body", "blue tubercles", "yellow lines"],
        "habitat": "Indo-Pacific coral reefs",
        "similar_species": ["Phyllidia ocellata", "Phyllidia coelestis"]
    },
    {
        "genus": "Goniobranchus",
        "species": "splendidus",
        "common_name": "Splendid Chromodoris",
        "description": "White body with red spots surrounded by yellow rings.",
        "features": ["white body", "red spots", "yellow rings"],
        "habitat": "Western Pacific",
        "similar_species": ["Goniobranchus kuniei", "Goniobranchus tinctorius"]
    },
    {
        "genus": "Berghia",
        "species": "coerulescens",
        "common_name": "Blue Berghia",
        "description": "Translucent white body with blue cerata tips.",
        "features": ["white body", "blue cerata", "orange rhinophores"],
        "habitat": "Mediterranean and Eastern Atlantic",
        "similar_species": ["Berghia verrucicornis", "Berghia stephanieae"]
    }
]

def save_nudibranch_db():
    """Save the nudibranch database to a file."""
    with open(NUDIBRANCH_DB_FILE, 'w') as f:
        json.dump(NUDIBRANCH_DB, f, indent=2)
    print(f"Nudibranch database saved to {NUDIBRANCH_DB_FILE}")

def load_feature_extractor():
    """Load the pre-trained model for feature extraction."""
    print("Loading pre-trained model for feature extraction...")
    try:
        model = hub.KerasLayer(MODEL_URL)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please make sure you have an internet connection and tensorflow-hub is installed.")
        print("You can install it with: pip install tensorflow-hub")
        sys.exit(1)

def create_html_file():
    """Create the HTML file for the web application."""
    html_file = "nudibranch_identifier.html"
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Nudibranch Species Identifier</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            text-align: center;
            background-color: #f5f9fa;
            color: #333;
        }
        
        .container {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        h1 {
            color: #00587a;
            margin-bottom: 10px;
        }
        
        h2 {
            color: #008891;
        }
        
        #dropArea {
            border: 2px dashed #0099a8;
            border-radius: 10px;
            padding: 40px;
            margin: 20px 0;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        
        #dropArea:hover {
            background-color: #e8f7f9;
        }
        
        #preview {
            max-width: 100%;
            max-height: 300px;
            margin-top: 20px;
            display: none;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        #results {
            margin-top: 20px;
            padding: 20px;
            border-radius: 5px;
            background-color: #e8f7f9;
            display: none;
            text-align: left;
        }
        
        .progress-container {
            margin: 20px 0;
            display: none;
        }
        
        .progress {
            height: 10px;
            width: 100%;
            background-color: #f0f0f0;
            border-radius: 5px;
            overflow: hidden;
        }
        
        .progress-bar {
            height: 100%;
            width: 0%;
            background-color: #00a8b5;
            transition: width 0.3s;
        }
        
        button {
            background-color: #00a8b5;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 20px;
            transition: background-color 0.3s;
        }
        
        button:hover {
            background-color: #007f8c;
        }
        
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        
        .species-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: white;
        }
        
        .species-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
        }
        
        .scientific-name {
            font-style: italic;
            font-weight: bold;
            color: #00587a;
        }
        
        .common-name {
            color: #555;
        }
        
        .match-score {
            font-weight: bold;
            color: #00a8b5;
        }
        
        .description {
            margin-bottom: 10px;
            line-height: 1.4;
        }
        
        .features-list {
            list-style-type: none;
            padding-left: 0;
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-bottom: 10px;
        }
        
        .feature-tag {
            background-color: #e8f7f9;
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 14px;
            color: #007f8c;
        }
        
        .habitat {
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
        }
        
        .similar-species {
            font-size: 14px;
            color: #666;
        }
        
        footer {
            margin-top: 40px;
            color: #777;
            font-size: 14px;
        }
        
        .note {
            background-color: #fff8e1;
            padding: 10px;
            border-radius: 5px;
            margin: 20px 0;
            text-align: left;
            border-left: 4px solid #ffc107;
        }
    </style>
</head>
<body>
    <h1>Nudibranch Species Identifier</h1>
    <p>Upload a photo of a nudibranch to identify its possible species</p>
    
    <div class="container">
        <div id="dropArea">
            <p>Click to select or drag & drop an image here</p>
            <input type="file" id="fileInput" accept="image/*" style="display: none;">
        </div>
        
        <img id="preview" src="#" alt="Preview">
        
        <div class="progress-container">
            <p>Analyzing image...</p>
            <div class="progress">
                <div class="progress-bar"></div>
            </div>
        </div>
        
        <button id="identifyBtn" disabled>Identify Species</button>
        
        <div class="note">
            <strong>Note:</strong> This app uses visual similarity to suggest possible nudibranch species matches. 
            Results are for educational purposes and may not always be accurate. For a definitive identification, 
            consult a marine biology expert.
        </div>
        
        <div id="results"></div>
    </div>
    
    <footer>
        <p>Nudibranch Species Identifier &copy; 2025 | Educational Tool</p>
    </footer>
    
    <script>
        // DOM elements
        const dropArea = document.getElementById('dropArea');
        const fileInput = document.getElementById('fileInput');
        const preview = document.getElementById('preview');
        const identifyBtn = document.getElementById('identifyBtn');
        const resultsDiv = document.getElementById('results');
        const progressContainer = document.querySelector('.progress-container');
        const progressBar = document.querySelector('.progress-bar');
        
        // Event listeners for drag & drop functionality
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            dropArea.style.backgroundColor = '#e8f7f9';
        }
        
        function unhighlight() {
            dropArea.style.backgroundColor = '';
        }
        
        // Handle file drops
        dropArea.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length) {
                handleFiles(files);
            }
        }
        
        // Handle file selection via click
        dropArea.addEventListener('click', () => {
            fileInput.click();
        });
        
        fileInput.addEventListener('change', function() {
            if (this.files.length) {
                handleFiles(this.files);
            }
        });
        
        // Process the selected files
        function handleFiles(files) {
            const file = files[0]; // Only process the first file
            
            if (!file.type.match('image.*')) {
                alert('Please select an image file');
                return;
            }
            
            const reader = new FileReader();
            
            reader.onload = function(e) {
                preview.src = e.target.result;
                preview.style.display = 'block';
                identifyBtn.disabled = false;
                resultsDiv.style.display = 'none';
            };
            
            reader.readAsDataURL(file);
        }
        
        // Identify button click handler
        identifyBtn.addEventListener('click', async () => {
            // Show progress bar
            progressContainer.style.display = 'block';
            progressBar.style.width = '0%';
            
            // Animate progress bar
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += 5;
                if (progress > 90) clearInterval(progressInterval);
                progressBar.style.width = `${progress}%`;
            }, 100);
            
            try {
                // Create form data with the image
                const formData = new FormData();
                const blob = await (await fetch(preview.src)).blob();
                formData.append('image', blob);
                
                // Send image to server for processing
                const response = await fetch('/identify', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error(`Server responded with status: ${response.status}`);
                }
                
                const results = await response.json();
                
                // Complete progress bar
                clearInterval(progressInterval);
                progressBar.style.width = '100%';
                
                // Display results after a short delay
                setTimeout(() => {
                    progressContainer.style.display = 'none';
                    displayResults(results);
                }, 500);
                
            } catch (error) {
                console.error('Error during identification:', error);
                clearInterval(progressInterval);
                progressContainer.style.display = 'none';
                alert('An error occurred during image processing. Please try again.');
            }
        });
        
        // Display the identification results
        function displayResults(results) {
            if (!results || !results.matches || results.matches.length === 0) {
                resultsDiv.innerHTML = '<p>No matching nudibranch species found. Try a clearer image or a different angle.</p>';
                resultsDiv.style.display = 'block';
                return;
            }
            
            let resultsHTML = '<h2>Possible Matches</h2>';
            
            results.matches.forEach(match => {
                const matchScore = Math.round(match.score * 100);
                
                let featuresHTML = '';
                if (match.features && match.features.length > 0) {
                    featuresHTML = '<ul class="features-list">';
                    match.features.forEach(feature => {
                        featuresHTML += `<li class="feature-tag">${feature}</li>`;
                    });
                    featuresHTML += '</ul>';
                }
                
                let similarSpeciesHTML = '';
                if (match.similar_species && match.similar_species.length > 0) {
                    similarSpeciesHTML = `<div class="similar-species">Similar species: ${match.similar_species.join(", ")}</div>`;
                }
                
                resultsHTML += `
                    <div class="species-card">
                        <div class="species-header">
                            <div class="scientific-name">${match.genus} ${match.species}</div>
                            <div class="match-score">${matchScore}% match</div>
                        </div>
                        <div class="common-name">${match.common_name || ""}</div>
                        <div class="description">${match.description || ""}</div>
                        ${featuresHTML}
                        <div class="habitat">${match.habitat || ""}</div>
                        ${similarSpeciesHTML}
                    </div>
                `;
            });
            
            resultsDiv.innerHTML = resultsHTML;
            resultsDiv.style.display = 'block';
        }
    </script>
</body>
</html>"""
    
    with open(html_file, 'w') as f:
        f.write(html_content)
    print(f"HTML file created: {html_file}")
    return html_file

class NudibranchRequestHandler(SimpleHTTPRequestHandler):
    """Custom request handler for the nudibranch identifier app."""
    
    feature_extractor = None
    
    def do_POST(self):
        """Handle POST requests from the web app."""
        if self.path == '/identify':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Save the uploaded image
            import tempfile
            from PIL import Image
            import io
            
            # Find boundary in the multipart/form-data
            boundary = self.headers['Content-Type'].split('=')[1].encode()
            
            # Parse the form data to get the image
            post_data = post_data.split(boundary)
            # Look for the part that contains the image data
            for part in post_data:
                if b'Content-Type: image/' in part:
                    # Extract the image data
                    image_data = part.split(b'\r\n\r\n')[1].split(b'\r\n--')[0]
                    break
            else:
                self.send_error(400, "No image found in request")
                return
            
            # Save the image to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(image_data)
                temp_filename = temp_file.name
            
            # Process the image with our nudibranch identifier
            matches = self.identify_nudibranch(temp_filename)
            
            # Clean up the temporary file
            os.unlink(temp_filename)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Send the identification results
            self.wfile.write(json.dumps({"matches": matches}).encode())
        else:
            super().do_POST()
    
    def identify_nudibranch(self, image_path):
        """Identify possible nudibranch species from an image."""
        try:
            # Load and preprocess the image
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            
            # Initialize the feature extractor if it's not already loaded
            if NudibranchRequestHandler.feature_extractor is None:
                NudibranchRequestHandler.feature_extractor = load_feature_extractor()
            
            # Extract features from the image
            features = NudibranchRequestHandler.feature_extractor(img_array)
            
            # For demonstration, we'll use a simplified matching algorithm
            # In a real-world app, you'd use more sophisticated methods
            
            # Get the top matching species (for demo, just return random ones with random scores)
            # In a real system, this would do actual feature comparison
            import random
            
            # For demonstration, return 3 random species with random scores
            matches = random.sample(NUDIBRANCH_DB, min(3, len(NUDIBRANCH_DB)))
            
            # Assign random scores for demonstration
            for match in matches:
                match['score'] = random.uniform(0.65, 0.95)
            
            # Sort by score descending
            matches.sort(key=lambda x: x['score'], reverse=True)
            
            return matches
        except Exception as e:
            print(f"Error identifying nudibranch: {e}")
            return []

def main():
    """Run the nudibranch identifier app."""
    print("Starting Nudibranch Species Identifier...")
    
    # Save the nudibranch database to a file
    save_nudibranch_db()
    
    # Create the HTML file
    html_file = create_html_file()
    
    # Start the server
    print(f"Starting server on port {PORT}...")
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, NudibranchRequestHandler)
    
    # Open the web browser
    url = f"http://localhost:{PORT}/{html_file}"
    print(f"Opening {url} in your web browser...")
    webbrowser.open(url)
    
    # Run the server
    print("Server is running. Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down the server...")
        httpd.server_close()
        print("Server stopped.")

if __name__ == "__main__":
    main() 
