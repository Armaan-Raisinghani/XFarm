import io
import os
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

# Import the app from your main application file
from app import app

# Initialize the test client
client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint returns the correct welcome message"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to XFarm Plant Disease Detection API"
    }


def create_test_image():
    """Helper function to create a test image"""
    # Create a simple test image (100x100 RGB)
    image = Image.new("RGB", (100, 100), color="green")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    return img_byte_arr


def test_predict_endpoint():
    """Test the prediction endpoint with a test image"""
    test_img = create_test_image()
    response = client.post(
        "/predict/", files={"file": ("test_image.png", test_img, "image/png")}
    )
    assert response.status_code == 200
    result = response.json()
    assert "predicted_class" in result
    assert "confidence" in result
    assert result["filename"] == "test_image.png"


def test_predict_endpoint_invalid_file():
    """Test prediction endpoint with invalid file type"""
    # Create a text file instead of an image
    text_file = io.BytesIO(b"This is not an image")

    response = client.post(
        "/predict/", files={"file": ("test.txt", text_file, "text/plain")}
    )

    # Check for 400 error response with error message
    assert response.status_code == 400
    assert "error" in response.json()


def test_predict_endpoint_no_file():
    """Test prediction endpoint with no file"""
    response = client.post("/predict/")
    assert response.status_code == 422  # Unprocessable Entity, FastAPI validation error
