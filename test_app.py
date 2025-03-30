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
        "message": "Welcome to XFarm Agriculture API"  # Updated to match the actual API response
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


# Tests for crop recommendation endpoint


def test_recommend_crop_endpoint():
    """Test the crop recommendation endpoint with valid parameters"""
    # Sample parameters based on main.py example
    params = {
        "n": 30.0,
        "p": 41.0,
        "k": 15.0,
        "temperature": 24.83,
        "humidity": 44.17,
        "ph": 5.88,
        "rainfall": 52.08,
    }

    response = client.get("/recommend_crop/", params=params)
    assert response.status_code == 200
    result = response.json()

    # Check response structure
    assert "recommended_crop" in result
    assert "soil_parameters" in result
    assert "climate_parameters" in result

    # Check soil parameters
    assert result["soil_parameters"]["nitrogen"] == params["n"]
    assert result["soil_parameters"]["phosphorus"] == params["p"]
    assert result["soil_parameters"]["potassium"] == params["k"]
    assert result["soil_parameters"]["ph"] == params["ph"]

    # Check climate parameters
    assert result["climate_parameters"]["temperature"] == params["temperature"]
    assert result["climate_parameters"]["humidity"] == params["humidity"]
    assert result["climate_parameters"]["rainfall"] == params["rainfall"]


def test_recommend_crop_missing_params():
    """Test crop recommendation with missing parameters"""
    # Missing some required parameters
    params = {
        "n": 30.0,
        "p": 41.0,
        # Missing k, temperature, etc.
    }

    response = client.get("/recommend_crop/", params=params)
    # FastAPI should return 422 Unprocessable Entity for missing required parameters
    assert response.status_code == 422


def test_recommend_crop_invalid_params():
    """Test crop recommendation with invalid parameter types"""
    # Invalid parameter types (string instead of number)
    params = {
        "n": 30.0,
        "p": 41.0,
        "k": 15.0,
        "temperature": "not-a-number",  # Invalid type
        "humidity": 44.17,
        "ph": 5.88,
        "rainfall": 52.08,
    }

    response = client.get("/recommend_crop/", params=params)
    # FastAPI should return 422 Unprocessable Entity for type validation errors
    assert response.status_code == 422
