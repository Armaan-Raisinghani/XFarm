from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import io
import uvicorn

# Import the model definition from plant_disease.py
from plant_disease import ResNet9, classes

app = FastAPI(title="XFarm Plant Disease Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define the image transformation
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

# Load the model
model = ResNet9(in_channels=3, num_diseases=38)
model.load_state_dict(
    torch.load("./plant-disease-model1.pth", map_location=torch.device("cpu"))
)
model.eval()


@app.get("/")
async def root():
    return {"message": "Welcome to XFarm Plant Disease Detection API"}


@app.post("/predict/")
async def predict_disease(file: UploadFile = File(...)):
    """
    Endpoint to predict plant disease from uploaded image
    """
    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Apply transformations
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)

        predicted_class = classes[predicted.item()]
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][
            predicted.item()
        ].item()

        return JSONResponse(
            content={
                "predicted_class": predicted_class,
                "confidence": round(confidence * 100, 2),
                "filename": file.filename,
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=400, content={"error": f"Invalid image file: {str(e)}"}
        )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
