from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import io
import uvicorn
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import datetime
from bs4 import BeautifulSoup
import requests

# Import the model definition from plant_disease.py
from plant_disease import ResNet9, classes

app = FastAPI(title="XFarm Agriculture API")

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

# Load the plant disease model
model = ResNet9(in_channels=3, num_diseases=38)
model.load_state_dict(
    torch.load("./plant-disease-model1.pth", map_location=torch.device("cpu"))
)
model.eval()

# Load the crop recommendation model
label_encoder = LabelEncoder()
# Load the dataset to fit the label encoder
crop_df = pd.read_csv("./Crop_recommendation.csv")
label_encoder.fit(crop_df["label"])
# Load the XGBoost model
crop_model = pickle.load(open("XGBoost.pkl", "rb"))


@app.get("/")
async def root():
    return {"message": "Welcome to XFarm Agriculture API"}


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


@app.get("/recommend_crop/")
async def recommend_crop(
    n: float = Query(..., description="Nitrogen content in soil"),
    p: float = Query(..., description="Phosphorus content in soil"),
    k: float = Query(..., description="Potassium content in soil"),
    temperature: float = Query(..., description="Temperature in Celsius"),
    humidity: float = Query(..., description="Humidity percentage"),
    ph: float = Query(..., description="pH value of soil"),
    rainfall: float = Query(..., description="Rainfall in mm"),
):
    """
    Endpoint to recommend suitable crops based on soil and climate parameters
    """
    try:
        # Prepare input data
        input_data = [[n, p, k, temperature, humidity, ph, rainfall]]

        # Make prediction
        prediction = crop_model.predict(input_data)

        # Decode prediction
        recommended_crop = label_encoder.inverse_transform(prediction)[0]

        return JSONResponse(
            content={
                "recommended_crop": recommended_crop,
                "soil_parameters": {
                    "nitrogen": n,
                    "phosphorus": p,
                    "potassium": k,
                    "ph": ph,
                },
                "climate_parameters": {
                    "temperature": temperature,
                    "humidity": humidity,
                    "rainfall": rainfall,
                },
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=400, content={"error": f"Error in recommendation: {str(e)}"}
        )


ids = {
    "cphBody_GridPriceData_Labdistrict_name_": "district name",
    "cphBody_GridPriceData_LabdMarketName_": "market name",
    "cphBody_GridPriceData_Labcomm_name_": "commodity",
    "cphBody_GridPriceData_LabdVariety_": ["variety", "grade"],
    "cphBody_GridPriceData_LabMinPrice_": "min price",
    "cphBody_GridPriceData_Labmaxpric_": "max price",
    "cphBody_GridPriceData_LabModalpric_": "modal price",
    "cphBody_GridPriceData_LabReportedDate_": "date",
}


def get_data(html_object):
    count = 0
    data = []
    while True:
        try:
            row = {}
            for key, value in ids.items():
                if isinstance(value, list):
                    vals = html_object.find_all(id=f"{key}{count}")
                    for k, v in zip(value, vals):
                        if v:
                            row[k] = v.text.strip()
                        else:
                            return data
                else:
                    val = html_object.find(id=f"{key}{count}")
                    if not val:
                        return data
                    if value == "date":
                        row[value] = datetime.datetime.strptime(
                            val.text.strip(), "%d %b %Y"
                        ).date()
                    else:
                        row[value] = val.text.strip()
            data.append(row)
            count += 1
        except AttributeError:
            break
    return data


@app.get("/market_analysis/")
async def market_analysis(district_name: str, commodity: str, state: str, market: str):
    """
    Endpoint to fetch market analysis data for a specific district and commodity
    """
    try:
        url = f"https://agmarknet.gov.in/SearchCmmMkt.aspx?Tx_Commodity=25&Tx_State=KK&Tx_District=0&Tx_Market=0&DateFrom=15-Feb-2025&DateTo=31-Mar-2025&Fr_Date=15-Feb-2025&To_Date=31-Mar-2025&Tx_Trend=0&Tx_CommodityHead=Potato&Tx_StateHead=Karnataka&Tx_DistrictHead=--Select--&Tx_MarketHead=--Select--"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        data = get_data(soup)
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(
            status_code=400, content={"error": f"Error fetching data: {str(e)}"}
        )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
