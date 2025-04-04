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
                        ).timestamp()
                    else:
                        row[value] = val.text.strip()
            data.append(row)
            count += 1
        except AttributeError:
            break
    return data


@app.post("/market_analysis/")
async def market_analysis(district: str, commodity: str, state: str, market: str):
    """
    Endpoint to fetch market analysis data for a specific district and commodity
    """
    print(district, commodity, state, market)
    commodities = {
    "Absinthe": 451,"Ajwan": 137,"Alasande Gram": 281,"Almond(Badam)": 325,"Alsandikai": 166,"Amaranthus": 86,"Ambada Seed": 130,"Ambady/Mesta": 417,"Amla(Nelli Kai)": 355,"Amphophalus": 102,"Amranthas Red": 419,"Antawala": 209,"Anthorium": 379,"Apple": 17,"Apricot(Jardalu/Khumani)": 326,"Arecanut(Betelnut/Supari)": 140,"Arhar (Tur/Red Gram)(Whole)": 49,"Arhar Dal(Tur Dal)": 260,"Asalia": 444,"Asgand": 505,"Ashgourd": 83,"Ashoka": 506,"Ashwagandha": 443,"Asparagus": 434,"Astera": 232,"Atis": 507,"Avare Dal": 269,"Bael": 418,"Bajji chilli": 491,"Bajra(Pearl Millet/Cumbu)": 28,"Balekai": 274,"balsam": 482,"Bamboo": 204,"Banana": 19,"Banana - Green": 90,"Banana flower": 483,"Banana Leaf": 485,"Banana stem": 484,"Barley (Jau)": 29,
    "basil": 435,
    "Bay leaf (Tejpatta)": 321,
    "Beans": 94,
    "Beaten Rice": 262,
    "Beetroot": 157,
    "Behada": 508,
    "Bengal Gram Dal (Chana Dal)": 263,
    "Bengal Gram(Gram)(Whole)": 6,
    "Ber(Zizyphus/Borehannu)": 357,
    "Betal Leaves": 143,
    "Betelnuts": 41,
    "Bhindi(Ladies Finger)": 85,
    "Bhui Amlaya": 448,
    "Big Gram": 113,
    "Binoula": 51,
    "Bitter gourd": 81,
    "Black Gram (Urd Beans)(Whole)": 8,
    "Black Gram Dal (Urd Dal)": 264,
    "Black pepper": 38,
    "BOP": 380,
    "Borehannu": 189,
    "Bottle gourd": 82,
    "Brahmi": 449,
    "Bran": 290,
    "Bread Fruit": 497,
    "Brinjal": 35,
    "Brocoli": 487,
    "Broken Rice": 293,
    "Broomstick(Flower Broom)": 320,
    "Bull": 214,
    "Bullar": 284,
    "Bunch Beans": 224,
    "Butter": 272,
    "buttery": 416,
    "Cabbage": 154,
    "Calendula": 480,
    "Calf": 215,
    "Camel Hair": 354,
    "Cane": 205,
    "Capsicum": 164,
    "Cardamoms": 40,
    "Carnation": 375,
    "Carrot": 153,
    "Cashew Kernnel": 238,
    "Cashewnuts": 36,
    "Castor Oil": 270,
    "Castor Seed": 123,
    "Cauliflower": 34,
    "Chakotha": 188,
    "Chandrashoor": 438,
    "Chapparad Avare": 169,
    "Chennangi (Whole)": 241,
    "Chennangi Dal": 295,
    "Cherry": 328,
    "Chikoos(Sapota)": 71,
    "Chili Red": 26,
    "Chilly Capsicum": 88,
    "Chironji": 509,
    "Chow Chow": 167,
    "Chrysanthemum": 402,
    "Chrysanthemum(Loose)": 231,
    "Cinamon(Dalchini)": 316,
    "cineraria": 467,
    "Clarkia": 478,
    "Cloves": 105,
    "Cluster beans": 80,
    "Coca": 315,
    "Cock": 368,
    "Cocoa": 104,
    "Coconut": 138,
    "Coconut Oil": 266,
    "Coconut Seed": 112,
    "Coffee": 45,
    "Colacasia": 318,
    "Copra": 129,
    "Coriander(Leaves)": 43,
    "Corriander seed": 108,
    "Cossandra": 472,
    "Cotton": 15,
    "Cotton Seed": 99,
    "Cow": 212,
    "Cowpea (Lobia/Karamani)": 92,
    "Cowpea(Veg)": 89,
    "Cucumbar(Kheera)": 159,
    "Cummin Seed(Jeera)": 42,
    "Curry Leaf": 486,
    "Custard Apple (Sharifa)": 352,
    "Daila(Chandni)": 382,
    "Dal (Avare)": 91,
    "Dalda": 273,
    "Delha": 410,
    "Dhaincha": 69,
    "dhawai flowers": 442,
    "dianthus": 476,
    "Double Beans": 492,
    "Dragon fruit": 495,
    "dried mango": 423,
    "Drumstick": 168,
    "Dry Chillies": 132,
    "Dry Fodder": 345,
    "Dry Grapes": 278,
    "Duck": 370,
    "Duster Beans": 163,
    "Egg": 367,
    "Egypian Clover(Barseem)": 361,
    "Elephant Yam (Suran)": 296,
    "Field Pea": 64,
    "Fig(Anjura/Anjeer)": 221,
    "Firewood": 206,
    "Fish": 366,
    "Flax seeds": 510,
    "Flower Broom": 365,
    "Foxtail Millet(Navane)": 121,
    "French Beans (Frasbean)": 298,
    "Galgal(Lemon)": 350,
    "Gamphrena": 471,
    "Garlic": 25,
    "Ghee": 249,
    "Giloy": 452,
    "Gingelly Oil": 276,
    "Ginger(Dry)": 27,
    "Ginger(Green)": 103,
    "Gladiolus Bulb": 364,
    "Gladiolus Cut Flower": 363,
    "Glardia": 462,
    "Goat": 219,
    "Goat Hair": 353,
    "golden rod": 475,
    "Gond": 511,
    "Goose berry (Nellikkai)": 494,
    "Gram Raw(Chholia)": 359,
    "Gramflour": 294,
    "Grapes": 22,
    "Green Avare (W)": 165,
    "Green Chilli": 87,
    "Green Fodder": 346,
    "Green Gram (Moong)(Whole)": 9,
    "Green Gram Dal (Moong Dal)": 265,
    "Green Peas": 50,
    "Ground Nut Oil": 267,
    "Ground Nut Seed": 268,
    "Groundnut": 10,
    "Groundnut (Split)": 314,
    "Groundnut pods (raw)": 312,
    "Guar": 75,
    "Guar Seed(Cluster Beans Seed)": 413,
    "Guava": 185,
    "Gudmar": 453,
    "Guggal": 454,
    "gulli": 461,
    "Gur(Jaggery)": 74,
    "Gurellu": 279,
    "gypsophila": 469,
    "Haralekai": 252,
    "Harrah": 512,
    "He Buffalo": 216,
    "Heliconia species": 474,
    "Hen": 369,
    "Hippe Seed": 125,
    "Honey": 236,
    "Honge seed": 124,
    "Hybrid Cumbu": 119,
    "hydrangea": 473,
    "Indian Beans (Seam)": 299,
    "Indian Colza(Sarson)": 344,
    "Irish": 465,
    "Isabgul (Psyllium)": 256,
    "Jack Fruit": 182,
    "Jaee": 513,
    "Jaffri": 406,
    "Jaggery": 151,
    "Jamamkhan": 175,
    "Jamun(Narale Hannu)": 184,
    "Jarbara": 376,
    "Jasmine": 229,
    "Javi": 250,
    "Jowar(Sorghum)": 5,
    "Jute": 16,
    "Jute Seed": 210,
    "Kabuli Chana(Chickpeas-White)": 362,
    "Kacholam": 317,
    "Kakada": 230,
    "kakatan": 501,
    "Kalihari": 456,
    "Kalmegh": 457,
    "Kankambra": 233,
    "Karamani": 115,
    "karanja seeds": 439,
    "Karbuja(Musk Melon)": 187,
    "Kartali (Kantola)": 305,
    "Kevda": 481,
    "Kharif Mash": 61,
    "Khirni": 514,
    "Khoya": 372,
    "Kinnow": 336,
    "Knool Khol": 177,
    "Kodo Millet(Varagu)": 117,
    "kokum": 458,
    "Kooth": 459,
    "Kuchur": 243,
    "Kulthi(Horse Gram)": 114,
    "Kutki": 415,
    "kutki": 426,
    "Ladies Finger": 155,
    "Laha": 515,
    "Lak(Teora)": 96,
    "Leafy Vegetable": 171,
    "Lemon": 310,
    "Lentil (Masur)(Whole)": 63,
    "Lilly": 378,
    "Lime": 180,
    "Limonia (status)": 470,
    "Linseed": 67,
    "Lint": 280,
    "liquor turmeric": 432,
    "Litchi": 351,
    "Little gourd (Kundru)": 302,
    "Long Melon(Kakri)": 304,
    "Lotus": 403,
    "Lotus Sticks": 339,
    "Lukad": 337,
    "Lupine": 479,
    "Ma.Inji": 504,
    "Mace": 107,
    "macoy": 427,
    "Mahedi": 411,
    "Mahua": 335,
    "Mahua Seed(Hippe seed)": 371,
    "Maida Atta": 288,
    "Maize": 4,
    "Mango": 20,
    "Mango (Raw-Ripe)": 172,
    "mango powder": 422,
    "Maragensu": 225,
    "Marasebu": 181,
    "Marget": 407,
    "Marigold(Calcutta)": 235,
    "Marigold(loose)": 405,
    "Marikozhunthu": 502,
    "Mash": 60,
    "Mashrooms": 340,
    "Masur Dal": 259,
    "Mataki": 93,
    "Methi Seeds": 47,
    "Methi(Leaves)": 46,
    "Millets": 237,
    "Mint(Pudina)": 360,
    "Moath Dal": 258,
    "Moath Dal": 95,
    "Mousambi(Sweet Lime)": 77,
    "Muesli": 446,
    "Muleti": 428,
    "Muskmelon Seeds": 516,
    "Mustard": 12,
    "Mustard Oil": 324,
    "Myrobolan(Harad)": 142,
    "Nargasi": 245,
    "Nearle Hannu": 222,
    "Neem Seed": 126,
    "Nelli Kai": 223,
    "Nerium": 500,
    "nigella seeds": 445,
    "nigella seeds": 424,
    "Niger Seed (Ramtil)": 98,
    "Nutmeg": 106,
    "Onion": 23,
    "Onion Green": 358,
    "Orange": 18,
    "Orchid": 381,
    "Other green and fresh vegetables": 420,
    "Other Pulses": 97,
    "Ox": 213,
    "Paddy(Dhan)(Basmati)": 414,
    "Paddy(Dhan)(Common)": 2,
    "Palash flowers": 441,
    "Papaya": 72,
    "Papaya (Raw)": 313,
    "Patti Calcutta": 404,
    "Peach": 331,
    "Pear(Marasebu)": 330,
    "Peas cod": 308,
    "Peas Wet": 174,
    "Peas(Dry)": 347,
    "Pegeon Pea (Arhar Fali)": 301,
    "Pepper garbled": 109,
    "Pepper ungarbled": 110,
    "Perandai": 489,
    "Persimon(Japani Fal)": 327,
    "Pigs": 220,
    "Pineapple": 21,
    "pippali": 431,
    "Plum": 329,
    "Pointed gourd (Parval)": 303,
    "Polherb": 240,
    "Pomegranate": 190,
    "Poppy capsules": 425,
    "poppy seeds": 421,
    "Potato": 24,
    "Pumpkin": 84,
    "Pundi": 254,
    "Pundi Seed": 128,
    "Pupadia": 447,
    "Raddish": 161,
    "Ragi (Finger Millet)": 30,
    "Raibel": 409,
    "Rajgir": 248,
    "Rala": 517,
    "Ram": 282,
    "Ramphal": 518,
    "Rat Tail Radish (Mogari)": 307,
    "Ratanjot": 460,
    "Raya": 65,
    "Rayee": 519,
    "Red Cabbage": 493,
    "Red Gram": 7,
    "Resinwood": 322,
    "Riccbcan": 62,
    "Rice": 3,
    "Ridgeguard(Tori)": 160,
    "Rose(Local)": 374,
    "Rose(Loose))": 228,
    "Rose(Tata)": 373,
    "Round gourd": 306,
    "Rubber": 111,
    "Sabu Dan": 291,
    "Safflower": 59,
    "Saffron": 338,
    "Sajje": 271,
    "salvia": 468,
    "Same/Savi": 122,
    "sanay": 433,
    "Sandalwood": 450,
    "Sarasum": 247,
    "Season Leaves": 277,
    "Seegu": 253,
    "Seemebadnekai": 176,
    "Seetapal": 201,
    "Sesamum(Sesame,Gingelly,Til)": 11,
    "sevanti": 464,
    "She Buffalo": 217,
    "She Goat": 283,
    "Sheep": 218,
    "Siddota": 183,
    "Siru Kizhagu": 490,
    "Skin And Hide": 226,
    "Snakeguard": 156,
    "Soanf": 135,
    "Soapnut(Antawala/Retha)": 207,
    "Soha": 520,
    "Soji": 286,
    "Sompu": 246,
    "Soyabean": 13,
    "spikenard": 455,
    "Spinach": 342,
    "Sponge gourd": 311,
    "Squash(Chappal Kadoo)": 332,
    "stevia": 440,
    "stone pulverizer": 430,
    "Sugar": 48,
    "Sugarcane": 150,
    "Sundaikai": 488,
    "Sunflower": 14,
    "Sunflower Seed": 285,
    "Sunhemp": 139,
    "Suram": 242,
    "Surat Beans (Papadi)": 300,
    "Suva (Dill Seed)": 255,
    "Suvarna Gadde": 178,
    "Sweet Potato": 152,
    "Sweet Pumpkin": 173,
    "Sweet Sultan": 466,
    "sweet william": 477,
    "T.V. Cumbu": 120,
    "Tamarind Fruit": 261,
    "Tamarind Seed": 208,
    "Tapioca": 100,
    "Taramira": 76,
    "Tea": 44,
    "Tender Coconut": 200,
    "Thinai (Italian Millet)": 116,
    "Thogrikai": 170,
    "Thondekai": 162,
    "Tinda": 349,
    "Tobacco": 141,
    "Tomato": 78,
    "Torchwood": 323,
    "Toria": 66,
    "Tube Flower": 234,
    "Tube Rose(Double)": 401,
    "Tube Rose(Loose)": 408,
    "Tube Rose(Single)": 377,
    "Tulasi": 503,
    "tulip": 463,
    "Turmeric": 39,
    "Turmeric (raw)": 309,
    "Turnip": 341,
    "vadang": 436,
    "Vatsanabha": 437,
    "Walnut": 343,
    "Water Apple": 496,
    "Water chestnut": 521,
    "Water Melon": 73,
    "Wax": 522,
    "Wheat": 1,
    "Wheat Atta": 287,
    "White Muesli": 429,
    "White Peas": 412,
    "White Pumpkin": 158,
    "Wild lemon": 498,
    "Wood": 203,
    "Wood Apple": 499,
    "Wool": 348,
    "Yam": 244,
    "Yam (Ratalu)": 297}
    states = {
    "Andaman and Nicobar": "AN",
    "Andhra Pradesh": "AP",
    "Arunachal Pradesh": "AR",
    "Assam": "AS",
    "Bihar": "BI",
    "Chandigarh": "CH",
    "Chattisgarh": "CG",
    "Dadra and Nagar Haveli": "DN",
    "Daman and Diu": "DD",
    "Goa": "GO",
    "Gujarat": "GJ",
    "Haryana": "HR",
    "Himachal Pradesh": "HP",
    "Jammu and Kashmir": "JK",
    "Jharkhand": "JR",
    "Karnataka": "KK",
    "Kerala": "KL",
    "Lakshadweep": "LD",
    "Madhya Pradesh": "MP",
    "Maharashtra": "MH",
    "Manipur": "MN",
    "Meghalaya": "MG",
    "Mizoram": "MZ",
    "Nagaland": "NG",
    "NCT of Delhi": "DL",
    "Odisha": "OR",
    "Pondicherry": "PC",
    "Punjab": "PB",
    "Rajasthan": "RJ",
    "Sikkim": "SK",
    "Tamil Nadu": "TN",
    "Telangana": "TL",
    "Tripura": "TR",
    "Uttar Pradesh": "UP",
    "Uttrakhand": "UC",
    "West Bengal": "WB"}
    try:
        url = f"https://agmarknet.gov.in/SearchCmmMkt.aspx?Tx_Commodity={commodities[commodity]}&Tx_State={states[state]}&Tx_District=0&Tx_Market=0&DateFrom=24-Mar-2025&DateTo=31-Mar-2025&Fr_Date=24-Mar-2025&To_Date=31-Mar-2025&Tx_Trend=0&Tx_CommodityHead={commodity}&Tx_StateHead={state}&Tx_DistrictHead=--Select--&Tx_MarketHead=--Select--"
        print(url)
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        data = get_data(soup)
        print(data)
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(
            status_code=400, content={"error": f"Error fetching data: {str(e)}"}
        )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
