import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
from sklearn.model_selection import train_test_split

PATH = "./Crop_recommendation.csv"
df = pd.read_csv(PATH)
features = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
target = df["label"]
labels = df["label"]
Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    features, target, test_size=0.2, random_state=2
)
Ytrain_encoded = label_encoder.fit_transform(Ytrain)
XB = pickle.load(open("XGBoost.pkl", "rb"))
a = XB.predict([[30.0, 41.0, 15.0, 24.83206631, 44.17085032, 5.88509677, 52.0810886]])
print(label_encoder.inverse_transform(a))
