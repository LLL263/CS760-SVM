import numpy as np
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import image

# load dataset and get images name and labels
script_dir = os.path.dirname(os.path.abspath(__file__))
tomatoDatasetPathTrain = os.path.join(script_dir, "Tomato pest-diseases.v1-tomato_v1.multiclass/train/_classes.csv")
df = pd.read_csv(tomatoDatasetPathTrain)
tomatoImagesName = df["filename"]

# get labels for each image
labels = df.drop('filename', axis=1)
image_labels = labels.to_dict(orient='records')

imageDataTrain = []
for i in tomatoDatasetPathTrain.glob("*.jpg"):
    img = image.load_img(i, target_size=(32,32))
    print(img)