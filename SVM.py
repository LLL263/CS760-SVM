import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from pathlib import Path


script_dir = Path(__file__).parent  
train_dir = script_dir / "train"
test_dir = script_dir / "test"


df_train = pd.read_csv(train_dir / "_classes.csv")
df_test = pd.read_csv(test_dir / "_classes.csv")

imageDataTrain = []
imageDataTest = []

labels_train = df_train.iloc[:, 1:].values  

labels_test = df_test.iloc[:, 1:].values


for filename in df_train["filename"]:
    img_path = train_dir / filename
    if img_path.is_file():
        img_array = mpimg.imread(img_path)
        img_array = np.resize(img_array, (8, 8, 3))
        img_array = img_array.flatten() / 255.0
        imageDataTrain.append(img_array)


for filename in df_test["filename"]:
    img_path = test_dir / filename
    if img_path.is_file():
        img_array = mpimg.imread(img_path)
        img_array = np.resize(img_array, (8, 8, 3))
        img_array = img_array.flatten() / 255.0
        imageDataTest.append(img_array)

imageDataTrain = np.array(imageDataTrain)
imageDataTest = np.array(imageDataTest)

svm_model = OneVsRestClassifier(SVC(kernel='linear', probability=True))
svm_model.fit(imageDataTrain, labels_train)

predictions = svm_model.predict(imageDataTest)

accuracies = []
for i in range(labels_test.shape[1]):  
    accuracies.append(accuracy_score(labels_test[:, i], predictions[:, i]))
average_accuracy = np.mean(accuracies)

print(f"Average model accuracy across all labels: {average_accuracy}")
