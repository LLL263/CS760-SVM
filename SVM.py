import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, ConfusionMatrixDisplay, classification_report, confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt
import itertools
import time

imageProcessStart = time.time()

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

imageProcessEnd = time.time()

print(f"number of training data: {len(imageDataTrain)}")
print(f"number of testing data: {len(imageDataTest)}")
print(f"image processing time: {imageProcessEnd - imageProcessStart}s")

trainingStart = time.time()
svm_model = OneVsRestClassifier(SVC(kernel='sigmoid', probability=True))
svm_model.fit(imageDataTrain, labels_train)
trainingEnd = time.time()
print(f"Training time: {trainingEnd - trainingStart}s")

predictStart = time.time()
predictions = svm_model.predict(imageDataTest)
predictEnd = time.time()
print(f"prediction time: {predictEnd - predictStart}s")

accuracies = []
y_true = labels_test[:,:] #117
pred = predictions[:, :]

for i in range(labels_test.shape[1]): 
    accuracies.append(accuracy_score(labels_test[:, i], predictions[:, i]))

average_accuracy = np.mean(accuracies)

print(f"Average model accuracy across all labels: {average_accuracy}")

labels = ["Aphids", "Early Blight", "Healthy Leaf-", "Leaf Curl", "Leafhoppers and Jassids", "Molds", "Mosaic Virus", "Septoria", "bactarial canker", "bacterial spot", "flea beetle", "late_blight", "leafminer", "powedry_mildew", "yellow curl virus"]
print(classification_report(y_true, pred, zero_division=1, target_names=labels))

# def draw_confusion_matrix(label_true, label_pred, label_name, title="Tomato Disease Confusion Matrix"):
#     cm = multilabel_confusion_matrix(label_true, label_pred)

#     plt.imshow(cm, cmap='Blues')
#     plt.title(title)
#     plt.xlabel("Predict label")
#     plt.ylabel("Truth label")
#     plt.yticks(range(len(label_name)), label_name)
#     plt.xticks(range(len(label_name)), label_name, rotation=45)
#     plt.tight_layout()
#     plt.colorbar()

#     for i in range(len(label_name)):
#         for j in range(len(label_name)):
#             color = (1,1,1) if i==j else (0,0,0)
#             value = float(format('%.2f' % cm[j,i]))
#             plt.text(i,j,value,  color=color)
    
#     plt.show()

# cnf_matrix = draw_confusion_matrix(np.array(y_true), np.array(pred), labels)
# print(cnf_matrix)

def draw_confusion_matrices(confusion_matrices, labels, cols):
    num_classes = len(labels)
    rows = int(np.ceil(num_classes / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))

    for i, (cm, label) in enumerate(zip(confusion_matrices, labels)):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.imshow(cm, cmap='Blues')
        ax.set_title(f'{label}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True Label')
        ax.set_xticks(range(2))
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_yticks(range(2))
        ax.set_yticklabels(['Negative', 'Positive'])

        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                ax.text(k, j, str(cm[j, k]), ha='center', va='center', color='red')

    for i in range(num_classes, rows*cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')

    plt.tight_layout()
    plt.show()


num_cols = 5 


cm = multilabel_confusion_matrix(y_true, pred)
draw_confusion_matrices(cm, labels, num_cols)