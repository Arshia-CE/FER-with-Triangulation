from delaunay import delaunay_triangulation
from face_recognition import detect_landmarks
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load and preprocess the dataset
# Path to the CK+ dataset directory
dataset_dir = "dataset"

label_map = {
    "anger": 0,
    "contempt": 1,
    "disgust": 2,
    "fear": 3,
    "happy": 4,
    "sadness": 5,
    "surprise": 6
}
# Load and preprocess the dataset
def load_ckplus_dataset(dataset_dir):
    images = []
    labels = []
    for subdir in os.listdir(dataset_dir):
        if subdir in label_map:  # Check if directory name is a valid label
            label = label_map[subdir]  # Get integer label from the mapping
            for filename in os.listdir(os.path.join(dataset_dir, subdir)):
                if filename.endswith(".png"):  # Load only image files (assuming images are in PNG format)
                    img_path = os.path.join(dataset_dir, subdir, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
                    img = cv2.resize(img, (100, 100))  # Resize image
                    images.append(img)
                    labels.append(label)

    return images, labels


# Load the dataset
images, labels = load_ckplus_dataset(dataset_dir)

# Step 2: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

X_train = np.array([img.flatten() for img in X_train])
X_test = np.array([img.flatten() for img in X_test])
# Step 3: Train the SVM classifier
svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_classifier.fit(X_train, y_train)

# Step 4: Evaluate the classifier
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Step 2: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Extract and flatten Delaunay triangulation features for training data
X_train_delaunay = [print(delaunay_triangulation(img, detect_landmarks(img)[1]).flatten()) for img in X_train]
X_train_delaunay = np.array(X_train_delaunay)
# Extract and flatten Delaunay triangulation features for testing data
X_test_delaunay = [delaunay_triangulation(img, detect_landmarks(img)[1]).flatten() for img in X_test]
X_test_delaunay = np.array(X_test_delaunay)


# Step 3: Train the SVM classifier
svm_classifier2 = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_classifier2.fit(X_train_delaunay, y_train)

# Step 4: Evaluate the classifier
y_pred = svm_classifier2.predict(X_test_delaunay)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Step 5: Save the trained classifier (optional)
# Save the model using joblib or pickle for future use

# Step 6: Deploy the classifier for real-time expression detection
# Load the saved classifier and use it to predict expressions in new images
