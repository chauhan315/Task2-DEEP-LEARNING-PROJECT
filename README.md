*COMPANY*: CODETECH IT SOLUTIONS

*NAME*: ANKUR SINGH CHAUHAN

*INTERN ID*: CT12WV42

*DOMAIN*: DATA SCIENCE

*DURATION*: 12 WEEKS

*MENTOR*: NEELA SANTOSH

# Vehicle Damage Classification using Custom CNN and OpenCV

This project focuses on building a **Vehicle Damage Classification** system using a custom Convolutional Neural Network (CNN) with PyTorch and OpenCV. The goal is to classify vehicle images into two categories: `damaged` and `whole`. The entire pipeline includes dataset preparation, custom OpenCV-based preprocessing, model training, evaluation, and visualization using Grad-CAM.

---

## Table of Contents

1. Introduction
2. Objective
3. Dataset
4. Project Structure
5. Data Preprocessing
6. Model Architecture
7. Training Process
8. Evaluation
9. Grad-CAM Visualization
10. Single Image Prediction
11. Future Improvements
12. Conclusion
13. Requirements
14. How to Run the Project

---

## 1. Introduction

Damage detection in vehicles is a vital task for the automobile industry, particularly in insurance and automated vehicle inspection systems. By classifying images into categories such as damaged and undamaged (whole), this project aims to create a basic deep learning solution using a custom-built CNN instead of relying on pretrained models like ResNet or VGG.

---

## 2. Objective

The main objectives of this project are:

- Download and organize a vehicle damage dataset.
- Apply image preprocessing techniques using OpenCV.
- Build and train a custom CNN classifier.
- Evaluate model performance using metrics and visualizations.
- Apply Grad-CAM for model interpretability.
- Predict the class for new images.

---

## 3. Dataset

The dataset used in this project is sourced from Kaggle:

- Dataset Name: `car-damage-detection`
- Author: anujms
- Link: https://www.kaggle.com/datasets/anujms/car-damage-detection

The dataset contains vehicle images categorized into two folders:

- `00-damage` – Images showing damaged vehicles.
- `01-whole` – Images showing undamaged (whole) vehicles.

These are further divided into `training` and `validation` folders.

---
## 4. Components


Other project components include:

- Custom dataset loader using OpenCV
- CNN model definition
- Training and validation loops
- Evaluation using classification metrics
- Grad-CAM for interpretability
- Single image prediction function

---

## 5. Data Preprocessing

Instead of using the typical `transforms.Compose` pipeline, we use OpenCV for preprocessing. This allows finer control over operations like contrast enhancement.

### Steps:

1. **Read image with OpenCV**
2. **Resize to 224x224**
3. **Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)**
4. **Convert LAB back to RGB**
5. **Normalize and convert to PyTorch tensor**

A custom transformation class `OpenCVTransform` and a custom loader `OpenCVImageFolder` are implemented to integrate this into PyTorch’s `ImageFolder`.

---

## 6. Model Architecture

The model `CarDamageCNN` is a custom CNN designed to classify RGB images into two categories.

### Architecture:

- **Conv Block 1:** Conv2D (3 → 32) → BatchNorm → ReLU → MaxPool
- **Conv Block 2:** Conv2D (32 → 64) → BatchNorm → ReLU → MaxPool
- **Conv Block 3:** Conv2D (64 → 128) → BatchNorm → ReLU → MaxPool
- **Flatten**
- **Fully Connected 1:** Linear → ReLU → Dropout
- **Fully Connected 2:** Linear → Output (2 classes)

This model has about 2.4 million trainable parameters and is lightweight compared to deeper pretrained models.

---

## 7. Training Process

### Loss Function:
- `CrossEntropyLoss` is used since this is a multi-class classification task with two classes.

### Optimizer:
- `Adam` optimizer is chosen for its adaptive learning rate property.

### Scheduler:
- A `StepLR` scheduler is used to reduce the learning rate every 5 epochs to allow better convergence.

### Early Stopping:
- Implemented to prevent overfitting, it stops training if validation accuracy does not improve for 3 consecutive epochs.

### Batch Size:
- 32

### Number of Epochs:
- Up to 20 epochs, with early stopping enabled.

### Training Output:
During training, both training and validation loss/accuracy are printed and logged. The model state with the highest validation accuracy is saved to disk as `best_car_damage_model.pth`.

A plot is generated after training showing both the loss and accuracy curves for training and validation sets.

---

## 8. Evaluation

After training, the model is evaluated using the validation set.

### Metrics used:
- Classification report (Precision, Recall, F1-score, Support)
- Confusion matrix heatmap

These are computed using `sklearn.metrics` functions.

The confusion matrix gives a clear idea of how well the model is distinguishing between damaged and whole cars.

---

## 9. Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize which parts of the image are influencing the model's decision.

### How it works:
- Targets the last convolutional layer `conv_block3[0]`
- Gradients of the class score are used to create a heatmap
- The heatmap is overlaid on the original image

### Purpose:
- Helps in understanding model decisions
- Useful for model debugging and transparency

Grad-CAM visualizations are especially helpful when dealing with real-world applications such as damage detection, where transparency is critical.

---

## 10. Single Image Prediction

A function `predict_single_image()` allows for easy prediction of a new image:

- Takes the image path and the trained model.
- Applies the same OpenCV-based preprocessing.
- Returns the predicted class label (either `damaged` or `whole`).

This function is useful for integrating with APIs or UI components.

---

## 11. Future Improvements

Several potential improvements could enhance this project:

- Use data augmentation (rotation, brightness, flipping) to increase dataset diversity.
- Integrate pretrained models like ResNet18 or MobileNet for better performance.
- Use Grad-CAM++ for sharper attention maps.
- Export the model as TorchScript or ONNX for production use.
- Deploy via Flask, FastAPI, or Streamlit.
- Add a confidence score or threshold-based classification.

---

## 12. Conclusion

This project demonstrates the power of building a custom CNN pipeline tailored for binary image classification. Using OpenCV for preprocessing gives additional control over image enhancement. The model achieves good accuracy and generalization on validation data.

Additionally, Grad-CAM provides valuable insights into model decisions, improving interpretability and trust.

The modular design of the pipeline allows for easy experimentation and extension, making this a strong foundation for more advanced applications in vehicle inspection or insurance automation.

---

## 13. Requirements

This project uses the following libraries:

- torch
- torchvision
- numpy
- matplotlib
- opencv-python
- scikit-learn
- pytorch-grad-cam
- tqdm

Install them with:
pip install torch torchvision opencv-python matplotlib scikit-learn pytorch-grad-cam tqdm


---

## 14. How to Run the Project

1. Clone the repository or download the code.
2. Upload your `kaggle.json` API token file.
3. Run the script to download and organize the dataset.
4. Train the model using the `train_model()` function.
5. Evaluate performance using `evaluate_model()`.
6. Run Grad-CAM visualizations on selected images.
7. Use the prediction function for real-world use.

Ensure you are using a GPU runtime (like Google Colab) for faster training.

---

**End of README**



