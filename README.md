# Project: Create Your Own Image Classifier - TensorFlow

This repository contains the implementation of a custom image classifier using TensorFlow and Transfer Learning. The project focuses on classifying images from the **Oxford Flowers 102 dataset**, leveraging pre-trained models and fine-tuning them for better accuracy. The final model is saved as a Keras HDF5 file and can be used for inference via a command-line application.

---

## Table of Contents

- [Project: Create Your Own Image Classifier - TensorFlow](#project-create-your-own-image-classifier---tensorflow)
  - [Table of Contents](#-table-of-contents)
  - [Overview](#-overview)
  - [Dataset](#-dataset)
  - [Implementation Details](#️-implementation-details)
    - [Key Steps in the Implementation](#key-steps-in-the-implementation)
  - [Files in the Repository](#-files-in-the-repository)
    - [Main Files](#main-files)
    - [Data Files](#data-files)
    - [Output Files](#output-files)
  - [How to Run the Project](#-how-to-run-the-project)
    - [Prerequisites](#prerequisites)
    - [Steps to Run](#steps-to-run)
  - [Results and Visualizations](#-results-and-visualizations)
  - [Contributions](#-contributions)
  - [Contact](#-contact)
- [Thank you for checking out this project!](#thank-you-for-checking-out-this-project-)

---

## Overview

The objective of this project is to:

1. **Load and Preprocess the Dataset**:
   - Use TensorFlow Datasets to load the Oxford Flowers 102 dataset.
   - Split the dataset into training, validation, and testing sets.
   - Normalize and resize images for input to the model.

2. **Build a Transfer Learning Model**:
   - Load the pre-trained MobileNet model from TensorFlow Hub.
   - Freeze the pre-trained layers and add a new feedforward classifier layer with the appropriate number of output neurons (corresponding to the number of classes in the dataset).

3. **Train the Model**:
   - Configure the model for training using the `compile` method.
   - Train the model using the `fit` method, incorporating the validation set.
   - Monitor training progress by plotting loss and accuracy metrics.

4. **Evaluate the Model**:
   - Measure the model's accuracy on the test set.
   - Save the trained model as a Keras HDF5 file.

5. **Deploy the Model**:
   - Implement a command-line application (`predict.py`) that allows users to:
     - Predict the most likely class for an image.
     - Display the top-K most probable classes.
     - Map class indices to human-readable labels using a JSON file.

6. **Regularization**:
   - Apply at least one form of regularization (e.g., dropout, L2 regularization) to reduce overfitting and ensure that the difference between training and validation accuracy is minimal (≤3%).

---

## Dataset

The dataset used in this project is the **Oxford Flowers 102 dataset**, which contains:
- **Classes**: 102 flower categories.
- **Images**: High-resolution images of flowers.
- **Split**: Divided into training, validation, and testing sets.

---

## Implementation Details

The project is implemented using Python with the following libraries:
- **TensorFlow**: For building and training the model.
- **TensorFlow Hub**: For loading pre-trained models like MobileNet.
- **NumPy**: For numerical operations.
- **Pandas**: For handling data structures.
- **Matplotlib**: For visualization.
- **JSON**: For mapping class indices to human-readable labels.

### Key Steps in the Implementation

1. **Data Loading and Preprocessing**:
   - Load the Oxford Flowers 102 dataset using TensorFlow Datasets.
   - Resize and normalize images to match the input requirements of MobileNet.
   - Split the dataset into training, validation, and testing sets.

2. **Transfer Learning**:
   - Load the pre-trained MobileNet model from TensorFlow Hub.
   - Freeze the pre-trained layers and add a new classifier layer with the appropriate number of output neurons.

3. **Model Training**:
   - Compile the model with an optimizer (e.g., Adam), loss function (e.g., categorical cross-entropy), and metrics (e.g., accuracy).
   - Train the model using the `fit` method, incorporating the validation set.
   - Apply regularization techniques (e.g., dropout) to prevent overfitting.

4. **Evaluation**:
   - Measure the model's accuracy on the test set.
   - Plot training and validation loss/accuracy curves using the history returned by the `fit` method.

5. **Model Saving and Loading**:
   - Save the trained model as a Keras HDF5 file.
   - Load the saved model for inference.

6. **Inference**:
   - Implement a `predict.py` script that:
     - Processes input images using the `process_image` function.
     - Uses the loaded model to predict the top-K classes for an image.
     - Maps class indices to human-readable labels using a JSON file.

7. **Visualization**:
   - Display the first image from the training set with its label.
   - Create a matplotlib figure showing an image and its top 5 predicted classes with actual flower names.

---

## Files in the Repository

The repository contains the following files:

### Main Files
- **`Project_Image_Classifier_Project.ipynb`**: Jupyter Notebook containing the implementation and training pipeline.
- **`predict.py`**: Command-line application for making predictions using the trained model.
- **`label_map.json`**: JSON file mapping class indices to human-readable labels.

### Data Files
- **`assets/`**: Folder containing assets used in the project.
- **`flower_classifier_model/`**: Folder for saving the trained model.
- **`test_images/`**: Folder containing sample images for testing.

### Output Files
- **`1738440917.h5`**: Trained Keras model saved as an HDF5 file.
- **`get-pip.py`**: Script for installing pip if not already installed.

---

## How to Run the Project

### Prerequisites
- **Python**: Ensure Python is installed on your machine.
- **Libraries**: Install the required libraries using `pip`:
  ```bash
  pip install tensorflow tensorflow-hub numpy pandas matplotlib json
  ```

### Steps to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/lameesbassam05/Image-Classifier-TensorFlow.git
   ```

2. **Navigate to the Directory**
   ```bash
   cd ImageClassifierProject
   ```

3. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook Project_Image_Classifier_Project.ipynb
   ```

4. **Train the Model**
   - Follow the steps in the Jupyter Notebook to load the dataset, build the model, train it, and save the results.

5. **Run the Command-Line Application**
   ```bash
   python predict.py --image <path_to_image> --model 1738440917.h5 --top_k 5 --category_names label_map.json
   ```

6. **View Results**
   - Training progress and evaluation metrics will be displayed in the Jupyter Notebook.
   - Prediction results will be printed to the console when running `predict.py`.

---

## Results and Visualizations

The project generates various outputs, including:
- **Training Progress**: Plots showing training and validation loss/accuracy.
- **Prediction Results**: Top-K classes and their probabilities for input images.
- **Visualization**: Matplotlib figures displaying images with their predicted classes.

Example visualizations include:
- Line plots showing training and validation loss/accuracy.
- Bar charts showing top-K predictions with class names.

---

## Contributions

If you'd like to contribute to this repository, feel free to:
1. Fork the repository.
2. Create a new branch for your changes.
3. Submit a pull request with a detailed explanation of your changes.

---

# Thank you for checking out this project!
