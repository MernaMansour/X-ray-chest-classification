# Chest X-Ray Classification Using a Convolutional Neural Network

A deep learning notebook exploring **four-class chest X-ray image classification** using a custom convolutional neural network (CNN) built with TensorFlow/Keras.

The notebook covers dataset preparation, exploratory visualization, CNN architecture design, training, and model evaluation. It was developed for **Google Colab** and includes Google Drive and Kaggle setup steps.

> **Disclaimer:** This is an educational machine-learning project, not a clinically validated diagnostic tool. Do not use its predictions for medical decisions.

## Classification Task

The model is designed to predict one of four chest X-ray categories:

- **COVID-19**
- **Normal**
- **Pneumonia-Bacterial**
- **Pneumonia-Viral**

## Dataset

The notebook downloads the Kaggle dataset [Curated Chest X-Ray Image Dataset for COVID-19](https://www.kaggle.com/datasets/unaissait/curated-chest-xray-image-dataset-for-covid19), and later reads a prepared four-class dataset from Google Drive at `/content/drive/MyDrive/covid-19/four_classes`.

The four-class folder is split into:

| Split | Ratio |
|---|---:|
| Training | 80% |
| Validation | 10% |
| Testing | 10% |

The splitting code uses `split-folders` with `seed=1337`. Images are loaded as **grayscale**, resized to **1024 × 1024 pixels**, and passed to Keras generators with a **batch size of 64**.

**Reproduction note:** The downloaded Kaggle archive and the Google Drive `four_classes` folder are separate paths in the notebook. To reproduce the experiment, prepare the four named class folders at the expected location or update the notebook paths. The notebook does not include the dataset files.

## Model Architecture

The implementation builds a **custom sequential CNN**, rather than using a pretrained backbone:

1. Convolutional layers with 32 and 64 filters, followed by batch normalization, ReLU, max pooling, and dropout.
2. Convolutional layers with 128 and 256 filters, followed by batch normalization, ReLU, max pooling, and dropout.
3. Two additional convolutional blocks using 512 filters, batch normalization, ReLU, max pooling, and dropout.
4. Flatten layer followed by dense layers with **256** and **512** units.
5. **Four-unit softmax output layer** for multiclass classification.

The model uses the **Adam** optimizer with a configured learning rate of `0.0001`, **categorical cross-entropy** loss, and **accuracy** as a metric. Training is configured for **50 epochs**, with a checkpoint intended to save the best model.

## Evaluation

The notebook includes code to:

- Plot training and validation **loss** and **accuracy**.
- Generate predictions on the held-out test set.
- Visualize a normalized **confusion matrix**.
- Produce a **classification report** using scikit-learn.
- Export the model architecture to `model.json`.

**Results:** The shared notebook does not contain saved execution outputs, so no numerical accuracy, precision, recall, or F1-score is claimed here. Run the notebook and record the resulting metrics before adding a performance table.

## Tools and Libraries

- Python, Google Colab, Jupyter Notebook
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- OpenCV
- scikit-learn
- `split-folders`
- Kaggle API

## How to Run

1. Open `neuralnetwork_project.ipynb` in **Google Colab**.
2. Obtain access to the dataset and configure the Kaggle API if using the download cells.
3. **Keep `kaggle.json` private**; do not commit credentials or tokens to GitHub.
4. Prepare the four-class image directory in Google Drive, or modify the notebook's input paths.
5. Install the required packages in the Colab environment and run the cells in order.
6. Examine the loss/accuracy plots, confusion matrix, and classification report.

> **Compatibility note:** This notebook uses some older Keras interfaces, including `fit_generator`, `predict_generator`, `Adam(lr=...)`, and `monitor='val_acc'`. These may need updating for modern TensorFlow/Keras releases (for example, `model.fit`, `model.predict`, `learning_rate=...`, and `val_accuracy`). The 1024 × 1024 input size and batch size of 64 may also require substantial GPU memory.

## Repository Structure

```text
.
├── neuralnetwork_project.ipynb   # Data preparation, CNN training and evaluation
└── README.md                      # Project documentation
```

## Potential Improvements

- Update deprecated Keras calls and verify compatibility with a current TensorFlow version.
- Make dataset paths configurable and clearly document dataset preparation.
- Add reproducible test-set metrics and training figures.
- Compare the custom CNN with transfer-learning models.
- Verify patient-level separation between training, validation, and test sets to limit data leakage.

## Project Scope

This repository demonstrates an end-to-end computer vision learning workflow for medical image classification. It is intended for research and educational exploration, **not clinical deployment**.
