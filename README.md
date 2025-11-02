# Food-Image-Classifier

Fine-Tuning ViT for Food Classification

This application is live on Streamlit Community Cloud.

[https://food-image-classifier-taqi-tallal.streamlit.app/](#)

## Project Overview

This project fine-tunes a Vision Transformer (ViT) model to classify images of food into 101 distinct categories. It is the final task in a transformer fine-tuning assignment, demonstrating the application of transformer architectures to computer vision.

Model: google/vit-base-patch16-224

Dataset: Food-41 (Food-101)

Fine-Tuned Model: syedtaqi/vit-food-classifier (Replace with your actual repo name)

## Objective

The goal was to adapt a pre-trained Vision Transformer, which was originally trained on the general-purpose ImageNet dataset (1000 classes), to become a specialist expert at identifying 101 different types of food.

## Methodology

Unlike the text-based tasks, this required an image processing pipeline.

Data Loading: The dataset was loaded using the official meta/train.txt (75,750 images) and meta/test.txt (25,250 images) files. This ensured we used the correct, balanced splits as intended by the dataset creators.

Image Processing: An AutoImageProcessor (the "tokenizer" for images) was used to resize all images to the ViT model's required input of 224x224 pixels and normalize their color channels.

Data Augmentation: To prevent overfitting and make the model more robust, the training images were randomly flipped horizontally and randomly cropped (RandomHorizontalFlip, RandomResizedCrop). The validation images were left un-augmented.

Model Head Replacement: We loaded the ViTForImageClassification model using ignore_mismatched_sizes=True. This critical step discarded the model's original 1000-class "head" and replaced it with a new, untrained 101-class head, which we then trained.

Training: The model was fine-tuned for 3 epochs using the Trainer, with load_best_model_at_end=True set to automatically save the checkpoint with the highest validation accuracy.

## Quantitative Results & Analysis

The training was extremely successful, achieving a peak accuracy of 90.6% on the validation set of 25,250 unseen images.

| Epoch | Training Loss | Validation Loss | Accuracy |
|-------|--------------|----------------|-----------|
| 1     | 0.8403      | 0.4512         | 0.8787   |
| 2     | 0.6440      | 0.3604         | 0.8991   |
| 3     | 0.5292      | 0.3318         | 0.9057   |

Analysis

High Accuracy (90.6%): This is a very strong, realistic, and believable score. It is not a "god model" (which would be ~99-100% and indicate a data leak), as it still made ~2,373 mistakes out of 25,250. This confirms a robust, well-generalized model.

Confusion Matrix: A plot of the confusion matrix showed a near-perfect, bright diagonal line, indicating that the high accuracy was distributed well across all 101 classes, with no major areas of confusion.

Sample Predictions: Qualitative testing (as seen in the app) confirms the model can easily distinguish between visually distinct dishes like "carrot_cake" and "beef_carpaccio."

## How to Run This App Locally

This app loads the model directly from the Hugging Face Hub.

### 1. Clone This Repository

```bash
git clone https://github.com/SyedTaqii/Food-Image-Classifier.git
cd Food-Image-Classifier
```

### 2. Install Dependencies

It is highly recommended to use a Python virtual environment. The project includes a `requirements.txt` file.

```bash
pip install -r requirements.txt
```

The `requirements.txt` for this project contains:

```
streamlit
torch
transformers
Pillow
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```


Your browser will automatically open, and the app will download the fine-tuned model from Hugging Face on the first run.