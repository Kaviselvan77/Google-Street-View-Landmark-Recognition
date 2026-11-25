# Google Street View – Landmark Recognition System

A CNN-based image classification project with a Flask web interface. Upload a landmark image and the trained model predicts its landmark class.

> **Important:** This ZIP contains the complete project code and setup files, but not the image dataset or a trained model. Landmark datasets can be very large and may have usage/licensing conditions. Add your permitted dataset, train the model, and the application will load the generated model automatically.

## Features
- Train a CNN using images arranged in class folders.
- Automatic image resizing and normalization.
- Data augmentation during training.
- Validation accuracy/loss plots and classification report.
- Flask web app for image upload and prediction confidence.
- Saves the trained model and class labels.

## 1. Requirements
- Windows 10/11
- Python 3.10 or 3.11 recommended
- VS Code (optional)
- At least 8 GB RAM recommended; GPU is optional.

## 2. Setup

Open this folder in VS Code and run:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt or run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## 3. Add your dataset

Place images in this structure. Each subfolder name becomes a predicted landmark label:

```text
dataset/
  train/
    Eiffel_Tower/
      image001.jpg
      image002.jpg
    Taj_Mahal/
      image001.jpg
  validation/
    Eiffel_Tower/
      image101.jpg
    Taj_Mahal/
      image101.jpg
```

- Use the **same class folder names** in `train` and `validation`.
- Each class should contain multiple images.
- Use a reasonable validation split (for example, 80% train / 20% validation).
- The number of classes is detected automatically; it does not have to be 186.
- Only use images you are allowed to use. This project does not download Google Street View imagery automatically.

## 4. Train the model

```bash
python train_model.py
```

Optional settings:
```bash
python train_model.py --epochs 20 --batch-size 32 --image-size 128
```

Training creates:
- `models/landmark_model.keras`
- `models/class_names.json`
- `results/training_curves.png`

## 5. Evaluate

```bash
python evaluate_model.py
```

Creates:
- `results/classification_report.txt`
- `results/confusion_matrix.png`

The evaluation script reports accuracy, precision, recall and F1-score on the validation folder.

## 6. Run the website

After training completes:

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser. Upload a JPG, JPEG or PNG image.

## Project structure

```text
Google_Street_View_Landmark_Recognition/
├── app.py
├── train_model.py
├── evaluate_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── dataset/
│   ├── train/
│   └── validation/
├── models/
├── results/
├── static/
│   ├── css/style.css
│   └── uploads/
└── templates/
    ├── index.html
    └── result.html
```

## Methodology

1. Load images from class-labelled directories.
2. Resize images to 128×128 RGB.
3. Apply random flipping, rotation, zoom and contrast augmentation to training batches.
4. Normalize pixel values to [0, 1].
5. Train a CNN with convolution, max-pooling, dropout and softmax classification layers.
6. Evaluate on a held-out validation set.
7. Use the trained model to classify an uploaded image.

## Limitations and improvements

- A prediction is only as reliable as the dataset, class balance and image quality.
- Validation results are not a substitute for testing on genuinely unseen images.
- Similar-looking landmarks may be confused.
- Possible improvements: transfer learning (MobileNetV2/EfficientNet), more balanced data, stronger augmentation, fine-tuning, and a separate test set.

## Troubleshooting

- **Model not found:** run `python train_model.py` successfully first.
- **Dataset folder empty:** check that images are inside class subfolders, not directly inside `train` or `validation`.
- **Class mismatch:** make sure train and validation contain matching class folders.
- **TensorFlow installation issue:** use Python 3.10/3.11 in a fresh virtual environment and install from `requirements.txt`.

## Academic integrity / results

Do not report invented accuracy values. Run training and evaluation on your dataset and use the actual generated report and plots in your submission.
