import torch
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms, models
from torch.utils.data import DataLoader
from dataset import HAM10000Dataset

# CONFIG
# Point to the Best Model saved during training
MODEL_PATH = "experiments/Phase6_Fixed/best_model.pth" 
IMG_DIR = "/content/final_images"
VAL_CSV = "data/splits/val_fold_0.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    print(f" Evaluating model from: {MODEL_PATH}")
    
    # 1. Load Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_dataset = HAM10000Dataset(VAL_CSV, IMG_DIR, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 2. Load Model Structure
    model = models.resnet50(weights=None) # No weights needed, we load our own
    model.fc = nn.Linear(model.fc.in_features, 7)
    
    # 3. Load Trained Weights
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    
    # 4. Get Predictions
    all_preds = []
    all_labels = []
    
    print(" Running predictions on Validation Set...")
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. Generate Report
    classes = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Phase 6')
    plt.show()
    
    # Classification Report
    print("\n Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=classes))

if __name__ == "__main__":
    evaluate()