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

#pointing to the Phase 7 Curriculum Model
MODEL_PATH = "experiments/Phase7_Curriculum/best_model_curriculum.pth"
IMG_DIR = "/content/final_images"
VAL_CSV = "data/splits/val_fold_0.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    print(f" Evaluating model from: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(" Error: Model file not found. Did Phase 7 finish saving?")
        return

    #Load Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_dataset = HAM10000Dataset(VAL_CSV, IMG_DIR, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    #Load Model Structure
    model = models.resnet50(weights=None) 
    model.fc = nn.Linear(model.fc.in_features, 7)
    
    #Load Trained Weights
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    
    #Get Predictions
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

    #Generate Report
    classes = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
    
    print("\n Classification Report (Phase 7 - Curriculum):\n")
    print(classification_report(all_labels, all_preds, target_names=classes))

if __name__ == "__main__":
    evaluate()