import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import os
import sys
from tqdm import tqdm

sys.path.append('src')
from dataset import HAM10000Dataset
from models import get_model
from utils import seed_everything
from ema import EMA 

# --- CONFIG ---
TEACHER_DECAY = 0.999
SUSPECT_WEIGHT = 0.5 

def train_one_epoch(student, teacher_ema, loader, optimizer, clean_ids, device):
    student.train()
    total_loss = 0
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    
    loop = tqdm(loader, leave=False)
    for images, labels, image_ids in loop:
        images, labels = images.to(device), labels.to(device)
        
        # FIX: Ensure IDs are strings and stripped of whitespace for matching
        # Check if the image_id is in the clean set
        is_clean = torch.tensor([str(uid).strip() in clean_ids for uid in image_ids], device=device)
        
        loss = torch.tensor(0.0, device=device)
        
        # A: CLEAN SAMPLES (Standard Label) -> Should generate NON-ZERO loss
        if is_clean.any():
            loss += criterion_ce(student_logits[is_clean], labels[is_clean])
        
        # B: SUSPECT SAMPLES (Teacher Refinement) -> Generates 0 loss at start (Normal)
        if (~is_clean).any():
            # 1. Student Forward Pass
            student_logits = student(images)

            suspect_images = images[~is_clean]
            with torch.no_grad():
                teacher_ema.apply_shadow() 
                teacher_logits = teacher_ema.model(suspect_images)
                teacher_ema.restore()
            
            loss += SUSPECT_WEIGHT * criterion_kl(
                F.log_softmax(student_logits[~is_clean], dim=1), 
                F.softmax(teacher_logits, dim=1)
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        teacher_ema.update()
        
        total_loss += loss.item()
        loop.set_description(f"Loss: {loss.item():.4f}")

    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy_csv", type=str, required=True)
    parser.add_argument("--clean_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="experiments/Phase6_Refinement")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--debug", action="store_true", help="Run in fast debug mode (1000 samples)")
    args = parser.parse_args()

    seed_everything(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" running on Device: {device}")

    # OPTIMIZATION: Small images for CPU speed
    img_size = 128 if device == "cpu" else 224
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_df = pd.read_csv(args.noisy_csv)
    
    # --- DEBUG MODE (Crucial for CPU) ---
    if args.debug:
        print(" DEBUG MODE ACTIVE: Training on random 1000 samples only.")
        full_df = full_df.sample(1000).reset_index(drop=True)

    dataset = HAM10000Dataset(full_df, args.image_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True) 
    
    # Load Clean IDs and strip whitespace
    clean_df = pd.read_csv(args.clean_csv)
    clean_ids = set(clean_df['image_id'].astype(str).str.strip().values)
    
    # OPTIMIZATION: Use ResNet18 for CPU
    model_name = "resnet18" if device == "cpu" else "resnet50"
    print(f" Model: {model_name} | Samples: {len(full_df)}")

    student = get_model(model_name, num_classes=7).to(device)
    teacher_ema = EMA(student, decay=TEACHER_DECAY)
    optimizer = optim.AdamW(student.parameters(), lr=1e-4)

    print(" Starting Phase 6: Adaptive Refinement...")
    for epoch in range(args.epochs):
        loss = train_one_epoch(student, teacher_ema, loader, optimizer, clean_ids, device)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {loss:.4f}")
        
    torch.save(student.state_dict(), os.path.join(args.output_dir, "refinement_final.pth"))
    print(" Phase 6 Refinement Complete.")

if __name__ == "__main__":
    main()