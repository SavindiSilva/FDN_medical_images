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
from utils import seed_everything, load_config
from ema import EMA 

# --- CONFIG ---
TEACHER_DECAY = 0.999  # How slowly the teacher updates
SUSPECT_WEIGHT = 0.5   # How much we trust the teacher on suspect samples

def train_one_epoch(student, teacher_ema, loader, optimizer, clean_ids, device):
    student.train()
    total_loss = 0
    
    # Loss functions
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    
    loop = tqdm(loader, leave=False)
    for batch_idx, (images, labels, image_ids) in enumerate(loop):
        images, labels = images.to(device), labels.to(device)
        
        # 1. Student Forward Pass
        student_logits = student(images)
        
        # 2. Split Batch into Clean vs Suspect
        # We check if the image_id is in our known 'clean_ids' set
        is_clean = torch.tensor([uid in clean_ids for uid in image_ids], device=device)
        
        loss = 0
        
        # --- A: CLEAN SAMPLES (Trust the Label) ---
        if is_clean.any():
            clean_logits = student_logits[is_clean]
            clean_labels = labels[is_clean]
            loss_clean = criterion_ce(clean_logits, clean_labels)
            loss += loss_clean
        
        # --- B: SUSPECT SAMPLES (Trust the Teacher) ---
        if (~is_clean).any():
            suspect_images = images[~is_clean]
            suspect_logits = student_logits[~is_clean]
            
            with torch.no_grad():
                # Get Teacher Prediction (Soft Targets)
                # We essentially perform a temporary update to get teacher state
                teacher_ema.apply_shadow() 
                teacher_logits = teacher_ema.model(suspect_images)
                teacher_ema.restore()
                
            # Soft Label Refinement (KL Divergence)
            # Student tries to match Teacher's probability distribution
            loss_suspect = criterion_kl(
                F.log_softmax(suspect_logits, dim=1), 
                F.softmax(teacher_logits, dim=1)
            )
            loss += SUSPECT_WEIGHT * loss_suspect

        # 3. Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 4. Update Teacher
        teacher_ema.update()
        
        total_loss += loss.item()
        loop.set_description(f"Loss: {loss.item():.4f}")

    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy_csv", type=str, required=True, help="Full dataset (with noise)")
    parser.add_argument("--clean_csv", type=str, required=True, help="Result from Phase 4 (Sieved)")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="experiments/Phase6_Refinement")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    seed_everything(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Data
    # We load the FULL dataset for training
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_df = pd.read_csv(args.noisy_csv)
    dataset = HAM10000Dataset(full_df, args.image_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    
    # 2. Identify Clean Samples
    # We load the cleaned CSV just to get the IDs
    clean_df = pd.read_csv(args.clean_csv)
    clean_ids = set(clean_df['image_id'].values) # O(1) lookup
    
    print(f"   Refinement Setup:")
    print(f"   Total Training Samples: {len(full_df)}")
    print(f"   Trusted (Clean) Samples: {len(clean_ids)}")
    print(f"   Suspect (Refined) Samples: {len(full_df) - len(clean_ids)}")

    # 3. Setup Student and Teacher
    student = get_model("resnet50", num_classes=7).to(device)
    teacher_ema = EMA(student, decay=TEACHER_DECAY) # Teacher starts as copy of student
    
    optimizer = optim.AdamW(student.parameters(), lr=1e-4)

    # 4. Training Loop
    best_acc = 0.0
    
    print(" Starting Phase 6: Adaptive Refinement...")
    for epoch in range(args.epochs):
        loss = train_one_epoch(student, teacher_ema, loader, optimizer, clean_ids, device)
        
        # Evaluate Student (Validation skipped for brevity, just saving checkpoints)
        # In a real run, pass a val_loader here
        
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {loss:.4f}")
        
        # Save every few epochs
        if (epoch+1) % 5 == 0:
            path = os.path.join(args.output_dir, f"refinement_epoch_{epoch+1}.pth")
            torch.save(student.state_dict(), path)
            print(f"   Saved: {path}")

    # Save Final
    torch.save(student.state_dict(), os.path.join(args.output_dir, "refinement_final.pth"))
    print("Phase 6 Refinement Complete.")

if __name__ == "__main__":
    main()