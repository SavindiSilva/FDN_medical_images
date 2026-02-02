import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import argparse
import copy
import numpy as np

# Import modules
from dataset import HAM10000Dataset
from models import get_model
from ema import EMA
from utils import seed_everything
from sklearn.metrics import matthews_corrcoef, f1_score

def get_transforms(mode="train"):
    """
    Define standard transforms for ResNet50 (224x224).
    """
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            norm
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            norm
        ])

def train_one_epoch(student, teacher, ema, loader, optimizer, device, consistency_weight=1.0):
    student.train()
    teacher.eval() # Teacher is always in eval mode
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, desc="[TEACHER-STUDENT]", leave=False)
    
    for images, labels, _ in loop:
        images, labels = images.to(device), labels.to(device)
        
        #Student Forward Pass
        student_logits = student(images)
        
        # Teacher Forward Pass using EMA weights (No Grad)
        with torch.no_grad():
            ema.apply_shadow()                 # put EMA weights onto teacher
            teacher_logits = teacher(images)   # forward with EMA teacher
            ema.restore()                      # restore teacher to original (optional but clean)
            
        #Supervised Loss (Student vs Real Labels)
        cls_loss = F.cross_entropy(student_logits, labels)
        
        #Consistency Loss (Student vs Teacher)
        # KL Divergence: aligns student probability distribution with teacher's
        student_log_softmax = F.log_softmax(student_logits, dim=1)
        teacher_softmax = F.softmax(teacher_logits, dim=1)
        const_loss = F.kl_div(student_log_softmax, teacher_softmax, reduction='batchmean')
        
        # Total Loss
        loss = cls_loss + (consistency_weight * const_loss)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update Teacher (EMA)
        ema.update(student)
        
        # Metrics
        running_loss += loss.item()
        preds = torch.argmax(student_logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        loop.set_postfix(loss=loss.item(), acc=correct/total)
        
    return running_loss / len(loader), correct / total

def validate(model, loader, device):
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    acc = (all_preds == all_labels).mean()
    mcc = matthews_corrcoef(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return running_loss / len(loader), acc, mcc, macro_f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_train", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Phase 7 model")
    parser.add_argument("--output_dir", type=str, default="experiments/Phase8_Refinement")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5) # Low LR for fine-tuning
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f" Phase 8: Refinement starting from {args.checkpoint}")

    #Load Data with ACTUAL Transforms (Not Strings)
    train_dataset = HAM10000Dataset(
        csv_file=args.csv_train, 
        image_dir=args.image_dir, 
        transform=get_transforms("train") 
    )
    
    val_dataset = HAM10000Dataset(
        csv_file="data/splits/val_fold_0.csv", 
        image_dir=args.image_dir, 
        transform=get_transforms("val")   
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Initialize Student (Load Phase 7 weights)
    student = get_model("resnet50", num_classes=7).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle state dict keys
    state_dict = checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    
    # Clean keys
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    student.load_state_dict(new_state_dict, strict=False)

    #Initialize Teacher (Copy of Student)
    teacher = copy.deepcopy(student)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False 
        
    #Setup EMA
    ema = EMA(teacher, decay=0.99)
    ema.register()

    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    
    best_mcc  = -1.0

    # 5. Training Loop
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        train_loss, train_acc = train_one_epoch(student, teacher, ema, train_loader, optimizer, device)
        val_loss, val_acc, val_mcc, val_f1 = validate(student, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | MCC: {val_mcc:.4f} | Macro-F1: {val_f1:.4f}")
        
        # Save Best Student
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            save_path = os.path.join(args.output_dir, "new_best_model_refinement.pth")
            torch.save(student.state_dict(), save_path)
            print(f"🌟 New Best Model Saved! ({val_mcc:.4f})")

    print(f"\nPhase 8 Complete. Best MCC: {best_mcc:.4f}")

if __name__ == "__main__":
    main()