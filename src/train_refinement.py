import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import copy

# Import your modules
from dataset import HAM10000Dataset
from models import get_backbone
from ema import EMA
from utils import set_seed

def train_one_epoch(student, teacher, ema, loader, optimizer, device, consistency_weight=0.1):
    student.train()
    teacher.eval() # Teacher is always in eval mode
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, desc="[TEACHER-STUDENT]", leave=False)
    
    for images, labels, _ in loop:
        images, labels = images.to(device), labels.to(device)
        
        # 1. Student Forward Pass
        student_logits = student(images)
        
        # 2. Teacher Forward Pass (No Grad)
        with torch.no_grad():
            teacher.logits = teacher(images)
            
        # 3. Supervised Loss (Student vs Real Labels)
        cls_loss = F.cross_entropy(student_logits, labels)
        
        # 4. Consistency Loss (Student vs Teacher)
        # KL Divergence: aligns student probability distribution with teacher's
        student_log_softmax = F.log_softmax(student_logits, dim=1)
        teacher_softmax = F.softmax(teacher.logits, dim=1)
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
        _, preds = torch.max(student_logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        loop.set_postfix(loss=loss.item(), acc=correct/total)
        
    return running_loss / len(loader), correct / total

def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return running_loss / len(loader), correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_train", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Phase 7 model")
    parser.add_argument("--output_dir", type=str, default="experiments/Phase8_Refinement")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5) # Low LR for fine-tuning
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🚀 Phase 8: Refinement starting from {args.checkpoint}")

    # 1. Load Data
    train_dataset = HAM10000Dataset(args.csv_train, args.image_dir, transform="train")
    val_dataset = HAM10000Dataset("data/splits/val_fold_0.csv", args.image_dir, transform="val")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # 2. Initialize Student (Load Phase 7 weights)
    student = get_backbone("resnet50", num_classes=7).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle state dict keys if needed
    state_dict = checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    student.load_state_dict(new_state_dict, strict=False)

    # 3. Initialize Teacher (Copy of Student)
    teacher = copy.deepcopy(student)
    teacher.eval() # Teacher is always eval
    for param in teacher.parameters():
        param.requires_grad = False # Teacher does not learn via gradient
        
    # 4. Setup EMA
    ema = EMA(teacher, decay=0.99)
    ema.register()

    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    
    best_acc = 0.0

    # 5. Training Loop
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        train_loss, train_acc = train_one_epoch(student, teacher, ema, train_loader, optimizer, device)
        val_loss, val_acc = validate(student, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        
        # Save Best Student
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.output_dir, "best_model_refinement.pth")
            torch.save(student.state_dict(), save_path)
            print(f"🌟 New Best Model Saved! ({val_acc:.4f})")

    print(f"\n✅ Phase 8 Complete. Best Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()