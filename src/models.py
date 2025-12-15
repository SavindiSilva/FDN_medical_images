import torch.nn as nn
import timm

def get_model(model_name='resnet50', num_classes=7):
    print(f"loading model: {model_name}...")
    #downloads the ResNet structure
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model