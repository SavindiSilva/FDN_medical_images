import torch
import torch.nn as nn
import timm

def get_model(model_name='resnet50', num_classes=7, pretrained=True):
    print(f"building {model_name}")
    try:
        #try loading from TIMM
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    except Exception as e:
        print(f"timm not found/error: {e}. using torchvision.")
        import torchvision.models as models
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    return model