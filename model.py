import torch.nn as nn
import torchvision.models as models

def get_model():
    # 1. ResNet18 천재 모델 불러오기
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 2. 마지막 출력층을 우리 문제(2개 분류)에 맞게 개조
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    return model