import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from dataset import BrainMRIDataset # dataset.py에서 불러옴
from model import get_model        # model.py에서 불러옴

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 모델 로드 (final_model.pth 파일이 있어야 함)
    model = get_model().to(device)
    
    # 모델 파일이 있는지 확인
    model_path = "final_model.pth"
    if not os.path.exists(model_path):
        print(f"🚨 오류: {model_path} 파일이 없습니다! train.py를 먼저 실행하세요.")
        exit()
        
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 2. 테스트용 전처리 (ResNet 규격)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 3. 데이터셋 연결
    test_dataset = BrainMRIDataset(root_dir="./dataset", split="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    correct = 0
    total = 0
    print("-" * 50)
    print("   [판독 결과]   |   [정답]   |  [판정]")
    
    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            is_correct = (predicted == labels).item()
            if is_correct: correct += 1
            total += 1
            
            # 틀린 것만 출력 (또는 다 출력)
            if not is_correct:
                file_name = os.path.basename(paths[0])
                pred = "종양" if predicted.item() == 1 else "정상"
                ans = "종양" if labels.item() == 1 else "정상"
                print(f"⚠️ 오답: {pred:^5} <-> {ans:^5} ({file_name})")

    print("-" * 50)
    print(f"🏆 최종 정확도: {100 * correct / total:.2f}%")