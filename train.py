import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# 👇 우리가 만든 파일들에서 불러오는 방식 (전문가 스타일)
from dataset import BrainMRIDataset 
from model import get_model        

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 장비: {device}")
    
    # 1. 전처리 설정 (ResNet 규격)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. 데이터셋 & 모델 준비 (import한 클래스 사용)
    train_dataset = BrainMRIDataset(root_dir="./dataset", split="train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = get_model().to(device) # model.py에서 가져온 함수

    # 3. 학습 도구
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 4. 학습 루프
    epochs = 20
    print(f"📚 최종 정리된 코드로 학습 시작! (Total Epochs: {epochs})")
    print("=" * 50)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels, _ in train_loader: # path는 학습 때 필요 없음
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        acc = 100 * correct / total
        print(f"[{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")

    # 👇 여기가 핵심! 파일명이 final_model.pth로 바뀝니다.
    torch.save(model.state_dict(), "final_model.pth")
    print("💾 모델 저장 완료: final_model.pth")