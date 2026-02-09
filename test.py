import torch
import torch.nn as nn
import os
import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models # 👈 Import 추가

# 1. 모델 함수 (Train과 동일)
def get_model():
    model = models.resnet18(weights=None) # 구조만 가져옴
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model

# 2. 데이터셋 (Train과 동일하게 Normalize 추가 필수!)
class BrainMRIDataset(Dataset):
    def __init__(self, root_dir, split="test"):
        self.data_dir = os.path.join(root_dir, split)
        self.normal_images = glob.glob(os.path.join(self.data_dir, "normal", "*.jpg"))
        self.tumor_images = glob.glob(os.path.join(self.data_dir, "tumor", "*.jpg"))
        self.all_images = self.normal_images + self.tumor_images
        self.labels = [0] * len(self.normal_images) + [1] * len(self.tumor_images)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # ⭐ 학습 때 쓴 정규화랑 똑같이 맞춰야 함!
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

# 3. 테스트 실행
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 로드
    model = get_model().to(device)
    try:
        # 가중치(Weights)만 불러옴
        model.load_state_dict(torch.load("resnet_brain_model.pth"))
        print("📂 ResNet(천재) 모델을 불러왔습니다!")
    except FileNotFoundError:
        print("🚨 모델 파일이 없습니다! train.py 먼저 실행하세요.")
        exit()

    model.eval()
    
    test_dataset = BrainMRIDataset(root_dir="./dataset", split="test")
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
            
            total += 1
            is_correct = (predicted == labels).item()
            if is_correct: correct += 1
            
            pred_str = "종양 발견" if predicted.item() == 1 else "정상 소견"
            label_str = "종양" if labels.item() == 1 else "정상"
            result_mark = "⭕" if is_correct else "❌"
            file_name = os.path.basename(paths[0])
            
            if not is_correct:
                 print(f"⚠️ 오답: {pred_str:^10} | {label_str:^8} | {result_mark} ({file_name})")
            else:
                 print(".", end="", flush=True)

    print("\n" + "-" * 50)
    print(f"🏆 최종 성적 (ResNet): {100 * correct / total:.2f}%")