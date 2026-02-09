# 🧠 Brain Tumor Detection with ResNet & CNN

## 📌 Project Overview
이 프로젝트는 뇌 MRI 스캔 이미지를 분석하여 **뇌종양(Brain Tumor)** 여부를 자동으로 판별하는 딥러닝 모델입니다.
초기에는 간단한 CNN 모델을 구축했으나, 성능 향상을 위해 **ResNet18 전이 학습(Transfer Learning)**을 도입했습니다.

## 🛠️ Tech Stack
* **Language:** Python 3.8+
* **Framework:** PyTorch
* **Model:** Simple CNN (Custom), ResNet18 (Pre-trained)
* **Environment:** WSL2 (Ubuntu), NVIDIA RTX 4060 Ti

## 📂 Dataset
* **Source:** Kaggle Brain MRI Images for Brain Tumor Detection
* **Structure:**
    * `train`: Normal / Tumor images (Agumentation applied)
    * `test`: Normal / Tumor images (Original)

## 🚀 Performance
| Model | Training Acc | Test Acc | Note |
|-------|--------------|----------|------|
| Simple CNN | 99.28% | 69.70% | 과적합(Overfitting) 발생 |
| **ResNet18** | **100.00%** | **81.82%** | **최종 채택 모델** |

### 🔍 Failure Analysis
* **특이사항:** `no 92.jpg`와 같은 특정 이미지는 모델 구조를 변경(CNN -> ResNet)해도 지속적으로 **False Positive(정상을 종양으로 오진)**가 발생함.
* **원인 추정:** 육안으로 식별하기 힘든 노이즈나 라벨링 데이터의 모호성으로 추정됨.

## 💻 How to Run
1. **환경 설정 (Requirements)**
   ```bash
   pip install torch torchvision pillow