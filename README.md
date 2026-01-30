# cv_study
**computer vision study repository**

컴퓨터비전 학습을 위한 코드를 모아둔 repo입니다. 

여러가지 프로젝트가 있으며, colab에서 실행 가능한 코드를 제공합니다. 
## C1

**********

## C2 - reCAPTCHA Image Classification & Retrieval using ResNet

### 프로젝트 설명
 - 본 프로젝트는 reCAPTCHA 데이터셋을 활용하여 이미지 분류(Classification) 모델을 학습시키고, 추출된 특징 벡터(Feature Vector)를 기반으로 유사 이미지를 검색(Retrieval)하는 시스템을 구현하였습니다. 
### 모델 특징
 - **Transfer Learning:** Pre-trained **ResNet18 model**을 이용하여 Fine-tuning 진행
 - **K-Fold Cross Validation:** 모델의 일반화 성능을 향상시키기 위해 K-Fold 교차 검증 진행
 - **Early Stopping:** Validation Loss 기반으로 Loss 값이 지속적으로 감소하지 않을 시 학습 조기 종료, 과적합 방지
 - **Image Retrieval: KNN(K-Nearest Neighbors)** 알고리즘을 사용한 유사 이미지 검색 및 Top-K 이미지 출력

### 기술 스택
- **Language**: Python
- **Framework**: PyTorch, Torchvision
- **Libraries**: Scikit-learn (K-Fold, KNN), Matplotlib, NumPy, Pandas

### 데이터셋 구조
'recaptcha-dataset'에서 추출한 10개의 클래스를 사용
- Classes: 'Bicycle', 'Bridge', 'Bus', 'Car', 'Chimney', 'Crosswalk', 'Hydrant', 'Motorcycle', 'Palm', 'Traffic Light'
- Input Size: 224x224 (RGB)

### 모델 학습 전략
 - **Optimizer:** SGD (lr=0.00069, momentum=0.9)
 - **Loss Function:** CrossEntropyLoss
 - **Data Augmentation**: RandomResizedCrop, RandomHorizontalFlip 등을 통한 데이터 다양성 확보
 - **Feature Extraction**: ResNet18의 마지막 FC 레이어 전단계에서 512차원 특징 벡터 추출

### 결과 (Result)
 - **Best Validation Accuracy:** 
 - **Classification Report:** 
   - **Precision:** 
   - **Recall:** 
   - **F1-score:** 

### 하이퍼파라미터 조정 과정
 - lr: 
 - momentum:
 - 

*********
