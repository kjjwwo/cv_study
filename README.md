# cv_study
**computer vision study repository**

컴퓨터비전 학습을 위한 코드를 모아둔 repositoy입니다. 

여러가지 프로젝트가 있으며, colab에서 실행 가능한 코드를 제공합니다. 
## C1

### 프로젝트 설명
 - 본 프로젝트는 reCAPTCHA 데이터셋을 활용하여 딥러닝 모델을 사용하지 않고, 수동으로 이미지의 특징을 설계(Hand-crafted Features)하여 이미지를 분류하고 검색하는 고전적 컴퓨터 비전 파이프라인을 구현한 프로젝트입니다.

### 모델 특징
 - **Image Preprocessing:** OpenCV를 활용한 denosing, edge detection, sharpening, histogram equalization 진행
 - **Advanced Feature Extraction:**
    - **LBP:** 이미지의 국소적 질감 패턴 분석
    - **HoG:** histogram of oriented gradients를 통한 객체의 형태적 특징 포착
    - **GLCM:** Contrast, Correlation, Energy, Honogeneity 등 통계적 질감 특징 추출
  - **Dimensionality Reduction:** 3,855 차원의 거대한 특징 벡터를 PCA(주성분 분석)를 통해 핵심 정보인 256 차원으로 압축
  - **Classification & Retrieval:** Scikit-learn의 KNN을 활용하여 클래스 분류 및 유사 이미지 Top-10 출력 

### 기술 스택
 - **Language:** Python
 - **Framework:** OpenCV
 - **Libraries:** Scikit-image (LBP, HoG, GLCM), Scikit-learn (PCA, StandardScaler, KNN)

### 결과 (Result)
  - 딥러닝 대비 가벼운 계산량으로도 분류 가능 확인
  - PCA 적용을 통해 차원의 저주 해결 확인
  - 하지만 딥러닝 대비 비교적 낮은 분류 성능 확인
  - 추가적으로 여러 조합의 특징을 조합해 보아야 할 필요 있음
**********

## C2 - reCAPTCHA Image Classification & Retrieval using ResNet

### 프로젝트 설명
 - 본 프로젝트는 reCAPTCHA 데이터셋을 활용하여 이미지 분류(Classification) 모델을 학습시키고, 추출된 특징 벡터(Feature Vector)를 기반으로 유사 이미지를 검색(Retrieval)하는 시스템을 구현하였습니다. 

### 모델 특징
 - **Transfer Learning:** Pre-trained **ResNet18 model**을 이용하여 Fine-tuning 진행
 - **K-Fold Cross Validation:** 모델의 일반화 성능을 향상시키기 위해 K-Fold 교차 검증 진행
 - **Early Stopping:** Validation Loss 기반으로 Loss 값이 지속적으로 감소하지 않을 시 학습 조기 종료, 과적합 방지
  - **Classification & Retrieval:** Scikit-learn의 KNN을 활용하여 클래스 분류 및 유사 이미지 Top-10 출력

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

### 실험 기록
  - 성능 최적화를 위해 수항한 11차례의 실험 기록입니다. 검증 정확도와 과적합 방지를 위해 다양한 구조적 변화와 하이퍼파라미터 조정을 시도했습니다.
| 실험 ID | 주요 변경 사항 (Modification) | 설정 (Settings) | 목적 및 결과 |
| :--- | :--- | :--- | :--- |
| **Exp 1-3** | K-Fold 도입 및 LR 조정 | 2~5 Fold, LR 0.001~0.0001 | 학습 안정성 확보 및 교차 검증 도입 |
| **Exp 4-6** | FC 레이어 확장 & Patience | Dense(256) 추가, Patience=10 | 모델 용량 증대 및 Early Stopping 적용 |
| **Exp 7-8** | Data Augmentation 적용 | Random Crop/Flip, LR 조정 | 일반화 성능 향상 시도 |
| **Exp 9-11** | **Final Architecture** | **Dropout(0.4) + 2nd Dense** | **과적합 해결 및 최종 성능 극대화** |

### 주요 인사이트 (Insights)
1. **Layer Depth**: 단순 ResNet18 뒤에 Dense 레이어를 2개 추가하고 Dropout을 적용했을 때, 복잡한 reCAPTCHA 이미지의 특징을 가장 잘 분류했습니다.
2. **Learning Rate**: 0.0006 ~ 0.0008 사이의 미세한 학습률 조정이 수렴 성능에 큰 영향을 미쳤습니다.
3. **Regularization**: 데이터 증강과 Dropout(0.4)의 조합이 Validation Loss의 진동을 줄이는 데 핵심적인 역할을 했습니다.

### 결과 (Result)
 - **Best Validation Accuracy:** 
 - **Classification Report:** 
   - **Precision:** 
   - **Recall:** 
   - **F1-score:**

*********
