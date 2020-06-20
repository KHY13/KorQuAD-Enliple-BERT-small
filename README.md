## KorQuAD
Enliple에서 공개한 BERT-Small 모델을 KorQuAD에 적용한 코드입니다.
Dev 데이터 기준 Single모델(EM 82.89%, F1=91.59%), Ensemble모델(EM=83.65%, 92.18%)의 결과를 얻었습니다.
Ensemble의 경우 간단하게 seed와 배치 사이즈만 변경하여 학습한 7개의 모델을 Max voting을 통해 예측을 하였습니다.

## Colab에서 모델 학습 및 평가하기

### 1. Drive 마운트
```
from google.colab import auth
auth.authenticate_user()
from google.colab import drive
drive.mount('/content/gdrive')
```

### 2. Mecab 설치
```
!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
!cd "Mecab-ko-for-Google-Colab" && bash "install_mecab-ko_on_colab190912.sh"
```

### 3. 데이터 전처리
```
!cd "/content/gdrive/" && git clone https://github.com/KHY13/KorQuAD-Enliple-BERT-small.git
!cd "/content/gdrive/KorQuAD-Enliple-BERT-small" && python mecab_preprocess.py --split_train_data 
```

### 4. 모델 학습
```
!cd "/content/gdrive/KorQuAD-Enliple-BERT-small" && python train.py --split_train_data --use_cuda
```

### 5. 모델 평가
```
!cd "/content/gdrive/KorQuAD-Enliple-BERT-small" && python train.py --split_train_data --use_cuda --eval
```