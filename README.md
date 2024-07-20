# SW 중심 대학 디지털 대회 AI 부문 10등 코드
## team : MOTA

## 시작
sh model.sh

## 디렉토리
./asset : csv 저장 폴더 <br />
./code/aasist : aasist model <br />
./code/experiment : 전처리 후처리 .py <br />
./code/rawboost : augmentation <br />
./data : 데이터 저장 <br />

## 대회 데이터 받기
sh data.sh

## 데이터 디렉토리 변경
./code/aasist/config/AASIST.conf 에서 database_path 수정

## 환경
./environment.yaml 참고

### 주요 라이브러리 버전 
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118

## 사전 학습 모델
AST (MIT/ast-finetuned-audioset-10-10-0.4593) : masking <br />
https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593 <br />

DeepFilterNet : Denoising <br />
https://github.com/Rikorose/DeepFilterNet <br />

## 사용 tech
augmentation : Rawboost, mix train <br />
model : AASIST, DANN <br />
pre-processing : DeepFilterNet <br />
post-processing : AST <br />

## 데이터 셋 설명 :

./data/data16k : 모든 기존 데이터의 sample rate 32k를 16k로 resampling 한 것 <br />
./data/rawboost : augmentation 작업을 마친 데이터 셋 <br />
./data/rawboost/train_all : original + mixing + Rawboost (algorithm 0~8) 의 모든 데이터 ( 데이터 개수 : 105438, sampling rate : 16k ) <br />
./data/rawboost/train : train_all에서 80% sampling (데이터 개수 : 84350) <br />
./data/rawboost/val :  train_all에서 train 데이터를 제외한 나머지 20%의 데이터 (데이터 개수 :21088) <br />
./data/rawboost/unlabeled_data : 기존의 1264개의 unlabeled_data를 16k로 샘플링 한 것 <br />
./data/rawboost/test : 16k로 resampling한 test 데이터 <br />

## 모델 학습에 사용하는 데이터 셋:
./data/rawboost/train <br />
./data/rawboost/val <br />
./data/rawboost/unlabeled_data <br />

## 모델 추론에 사용되는 데이터 셋:
./data/rawboost/test <br />


## AASIST.conf 파일에서 필요한 것:
database_path : 데이터 경로 (기본 ./data/rawboost) <br />
asset_path : 결과물 저장 루트 (기본 /root/asset) <br />
batch_size : 배치 사이즈 (기본 32) <br />
num_epochs : 에폭 횟수 (기본 20) <br />
optim_config : 옵티마이저와 스케줄링 설정 <br />
{ <br />
        "optimizer": "adam", <br />
        "amsgrad": "False", <br />
        "base_lr": 0.0001, <br />
        "lr_min": 0.000005, <br />
        "betas": [0.9, 0.999], <br />
        "weight_decay": 0.0001, <br />
        "scheduler": "cosine" <br />
}, <br />
{ <br />
        "optimizer": "sgd", <br />
        "amsgrad": "False", <br />
        "momentum": 0.2, <br />
        "nesterov": "False", <br />
        "base_lr": 0.01, <br />
        "lr_min": 0.001, <br />
        "betas": [0.9, 0.999], <br />
        "weight_decay": 0.0001, <br />
        "scheduler": "sgdr", <br />
        "T0" : 3, <br />
        "Tmult" : 1 <br />
} <br />

모델 파라미터 : <br />
./parameter.pth  <br />
( 약 0.22 ~ 0.24 test score (6~10 epochs) :  ensemble 로 0.19 가능 ) <br />
