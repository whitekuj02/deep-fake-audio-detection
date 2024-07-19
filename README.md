# SW 중심 대학 디지털 대회 AI 구문 10등 코드
## team : MOTA

## 시작
sh model.sh

## 디렉토리
./asset : csv 저장 폴더
./code/aasist : aasist model
./code/experiment : 전처리 후처리 .py
./code/rawboost : augmentation
./data : 데이터 저장
./parameter.pth : ensemble 한 것 중 하나의 모델 파라미터

## 대회 데이터 받기
sh data.sh

## 데이터 디렉토리 변경
./code/aasist/config/AASIST.conf 에서 database_path 수정

## 환경
./environment.yaml 참고

### 주요 라이브러리 버전 
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118



