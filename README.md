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
./parameter.pth : ensemble 한 것 중 하나의 모델 파라미터 <br />

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
