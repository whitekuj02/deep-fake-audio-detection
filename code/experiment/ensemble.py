import os
import pandas as pd
import numpy as np

# 경로 설정
csv_dir = "/root/asset/best"
output_csv = "/root/asset/ensemble/ensemble_result.csv"

# CSV 파일 목록 가져오기
csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')]

# 첫 번째 CSV 파일 읽기
ensemble_df = pd.read_csv(csv_files[0])

# 'fake'와 'real' 컬럼 초기화
ensemble_df['fake'] = 0.0
ensemble_df['real'] = 0.0

# 나머지 CSV 파일 읽고 누적합 계산
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    ensemble_df['fake'] += df['fake']
    ensemble_df['real'] += df['real']

# 평균 계산
ensemble_df['fake'] /= len(csv_files)
ensemble_df['real'] /= len(csv_files)

# 결과를 CSV 파일로 저장
ensemble_df.to_csv(output_csv, index=False)
print(f"Ensemble result saved to {output_csv}")
