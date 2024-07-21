import os
import csv

import pandas as pd

df = pd.read_csv('/root/asset/infer_masking.csv')

df_list = []

for i, r in df.iterrows():
    df_list.append(r['id'].strip())


df_list.pop(0)
    

def mask_zero(submission_path, diff_df_1, masked_file_path, output_path):

    # 제출 파일과 마스킹 파일 열기
    with open(submission_path, "r") as f, \
         open(masked_file_path, "r") as sf, \
         open(output_path, "w", newline='') as wf:

        # csv 작성자 설정
        writer = csv.writer(wf)

        # 헤더 작성
        header = next(f)
        writer.writerow(header.strip().split(","))

        # 마스킹 파일 데이터를 딕셔너리로 로드
        sf_reader = csv.reader(sf)
        next(sf_reader)  # 헤더 건너뜀
        sf_data = {row[0]: (row[1], row[2]) for row in sf_reader}

        # 제출 파일 처리
        reader = csv.reader(f)
        for row in reader:
            _id, _fake, _real = row
            if _fake == '0.0' and _real == '0.0':
                if _id in diff_df_1 and _id in sf_data:
                    _fake, _real = sf_data[_id]  # sf의 id 위치의 fake와 real 값을 사용

            writer.writerow([_id, _fake, _real])

    print("Masking (post-processing) done!")

# 파일 경로 설정
submission_path = '/root/asset/ensemble/ensemble_best_0.193.csv'
masked_file_path = '/root/asset/best/submission0719_DNFN_AASIST_MASKED.csv'
output_path = '/root/asset/submission.csv'

# 함수 호출
mask_zero(submission_path, df_list, masked_file_path, output_path)
