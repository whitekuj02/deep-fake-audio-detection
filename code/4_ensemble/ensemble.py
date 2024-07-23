import os
import glob
import argparse
import pandas as pd

def main(args):

    # CSV 파일 목록 가져오기
    csv_files = glob.glob(os.path.join(args.input_path, "*.csv"))

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

    # masking
    nospeech_ids = []
    with open("../../asset/masking.csv") as f:
        f.readline()
        nospeech_ids = [line.strip() for line in f.readlines()]

    ensemble_df.loc[ensemble_df['id'].isin(nospeech_ids), ['fake', 'real']] = 0.0

    # 결과를 CSV 파일로 저장
    ensemble_df.to_csv(os.path.join(args.output_path, "ensemble_aasist_rawboost.csv"), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="/root/code/ensemble/input")
    parser.add_argument('--output_path', type=str, default="/root/code/ensemble/output")
    args = parser.parse_args()

    main(args)
