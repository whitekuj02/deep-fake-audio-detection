
import pandas as pd

# 데이터프레임 읽기
df_1 = pd.read_csv('/root/asset/ensemble/ensemble_best_0.193.csv')
df_2 = pd.read_csv('root/asset/best/submission0719_DNFN_AASIST_MASKED.csv')

# 결합 조건에 따라 값 결합
def weighted_average(val1, val2, weight1, weight2):
    return (val1 * weight1 + val2 * weight2) / (weight1 + weight2)

for index, row in df_1.iterrows():
    if 0.4 <= row['fake'] <= 0.6 or 0.4 <= row['real'] <= 0.6:
        df_1.at[index, 'fake'] = weighted_average(row['fake'], df_2.at[index, 'fake'], 4, 1)
        df_1.at[index, 'real'] = weighted_average(row['real'], df_2.at[index, 'real'], 4, 1)

# 결합된 데이터프레임 저장
df_1.to_csv('/root/asset/submission.csv', index=False)

print("Combination (weighted averaging) done!")
