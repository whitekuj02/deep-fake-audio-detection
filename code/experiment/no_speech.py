non_speech_list = []

with open("/root/asset/non_speech.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        file_path = line.split(' ')[0]
        file_name = file_path.split('/')[-1][:-4]
        non_speech_list.append(file_name)


import pandas as pd

submission_df = pd.read_csv("/root/code/aasist/exp_result_rawboost/LA_AASIST_ep100_bs32/test/submission_ep0.csv")
submission_df.loc[submission_df['id'].isin(non_speech_list), ['fake', 'real']] = 0

submission_df.to_csv("./masked_result.csv", index=False)