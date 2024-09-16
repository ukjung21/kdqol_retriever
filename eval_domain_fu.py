import json
import pandas as pd
import math
import numpy as np
import re
from sklearn.metrics import r2_score, mean_squared_error

with open('./data/gpt_results/qol_fu_retrieval_async_all.json', 'r', encoding="utf-8") as f:
    response = json.load(f)
    
df = pd.read_csv("./data/new_0820_all_cause_death.csv")
qol_cols = [col for col in list(df.columns) if col.startswith('qol')]
labels = df[qol_cols]

preds = []
for id, value in response.items():
    answer = value['response'].split('[')[-1].split(']')[0]
    answer = answer.replace('Not Applicable', 'nan').replace('N/A', 'nan').replace('None', 'nan')
    pred_list = eval(answer, {"nan": math.nan})
    
    preds.append(pred_list)
    
preds = pd.DataFrame(preds, columns=qol_cols)

# 두 DataFrame에서 NaN 값을 포함하는 행을 제외합니다.
valid_indices = labels.notna().all(axis=1) & preds.notna().all(axis=1)
df_true_filtered = labels[valid_indices]
df_pred_filtered = preds[valid_indices]

# 각 열에 대해 R^2를 계산합니다.
r2_scores = {col: r2_score(df_true_filtered[col], df_pred_filtered[col]) for col in df_true_filtered}

# 전체 DataFrame에 대해 RMSE를 계산합니다.
rmse = mean_squared_error(df_true_filtered.values, df_pred_filtered.values, squared=False)

# 평가 지표를 출력합니다.
print("R^2 scores by domain:")
for col, r2 in r2_scores.items():
    print(f"{col}: {r2:.3f}")
print(f"Overall RMSE: {rmse:.3f}")