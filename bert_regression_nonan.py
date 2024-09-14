import torch
from torch.nn import MSELoss
from torch import nn
from bert_for_longer_texts.belt_nlp.bert_regressor_with_pooling import BertRegressorWithPooling
from datasets import load_from_disk
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="domain")
args = parser.parse_args()

dataset = load_from_disk(f"data/{args.task}_reg_dataset")
X_train = [data["text"] for data in dataset["train"]]
y_train = [data["score"] for data in dataset["train"]]
X_test = [data["text"] for data in dataset["test"]]
y_test = [data["score"] for data in dataset["test"]]
y_cls = [data["class"] for data in dataset["test"]]
print("Data loaded")

pretrained_model_name_or_path = "emilyalsentzer/Bio_ClinicalBERT"

MODEL_PARAMS = {
    "pretrained_model_name_or_path": pretrained_model_name_or_path,
    "batch_size": 4,
    "learning_rate": 5e-5,
    "epochs": 5,
    "chunk_size": 510,
    "stride": 256,
    "minimal_chunk_length": 510,
    "maximal_text_length": 510 * 8,
    "pooling_strategy": "max",
    "device": "cuda:0",
    "many_gpus": True,
}
model = BertRegressorWithPooling(**MODEL_PARAMS)
# model = nn.DataParallel(model)
print("Model created")

# model.module.fit(X_train, y_train, epochs=5)  # Warning about tokeninizing too long text is expected
model.fit(X_train, y_train, epochs=5)
print("Model trained")

# scores = model.module.predict(X_test)
scores = model.predict(X_test)
results = torch.flatten(scores).cpu()

rmse = np.sqrt(mean_squared_error(y_test, results))
# r2 = r2_score(y_test, results.detach().numpy())
pred_test = pd.DataFrame({"y_cls": y_cls, "y_test": y_test, "y_pred": results})
r_squared = {}
domains = ["PCS", "MCS", "SPKD", "EKD", "BKD"]
for domain in domains:
    domain_df = pred_test[pred_test["y_cls"] == domain]
    rmse = np.sqrt(mean_squared_error(domain_df["y_test"], domain_df["y_pred"]))
    r2 = r2_score(domain_df["y_test"], domain_df["y_pred"])
    r_squared[domain] = r2

results = results.detach().numpy()
pairs = list(zip(y_test, results))

data = []
for pair in pairs:
    data.append({"y_test": str(pair[0]), "y_pred": str(pair[1])})

with open(f"data/{args.task}_reg_pairs.json", "w") as f:
    json.dump(data, f)

with open(f"data/{args.task}_reg_evals.txt", "w") as f:
    f.write("RMSE: " + str(rmse) + "\n")
    for domain, r2 in r_squared.items():
        f.write(f"{domain} R-squared: {r2}\n")
    # f.write("R-squared: " + str(r2) + "\n")