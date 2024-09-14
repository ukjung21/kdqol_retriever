from datasets import Dataset, DatasetDict
import yaml
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

contexts = []
# with open("domain_contexts_eng.jsonl", 'r', encoding='utf-8') as file:
with open("quest_contexts_eng.jsonl", 'r', encoding='utf-8') as file:
    for line in file:
        contexts.append(json.loads(line))

score_df = pd.read_csv("../data/gpt_results/scores_df.csv")
# score_df = pd.read_csv("../data/gpt_results/domain_scores_df.csv")
    
# query_to_retriever = [
#     "Tell me the score of how the patient described their health.",
#     "Tell me the score of whether the patient said they were unable to do their work as carefully as usual due to psychological problems in the past 4 weeks.",
#     "Tell me the score of whether the patient said they had problems with social activities with family, friends, and neighbors due to physical or psychological health problems in the past 4 weeks.",
#     "Tell me the score of whether the patient said they feel they get sick more easily than others.",
#     "Tell me the score of whether the patient said they feel frustrated when managing their kidney disease.",
#     "Tell me the score of whether the patient said they had trouble recognizing people or had a poor sense of time and place in the past 4 weeks.",
#     "Tell me the score of whether the patient said they were somewhat troubled by stress or worry due to kidney disease.",
#     "Tell me the score of whether the patient said they were able to sleep as much as needed.",
#     "Tell me the score of whether the patient thinks it will be difficult to continue working at their job due to their health.",
#     "Tell me the score of whether the patient thinks the doctors and nurses at the dialysis center help them cope well with their kidney disease.",
# ]

query_to_retriever = [
    "How high did the patient rate their Physical Component Summary score?",
    "How high did the patient rate their Mental Component Summary score?",
    "How high did the patient rate their Symptom List score?",
    "How high did the patient rate their Effect of Kidney Disease score?",
    "How high did the patient rate their Burden of Kidney Disease score?"
] # english version
domains = ["PCS", "MCS", "SPKD", "EKD", "BKD"]

contexts = [line["context"] for line in contexts]
reg_triples = []
for i, question in enumerate(query_to_retriever):
    answers = list(score_df.iloc[:, i])
    # Extend the reg_triples list with new entries for each context-answer pair
    for context, answer in zip(contexts, answers):
        # if np.isnan(answer):
        #     answer = (int(0), -1)
        # else:
        #     answer = (int(1), answer)
        if np.isnan(answer):
            continue
        text = '\n\n'.join([context, question])
        reg_triples.append({"text": text, "score": answer, "class":domains[i]})

# Create a Dataset from the list of dictionaries
reg_dataset = Dataset.from_dict({"data": reg_triples})

# Split the dataset into training, validation, and test sets
reg_df = pd.DataFrame(reg_triples, columns=["text", "score", "class"])
train_df, test_df = train_test_split(reg_df, test_size=0.1, stratify=reg_df["class"])

train_dataset = Dataset.from_dict(train_df)
test_dataset = Dataset.from_dict(test_df)

# Create a DatasetDict
dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# Save the dataset_dict to disk
dataset_dict.save_to_disk('./data/domain_reg_dataset')