import json
import os
from openai import OpenAI

import pandas as pd
import yaml
import random

with open("../config.yaml", 'r') as f:
    config = yaml.safe_load(f)
OPENAI_API_KEY = config['key']

client = OpenAI(api_key=OPENAI_API_KEY)
# <KDQOL domains & items>
# - **Physical Component Summary (PCS)**: 1, 3a-j, 4a-d, 7, 8, 11a-d
# - **Mental Component Summary (MCS)**: 5a-c, 6, 9, 10
# - **Symptoms and Problems List (PKD)**: 14a-l
# - **Effect of Kidney Disease (EKD)**: 15a-h
# - **Burden of Kidney Disease (BKD)**: 12a-d
def call_GPTcompletions(qol, prompts, query, seed=0):
    user_prompt = f'''{prompts['usr_prompt1']}
    <KDQOL>
    {qol}
    
    {prompts['usr_prompt2']}
    
    Please write Python code to answer the physician's queries below:
    <Physician's queries>
    {query}
    
    {prompts['usr_prompt3']}'''

    response = client.chat.completions.create(
		model="gpt-4o-2024-05-13", #"gpt-4-turbo-2024-04-09", #"gpt-4-1106-preview", #gpt-4-0125-preview	
		messages=[
            {
            "role": "system",
			"content": prompts['sys_prompt']
            },
			{
			"role": "user",
			"content": user_prompt
			},
		],
		max_tokens=4096,
		temperature=0.1,
		top_p=0.1,
		seed=seed
    )
    return response.choices[0].message.content

with open("./data/kdqol_questionnaire.yaml", "r", encoding="utf-8") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
with open("./data/prompt/qol_retrieval_csv_code_prompt.yaml", "r", encoding="utf-8") as f:
    retrieve_prompt = yaml.load(f, Loader=yaml.FullLoader)

result = {}

query_to_retriever = [
    "- How high did the patient rate their Physical Component Summary score?",
    "- How high did the patient rate their Mental Component Summary score?",
    "- How high did the patient rate their Symptom List score?",
    "- How high did the patient rate their Effect of Kidney Disease score?",
    "- How high did the patient rate their Burden of Kidney Disease score?"
] # english version

qol = data['eng_qol']
queries = query_to_retriever
query = "\n".join(queries)
response = call_GPTcompletions(qol, retrieve_prompt, query, seed=0)
result = {"qol":qol, "response":response}

with open('./data/gpt_results/qol_domain_retrieval_code_generation_noprior_1.json', 'w', encoding="utf-8") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)