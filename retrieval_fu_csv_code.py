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

def call_GPTcompletions(qol, sys_prompt, usr_prompt1, query, usr_prompt2, seed=0):
    user_prompt = '''<KDQOL>
    {qol}
    
    {usr_prompt1}
    
    <KDQOL domains & items>
    The KDQOL survey is divided into specific domains, with each domain associated with particular questions (items).
    Below is a list of the KDQOL domains and the corresponding item numbers:
    - **Physical Component Summary (PCS)**: 1, 3a-j, 4a-d, 7, 8, 11a-d
    - **Mental Component Summary (MCS)**: 5a-c, 6, 9, 10
    - **Symptoms and Problems List (SPKD)**: 14a-l
    - **Effect of Kidney Disease (EKD)**: 15a-h
    - **Burden of Kidney Disease (BKD)**: 12a-d
    
    {query}
    
    {usr_prompt2}
    '''

    response = client.chat.completions.create(
		model="gpt-4o-2024-05-13", #"gpt-4-turbo-2024-04-09", #"gpt-4-1106-preview", #gpt-4-0125-preview	
		messages=[
            {
            "role": "system",
			"content": sys_prompt
            },
			{
			"role": "user",
			"content": user_prompt.format(qol=qol, usr_prompt1=usr_prompt1, query=query, usr_prompt2=usr_prompt2)
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
with open("./data/prompt/qol_fu_csv_code_prompt.yaml", "r", encoding="utf-8") as f:
    retrieve_prompt = yaml.load(f, Loader=yaml.FullLoader)

result = {}

query_to_retriever = [
    "- Has the score for the Physical Component Summary (PCS) improved?",
    "- Has the score for the Mental Component Summary (MCS) improved?",
    "- Has the score for the Symptoms and Problems List (SPKD) improved?",
    "- Has the score for the Effect of Kidney Disease (EKD) improved?",
    "- Has the score for the Burden of Kidney Disease (BKD) improved?"
] # english version

qol = data['eng_qol']
query = "\n".join(query_to_retriever)
response = call_GPTcompletions(qol, retrieve_prompt['sys_prompt'], retrieve_prompt['usr_prompt1'], query, retrieve_prompt['usr_prompt2'], seed=0)
result = {"qol":qol, "response":response}

with open('./data/gpt_results/qol_fu_retrieval_code_generation.json', 'w', encoding="utf-8") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)