import json
from openai import AsyncOpenAI
import openai
import asyncio
import pandas as pd
import yaml
import time
import re

with open("../config.yaml", 'r') as f:
    config = yaml.safe_load(f)
OPENAI_API_KEY = config['key']

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # <KDQOL domains & items>
    # The KDQOL survey is divided into specific domains, with each domain associated with particular questions (items).
    # Below is a list of the KDQOL domains and the corresponding item numbers:
    # - **Physical Component Summary (PCS)**: 1, 3a-j, 4a-d, 7, 8, 11a-d
    # - **Mental Component Summary (MCS)**: 5a-c, 6, 9, 10
    # - **Symptoms and Problems of Kidney Disease List (SPKD)**: 14a-l
    # - **Effect of Kidney Disease (EKD)**: 15a-h
    # - **Burden of Kidney Disease (BKD)**: 12a-d

async def call_gpt_completions_async(qol, prompts, query, seed=0, max_attempts=5, delay=1):
    user_prompt = f'''{prompts['usr_prompt1']}
    <KDQOL>
    {qol}
    
    {prompts['usr_prompt2']}
    
    Please answer the physician's questions below:
    <Physician's queries>
    {query}
    
    {prompts['usr_prompt3']}'''

    for i in range(max_attempts):
        await asyncio.sleep(delay)
        response = await client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=[{"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}]
        )
        gpt_response = response.choices[0].message.content
        matches = re.findall(pattern, str(gpt_response))
        if matches:
            return gpt_response
        
    return "Not matched"


with open('./data/qol_eng.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
with open("./data/prompt/qol_retrieve_avg_prompt.yaml", "r", encoding="utf-8") as f:
    retrieve_prompt = yaml.load(f, Loader=yaml.FullLoader)

result = {}

queries = [
    "- How high did the patient rate their Physical Component Summary score?",
    "- How high did the patient rate their Mental Component Summary score?",
    "- How high did the patient rate their Symptom List score?",
    "- How high did the patient rate their Effect of Kidney Disease score?",
    "- How high did the patient rate their Burden of Kidney Disease score?"
] # english version
query = "\n".join(queries)

async def main():
    result = {}

    data_items = list(data.items())
    pattern = r'\*\*Answer:\*\* \[(.+)\]'
    start = time.time()
    tasks = [call_gpt_completions_async(qol, retrieve_prompt, query, seed=0) for _, qol in data_items]
    responses = await asyncio.gather(*tasks)
    for j, (id, qol) in enumerate(data_items):
        result[id] = {"qol": qol, "response": responses[j]}
    
    elapsed = time.time() - start
    print(f"Completed! Elapsed time: {elapsed:.2f} seconds")

    with open('./data/gpt_results/qol_domain_retrieval_async_noprior_1.json', 'w', encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
	asyncio.run(main())