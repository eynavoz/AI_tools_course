from openai import AzureOpenAI
import os
import pandas as pd
import gt
from dotenv import load_dotenv
load_dotenv()


MODEL_4o = 'gpt-4o-mini'
OPEN_API_VERSION_4o = '2024-08-01-preview'


AZURE_OPENAI_API_KEY = os.getenv('SUBSCRIPTION_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('SUBSCRIPTION_OPENAI_ENDPOINT')
OPENAI_API_VERSION = '2023-12-01-preview'
MODEL = 'gpt-35-16k'
TEMP = .3
MAX_TOKENS = 7000
LOG_PROBS = False

client = AzureOpenAI(
    api_key= AZURE_OPENAI_API_KEY,
    api_version= OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

sys_prompt = {"role": "system", "content": "You are a dietitian and a nutrition expert."}
user_prompt = {"role": "user", "content": """A restaurant wants to provide nutritional information \
for each dish (menu item) it offers. What information should it provide?
Limit your response to no more than 10 requirements."""}
msg = [sys_prompt, user_prompt]
responses = client.chat.completions.create(messages=msg, model=MODEL, temperature=TEMP, max_tokens=MAX_TOKENS, logprobs= LOG_PROBS, n=1)
print(responses.choices[0].message.content)


