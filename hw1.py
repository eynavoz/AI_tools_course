from openai import AzureOpenAI
import os
import pandas as pd
import re
import time
#import gt
from dotenv import load_dotenv
load_dotenv()

# OpenAI connection
AZURE_OPENAI_API_KEY = os.getenv('CLASS_AZURE_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('SUBSCRIPTION_OPENAI_ENDPOINT')

MODEL_4o = 'gpt-4o-mini'
OPENAI_API_VERSION_4o = '2024-08-01-preview'

MODEL_35 = 'gpt-35-16k'
OPENAI_API_VERSION_35 = '2023-12-01-preview'


TEMP_1 = 0
TEMP_2 = .5
TEMP_3 = .9
MAX_TOKENS = 7000
LOG_PROBS = False

# ---- Function: Generate response ------ 
def generate_response(version, system_prompt, prompt, model, temp):
    """
    Get a response from the specified OpenAI model using the given prompt.
    
    Args:
        version (str): OPEN_API_VERSION
        system_prompt (str): The system prompt to provide the model
        prompt (str): The prompt to provide to the open AI model
        model (str): The name of ID of the openAI engine to use
        temp (str): the temperature parameter
    """

    client = AzureOpenAI(
        api_key= AZURE_OPENAI_API_KEY,
        api_version= version,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    sys_prompt = {"role": "system", "content": system_prompt}
    user_prompt = {"role": "user", "content": prompt}
    msg = [sys_prompt, user_prompt]
    responses = client.chat.completions.create(messages=msg, model=model, temperature=temp, max_tokens=MAX_TOKENS, logprobs= LOG_PROBS, n=1)
    return responses.choices[0].message.content

# ---- Function: Classify LLM response ------ 
def classify_llm_response(response_text):
    if not isinstance(response_text, str):
        return 'unknown'
    
    match = re.search(r"Issue\s*Count\s*:\s*(\d+)", response_text, re.IGNORECASE)
    
    if match:
        count = int(match.group(1))
        if count == 1:
            return 'no-split'
        elif count == 2:
            return 'split'
    
    return 'unknown'

# ---- Function: Parse input file ------ 
def parse_ticket_file(file_path):
    tickets = []

    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Parse each line
    for line in lines:
        line = line.strip()
        if line:
            match = re.match(r"(\d+):\s*(.+)", line)
            if match:
                number = int(match.group(1))
                ticket = match.group(2)
                tickets.append({'number': number, 'ticket': ticket})
    
    # Create DataFrame
    df = pd.DataFrame(tickets)
    return df


# ---- Prompt templates ------ 
sys_prompt_part1 = "You are a helpful assistant that classifies support tickets."

user_prompt_part1 = """
    You are given a ticket of a software bug that a customer experiences. 
    Your task is to determine if a ticket describes a single issue or is really 2 different issues.

    Use the following categories for issues classification:
    - Interface
    - Lacking Feature
    - Logic Defect
    - Data
    - Security and Access Control
    - Configuration
    - Stability
    - performance

    Eamples
    Here is the tickect to classify:
    {ticket}
    End of ticket

    Be concise and structured in your output. Return your result ONLY in this format:
    {{
    Issue Count: [1 or 2]
    Issue 1: category name
    Detail 1: short summary to support your decision
    Issue 2: categor name (Only include this if there are 2 issues.)
    Detail 2: short summary to support your decision
    }}
    Do not add any additional text besides this format."""

# --- Main program ----
def main():
    """--- Part 1 ----
    Read the file, tkts_1.txt
    For each ticket in the file, your program needs to read the ticket and prompt the LLM 
    to determine if the ticket is a single issue or is really 2 different issues."""
    file_path = "./tkts_1.txt"
    print("reading ticket file")
    df = parse_ticket_file(file_path)

    # --- Test configurations ---
    tests = [
        {"model": MODEL_35, "version": OPENAI_API_VERSION_35, "temperature": 0},
        {"model": MODEL_4o, "version": OPENAI_API_VERSION_4o,"temperature": 0},
        {"model": MODEL_4o, "version": OPENAI_API_VERSION_4o,"temperature": 0.9}
    ]

    # --- Output Collector ---
    output_lines = []

    # --- Processing ----
    for idx, row in df.iterrows():
        ticket_number = row['number']
        ticket_text = row['ticket']
        output_lines.append(f"{ticket_number}:")

        for test in tests:
            print("running test")
            model = test['model']
            temp = test['temperature']
            version = test['version']
            try:
                response = generate_response(
                    version,  # Or adapt if version changes
                    sys_prompt_part1,
                    user_prompt_part1.format(ticket=ticket_text),
                    model,
                    temp
                )
                llm_output = response.strip()
            except Exception as e:
                llm_output = f"ERROR: {e}"

            classification = classify_llm_response(llm_output)
            output_lines.append(f"{model}, {temp}: {classification}")
            output_lines.append(f"LLM response: {llm_output}")
      
        output_lines.append("")  # Empty line between tickets
    
    print("writing results to ouput file")
    # ---- Save to File ---
    with open("split.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print("✅ Results written to split.txt")


# ---- Run if executed from terminal ---
if __name__ == "__main__":
    main()