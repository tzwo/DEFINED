import json
import os
import re
import time
import random
from openai import OpenAI
from prompt_creativity import CREATIVITY_PROMPT_DEBATE

# Configuration
API_KEY = ""
BASE_URL = ""
MODEL_NAME = ""
INPUT_FILE = ""
OUTPUT_FILE = f"experiment_{MODEL_NAME}.json"

def get_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Case 1: Dict with "all" key
    if isinstance(data, dict) and 'all' in data:
        return data['all']
    
    # Case 2: List directly
    if isinstance(data, list):
        return data
        
    # Case 3: Dict of lists (experiment.json format)
    if isinstance(data, dict):
        all_items = []
        for key, value in data.items():
            if isinstance(value, list):
                all_items.extend(value)
        return all_items
        
    return []

def fill_prompt(template, item):
    prompt = template
    prompt = prompt.replace("{辩题：}", f"辩题：{item.get('辩题', '')}")
    prompt = prompt.replace("{持方：}", f"持方：{item.get('持方', '')}")
    
    debate_scripts = f"正方一辩稿：{item.get('正方一辩稿', '')}\n反方一辩稿：{item.get('反方一辩稿', '')}"
    prompt = prompt.replace("{正方一辩稿：反方一辩稿：}", debate_scripts)
    
    prompt = prompt.replace("{对方发言：}", f"对方发言：{item.get('对方发言', '')}")
    prompt = prompt.replace("{本轮发言：}", f"本轮发言：{item.get('本轮发言', '')}")
    
    return prompt

def extract_json(content):
    # Try to find JSON block enclosed in ```json ... ```
    match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Try to find JSON block enclosed in ``` ... ```
        match = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Assume the whole content might be JSON if no code blocks
            json_str = content
            
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {content}")
        return None

def chat_completion_with_retry(client, model, messages, temperature=0.7, max_retries=10, initial_delay=5):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                # Exponential backoff
                delay = initial_delay * (2 ** attempt) + random.uniform(0, 3)
                # Cap delay at 120s
                delay = min(delay, 120)
                print(f"API Error (Attempt {attempt+1}/{max_retries}): {error_msg}. Retrying in {delay:.2f}s...")
                time.sleep(delay)
            else:
                raise e

def main():
    # Ensure we are in the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"Working directory: {os.getcwd()}")
    
    client = get_client()
    data = load_data(INPUT_FILE)
    
    # --- Resume Logic ---
    results = []
    processed_prompts = set()
    
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    results = json.loads(content)
                    if isinstance(results, list):
                        print(f"Resuming: Loaded {len(results)} existing results from {OUTPUT_FILE}")
                        for res in results:
                            processed_prompts.add(res.get("prompt"))
                    else:
                        print("Warning: Existing output file is not a list. Starting fresh.")
                        results = []
        except Exception as e:
            print(f"Error loading existing results: {e}. Starting fresh.")
            results = []
    
    print(f"Found {len(data)} items total. {len(processed_prompts)} already processed.")
    
    for i, item in enumerate(data):
        filled_prompt = fill_prompt(CREATIVITY_PROMPT_DEBATE, item)
        
        # Check if already processed
        if filled_prompt in processed_prompts:
            continue
            
        print(f"Processing item {i+1}/{len(data)}...")
        
        # Retry logic for the item processing (API + Parsing)
        max_item_retries = 3
        success = False
        
        for attempt in range(max_item_retries):
            try:
                response = chat_completion_with_retry(
                    client,
                    model=MODEL_NAME,
                    messages=[
                        {"role": "user", "content": filled_prompt}
                    ],
                    temperature=0.7 
                )
                
                content = response.choices[0].message.content
                generated_values = extract_json(content)
                
                if generated_values is None:
                    print(f"  Attempt {attempt+1}/{max_item_retries}: JSON extraction failed. Retrying...")
                    continue
                
                result_item = {
                    "prompt": filled_prompt,
                    "generated_values": generated_values,
                    "ground_truth_values": item.get("values", item.get("value"))
                }
                results.append(result_item)
                processed_prompts.add(filled_prompt)
                success = True
                
                # Incremental Save
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                # Small delay to be polite
                time.sleep(2)
                break
                
            except Exception as e:
                print(f"  Attempt {attempt+1}/{max_item_retries} failed: {e}")
                time.sleep(2)
        
        if not success:
            print(f"Error: Failed to process item {i+1} after {max_item_retries} attempts.")
            
    print(f"All done. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
