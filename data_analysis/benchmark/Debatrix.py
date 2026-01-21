import os
import json
import re
import time
import random
from openai import OpenAI
from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment

# Configuration
API_KEY = ""
BASE_URL = ""
MODEL = ""
INPUT_FILE = ""
OUTPUT_FILE = ""

# Templates from score_debatrix.py
PANEL_SCHEMA_PROMPT = """
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}, the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
```
{{ schema }}
```
""".strip()

SPEECH_PROMPT_TEMPLATE = """
The following paragraph is a judgment on a specific speech in a debate.

The debate motion is: {{ info.motion }}

Pro side debaters are: {{ info.pro_side|map(attribute='name')|join(', ') }}

Con side debaters are: {{ info.con_side|map(attribute='name')|join(', ') }}

The speaker of the speech is: {{ debater_name }}

--------------
{{ judgment }}
--------------

Please, according to the judgment, generate a score for {{ debater_name }} from 0 to 100.
""".strip()

UPDATE_SYSTEM_TEMPLATE = """
Now, {{ new_speech.debater_name }} gives Speech {{ prev_content|length + 1 }}. You are given the debate info slide. {% if prev_content|length > 0 %}Also, you are given all previous speeches made in the debate.{% endif %}

{{ analyze_speech }}

{% if prev_content|length > 0 %}{{ use_previous }}{% endif %}

Please think critically before responding.
""".strip()

UPDATE_HUMAN_TEMPLATE = """
# Info Slide

{{ info_slide }}

{% for source, content in prev_content %}
# {{ source }}

{{ content }}
{% endfor %}

# New Speech by {{ new_speech.debater_name }}

{{ new_speech.content }}
""".strip()

GENERAL_ANALYZE_SPEECH = """
Please provide references of how the debater argues in the new speech. This includes what arguments are made, whether they are on topic and sound, whether their backing premises and logic are clear and reasonable, whether they are supported by credible evidence, and sections that affects the clarity level and the conduct level. Lack of evidence, evidence with unreliable sources, source spam without relevant analysis, extremely illegible paragraphs and extremely unsportsmanlike or outright toxic behaviors should be reported; however, you need to think critically of whether they are really weakening the arguments.
""".strip()

GENERAL_USE_PREVIOUS = """
To clarify arguments already made in the debate and quotes across speech, you should refer to previous speeches; you need to distinguish quotes from normal sections. You also need to report whether the counterarguments provide effective refutation against the opponent, or whether it fails to respond to some arguments from the opponent. In addition, analyze the clashes between both side's argument sets; for each clash, think critically and logically to compare the relative strength between sides.
""".strip()

def get_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)

def chat_completion_with_retry(client, model, messages, temperature=0.7, max_tokens=1024, max_retries=10, initial_delay=5):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False  # Changed to False for simplicity with OpenAI client, can be True if needed but requires different handling
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
                print(f"API Call Failed after {max_retries} attempts: {error_msg}")
                return None

def _call_api(client, model, messages, max_tokens=1024, temperature=0.7):
    response = chat_completion_with_retry(client, model, messages, temperature, max_tokens)
    if response and response.choices:
        return response.choices[0].message.content
    return None

def _format(template, **kwargs):
    return SandboxedEnvironment(undefined=StrictUndefined).from_string(template).render(**kwargs)

def _score_schema():
    return json.dumps({"title": "Score for Specific Debater", "type": "object", "properties": {"score": {"title": "Score for Specific Debater", "description": "a score for the debater", "type": "integer"}}, "required": ["score"]})

def _parse_score(content: str):
    if not content:
        return None
    obj = None
    try:
        obj = json.loads(content)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", content)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if isinstance(obj, dict):
        v = obj.get("score")
        try:
            return int(v)
        except Exception:
            return None
    return None

def process_data(use_cache=True):
    client = get_client()
    
    # 1. Read Input Data
    abs_input_path = os.path.abspath(INPUT_FILE)
    print(f"Reading input data from {abs_input_path}")
    
    if not os.path.exists(abs_input_path):
        print(f"Input file not found: {abs_input_path}")
        return

    with open(abs_input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)
        
    # 2. Read Existing Output Data (Resume Capability)
    results = []
    results_map = {} # Map (debate_name, turn_index) -> index in results list
    processed_history_cache = {} # Map context_key -> list of processed speech history steps

    abs_output_path = os.path.abspath(OUTPUT_FILE)
    if os.path.exists(abs_output_path):
        print(f"Reading existing results from {abs_output_path}")
        try:
            with open(abs_output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                for idx, entry in enumerate(results):
                    key = (entry.get("debate_name"), entry.get("turn_index"))
                    results_map[key] = idx
                    
                    if use_cache:
                        # Cache successful histories
                        history = entry.get("speech_analysis_history", [])
                        if history:
                            # Find the index of "本轮发言"
                            current_idx = -1
                            for h_i, h_item in enumerate(history):
                                if h_item.get("role") == "本轮发言":
                                    current_idx = h_i
                                    break
                            
                            if current_idx > 0:
                                # Construct cache key from items BEFORE current speech
                                cache_key_parts = []
                                valid_cache = True
                                for h_item in history[:current_idx]:
                                    # Check if score is valid
                                    if h_item.get("score") is None or h_item.get("score") == 0:
                                        valid_cache = False
                                        break
                                    cache_key_parts.append((h_item.get("role"), h_item.get("content")))
                                
                                if valid_cache and cache_key_parts:
                                    cache_key = tuple(cache_key_parts)
                                    # Store the processed history segments
                                    processed_history_cache[cache_key] = history[:current_idx]

        except Exception as e:
            print(f"Error reading output file: {e}. Starting fresh.")
            results = []
            results_map = {}
            processed_history_cache = {}
    
    # 3. Iterate and Process
    for debate_entry in input_data:
        for debate_name, turns in debate_entry.items():
            print(f"Scanning debate: {debate_name}")
            for i, turn in enumerate(turns):
                value = turn.get("value", 0)
                if value > 70:
                    continue
                
                # Check if this turn needs processing
                key = (debate_name, i)
                existing_index = results_map.get(key)
                
                # Prepare Speech Sequence
                motion = turn.get("辩题", "")
                side_label = turn.get("持方", "")
                current_speech_content = turn.get("本轮发言", "")
                
                speech_sequence = []
                # 1. Pro 1st
                if turn.get("正方一辩稿"):
                    speech_sequence.append({
                        "role": "正方一辩",
                        "debater_name": "正方",
                        "content": turn["正方一辩稿"]
                    })
                # 2. Con 1st
                if turn.get("反方一辩稿"):
                    speech_sequence.append({
                        "role": "反方一辩",
                        "debater_name": "反方",
                        "content": turn["反方一辩稿"]
                    })
                # 3. Opponent
                opponent_content = turn.get("对方发言")
                if opponent_content:
                    is_duplicate = False
                    for s in speech_sequence:
                        if s["content"].strip() == opponent_content.strip():
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        speech_sequence.append({
                            "role": "对方发言",
                            "debater_name": "对方",
                            "content": opponent_content
                        })
                
                # Check cache for background info (items before "本轮发言")
                background_key_parts = []
                for s in speech_sequence:
                    background_key_parts.append((s["role"], s["content"]))
                background_key = tuple(background_key_parts)
                
                # 4. Current
                speech_sequence.append({
                    "role": "本轮发言",
                    "debater_name": side_label,
                    "content": current_speech_content
                })
                
                # Determine Start State
                start_index = 0
                processed_history = []
                speech_analysis_history = []
                needs_processing = True
                
                # Priority 1: Check if already processed in results
                if existing_index is not None:
                    # Entry exists, check if valid
                    entry = results[existing_index]
                    history = entry.get("speech_analysis_history", [])
                    
                    # Check for invalid scores (0 or None)
                    first_invalid_idx = -1
                    if not history or len(history) != len(speech_sequence):
                         # History mismatch or empty, re-process all
                         first_invalid_idx = 0
                    else:
                        for idx, step in enumerate(history):
                            s = step.get("score")
                            if s is None or s == 0:
                                first_invalid_idx = idx
                                break
                    
                    if first_invalid_idx == -1:
                        # All valid
                        print(f"  Skipping {debate_name} Turn {i} (Already completed)")
                        needs_processing = False
                    else:
                        print(f"  Resuming {debate_name} Turn {i} from step {first_invalid_idx}")
                        start_index = first_invalid_idx
                        speech_analysis_history = history[:start_index]
                        # Rebuild context
                        for step in speech_analysis_history:
                            combined = f"{step['content']}\n\n[Judge's Analysis]:\n{step['judgment']}"
                            processed_history.append((step['role'], combined))
                
                # Priority 2: If not fully processed, check cache for background reuse
                elif use_cache and background_key in processed_history_cache:
                    cached_history = processed_history_cache[background_key]
                    print(f"  Reuse cached background for {debate_name} Turn {i}")
                    
                    # Use cached history
                    speech_analysis_history = list(cached_history) # Copy
                    start_index = len(cached_history) # Should be equal to len(speech_sequence) - 1 (the index of current speech)
                    
                    # Rebuild context
                    for step in speech_analysis_history:
                        combined = f"{step['content']}\n\n[Judge's Analysis]:\n{step['judgment']}"
                        processed_history.append((step['role'], combined))
                    
                    print(f"  Skipping first {start_index} steps (Cached)")
                
                else:
                    print(f"  Processing {debate_name} Turn {i} (New)")
                
                if not needs_processing:
                    continue

                # Process Loop
                final_score = None
                final_judgment = None
                final_score_raw = None
                
                for idx in range(start_index, len(speech_sequence)):
                    speech_item = speech_sequence[idx]
                    role = speech_item["role"]
                    debater_name = speech_item["debater_name"]
                    content = speech_item["content"]
                    is_last = (idx == len(speech_sequence) - 1)
                    
                    print(f"    Processing step {idx}: {role}")
                    
                    new_speech_obj = {"debater_name": debater_name, "content": content}
                    info_slide_text = f"Motion: {motion}\nSide: {debater_name}"
                    
                    # 1. Get Judgment (Retry Loop)
                    judgment = None
                    while True:
                        system_prompt = _format(UPDATE_SYSTEM_TEMPLATE, 
                                              new_speech=new_speech_obj, 
                                              prev_content=processed_history, 
                                              analyze_speech=GENERAL_ANALYZE_SPEECH, 
                                              use_previous=GENERAL_USE_PREVIOUS)
                        
                        user_prompt = _format(UPDATE_HUMAN_TEMPLATE, 
                                            info_slide=info_slide_text, 
                                            prev_content=processed_history, 
                                            new_speech=new_speech_obj)
                        
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                        
                        judgment = _call_api(client, MODEL, messages)
                        if judgment and len(judgment.strip()) > 10: # Simple validation
                            break
                        print(f"      Judgment failed or empty, retrying...")
                        time.sleep(2)

                    # 2. Get Score (Retry Loop)
                    score = None
                    score_resp = None
                    while True:
                        info = {"motion": motion, "pro_side": [{"name": "Pro"}], "con_side": [{"name": "Con"}]}
                        speech_prompt = _format(SPEECH_PROMPT_TEMPLATE, 
                                              info=info, 
                                              debater_name=debater_name, 
                                              judgment=judgment)
                        
                        schema_prompt = _format(PANEL_SCHEMA_PROMPT, schema=_score_schema())
                        final_prompt = speech_prompt + "\n\n" + schema_prompt
                        
                        messages_score = [{"role": "user", "content": final_prompt}]
                        score_resp = _call_api(client, MODEL, messages_score)
                        score = _parse_score(score_resp)
                        
                        if score is not None and score > 0:
                            break
                        print(f"      Score invalid ({score}), retrying...")
                        time.sleep(2)
                    
                    print(f"      [{role}] Score: {score}")
                    
                    # Record
                    step_record = {
                        "role": role,
                        "debater_name": debater_name,
                        "content": content,
                        "judgment": judgment,
                        "score": score,
                        "score_raw_response": score_resp
                    }
                    speech_analysis_history.append(step_record)
                    
                    # Update Context
                    combined_content = f"{content}\n\n[Judge's Analysis]:\n{judgment}"
                    processed_history.append((role, combined_content))
                    
                    if is_last:
                        final_score = score
                        final_judgment = judgment
                        final_score_raw = score_resp
                
                # Update Cache if applicable
                if use_cache and background_key_parts: # If there was a background
                    # Find where the background ends in the newly processed history
                    # It should be the part before "本轮发言"
                    current_idx_in_new = -1
                    for h_i, h_item in enumerate(speech_analysis_history):
                        if h_item.get("role") == "本轮发言":
                            current_idx_in_new = h_i
                            break
                    
                    if current_idx_in_new > 0:
                        # Double check if this prefix matches the background key
                        new_background_parts = []
                        valid_new_cache = True
                        for h_item in speech_analysis_history[:current_idx_in_new]:
                             if h_item.get("score") is None or h_item.get("score") == 0:
                                valid_new_cache = False
                                break
                             new_background_parts.append((h_item.get("role"), h_item.get("content")))
                        
                        if valid_new_cache and tuple(new_background_parts) == background_key:
                            processed_history_cache[background_key] = speech_analysis_history[:current_idx_in_new]
                            # print(f"    Updated cache for background")

                # Construct Final Result
                result_entry = {
                    "debate_name": debate_name,
                    "turn_index": i,
                    "motion": motion,
                    "side": side_label,
                    "speech": current_speech_content,
                    "ground_truth": value,
                    "predicted_score": final_score,
                    "judgment": final_judgment,
                    "score_raw_response": final_score_raw,
                    "speech_analysis_history": speech_analysis_history
                }
                
                # Update Results List
                if existing_index is not None:
                    results[existing_index] = result_entry
                else:
                    results.append(result_entry)
                    results_map[key] = len(results) - 1
                
                # Incremental Save
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"    Saved progress to {OUTPUT_FILE}")

    # Final Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"All done. Saved {len(results)} results to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_data(use_cache=True)
