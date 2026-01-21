import json
import os
import re
import requests
import backoff
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from tqdm import tqdm
import tempfile
import shutil

# Configuration
OPENAI_API_KEY = ""
BASE_URL = ""
MODEL = ""
SERPER_API_KEY = ""
INPUT_FILE = ""
OUTPUT_DIR = ""
SUBJECTIVE_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "")
OBJECTIVE_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "")

# Toggle Switches
ENABLE_SUBJECTIVE = True
ENABLE_OBJECTIVE = True

class InspireScore:
    def __init__(self, openai_api_key: str, serper_api_key: str = None, model: str = None, base_url: str = None):
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=base_url
        )
        self.serper_api_key = serper_api_key
        self.model = model

    def evaluate_subjective(self, topic: str, debate_text: str) -> str:
        system_prompt = """
        You are an experienced debate judge tasked with evaluating debates. For each debate, you will assess both sides based on four key criteria: Emotional Appeal, Clarity of Argument and Reasoning, Logical Arrangement of Arguments, and Relevance to Debate Topic.

        For each of the four subdimensions, provide a score from 0 to 100 (with 0 being the lowest and 100 being the highest) for the argument provided. Additionally, provide a brief analysis for each subdimension.
        
        Scoring Criteria:
            1. **Emotional Appeal**  
                - How effectively does the argument connect with the audience emotionally? Does the argument evoke empathy, passion, or values?
                - **0**: No emotional appeal. The argument feels cold or disconnected.
                - **100**: Highly engaging emotionally, strongly connects with the audience.

            2. **Clarity of Argument and Reasoning**  
                - Are the arguments clearly presented? Is the reasoning sound and easy to follow?
                - **0**: The arguments are unclear or confusing.
                - **100**: The arguments are well-structured and easy to understand.

            3. **Logical Arrangement of Arguments**  
                - Is the argument presented in a logical, coherent manner? Does each point flow into the next without confusion?
                - **0**: The arguments are disorganized and difficult to follow.
                - **100**: The arguments follow a clear and logical progression.

            4. **Relevance to Debate Topic**  
                - Does the argument directly address the debate topic? Are there any irrelevant points or off-topic distractions?
                - **0**: Arguments that stray far from the topic.
                - **100**: Every argument is focused and relevant to the topic.

        Please output the result in the following JSON format:
        {
            "emotional_appeal": {
                "score": number,
                "analysis": "string"
            },
            "clarity": {
                "score": number,
                "analysis": "string"
            },
            "arrangement": {
                "score": number,
                "analysis": "string"
            },
            "relevance": {
                "score": number,
                "analysis": "string"
            },
            "total_score": number,
            "overall_analysis": "string"
        }
        """
        
        user_prompt = f"""
        Evaluate the debate argument on the topic: '{topic}'
        Argument to evaluate:
        {debate_text}
        
        Provide a JSON formatted response with scores (0-100) and comments for each criterion.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in subjective evaluation: {e}")
            return "{}"

    def evaluate_logical_validity(self, topic: str, debate_text: str) -> str:
        system_prompt = """
        Task: Logical Inference

        Input:
        <Reasoning and Analysis Process>: Provide a step-by-step analysis leading to the formulation of the argument.
        <Argument>: Summarize the primary argument derived from the analysis.

        Output:
            1. Convert Reasoning and Argument to First-Order Logic(FOL): Transform the reasoning and statements into formalized logic expressions using the following rules:
                1. Logical conjunction of expr1 and expr2: expr1 ∧ expr2
                2. Logical disjunction of expr1 and expr2: expr1 ∨ expr2
                3. Logical exclusive disjunction of expr1 and expr2: expr1 ⊕ expr2
                4. Logical negation of expr1: ¬expr1
                5. expr1 implies expr2: expr1 → expr2
                6. expr1 if and only if expr2: expr1 ↔ expr2
                7. Logical universal quantification: ∀x
                8. Logical existential quantification: ∃x
            2. Generate Inference Plan: Outline a plan to evaluate whether the conclusions logically follow from the premises using first-order logic inference rules.
            3. Solve Logic Puzzle: Determine the truth value (true, false, unknown) of each conclusion based on the premises and logical inferences.
                Make sure you carefully and fully understand the below requirements before execution the conclusion:
                1.Please clearly indicate whether the conclusion statement is true, false or unknown using curly bracket {true/false/unknown}!!! The answer will only be either true, false or unknown.
                The definition of the three options are:
                    True: A statement is "true" if it necessarily follows from the given premises using logical rules.
                    False: A statement is "false" if it is contradicted by the premises or its negation is logically inferred from them.
                    Unknown: A statement is "unknown" if there is insufficient information in the premises to determine its truth value conclusively.
                2. Make sure you must only use the premises to infer the conclusion. Do not use any information that is not exist or cannot be inferred from the premises.If some premise is semantically equal, such as "love the most" and "favorite", you can consider this as a valid assumption. You can make assumption to entity if it is very obvious but not logical relationship. For instance, an entity with an obvious human name can be inferred as a human.
                3. Make sure you abide the 16 provided first-order logic rules and formula when making logical inference. You need to clearly indicate what logic rules and formula you used.
                4. Please note that in first-order logic if there exists a conditional statement in the conclusion such as "If...", the if part will be considered as a premise. And if there is premise contradicts the if statement, you need to use the premise in the if statement as priority and neglect the contradicted one.
                5. Be careful with the parentheses. Make sure you following the rules such as Order of Operations (The order is usually: negation (¬), conjunction (and, ∧), disjunction (or, ∨), implication (→), and biconditional (↔). ), Nested Parentheses (The expression inside the innermost set of parentheses is evaluated first, then the next outer set, and so on.). 
                6. Make sure you not only access the premises in first-order logic, but also access its corresponding natural language format. The natural language format premises should be prioritized when there is inconsistent between natural language and first-order logic.
                7. When inferring new knowledge, please clear indicate which premises you used or the steps you refer to. For instance, if you use Premise 1 and a knowledge from Step 5, you should clearly indicate that "Combine Premise 1 and Step 5".
                8. You should also use natural language to explain the logical process in each step. Please also indicate the premises and steps you refer to when making the logical process.
        
        Provide the output in a structured format.
        """
        
        user_prompt = f"""
        Topic: {topic}
        
        Argument to analyze:
        {debate_text}
        
        Perform the logical validity evaluation as described.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in logical evaluation: {e}")
            return ""

    def extract_atomic_facts(self, topic: str, debate_text: str) -> str:
        system_prompt = """
        You are tasked with breaking down reasoning processes and arguments into atomic facts. Follow these instructions:
        1. An atomic fact is a single, standalone statement containing one idea or piece of information.
        2. Each atomic fact should capture a distinct piece of information and avoid overlaps.
        3. Break down the argument into separate facts labeled sequentially as fact-1, fact-2, and so on.
        4. Provide the output in JSON format as follows:
        {
          "facts": [
            "fact-1: <atomic fact>",
            "fact-2: <atomic fact>",
            ...
          ]
        }
        """
        
        user_prompt = f"""
        Topic: {topic}
        
        Argument:
        {debate_text}
        
        Break the argument into atomic facts according to the instructions. Provide the response in JSON format.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in fact extraction: {e}")
            return "{}"

    def web_search(self, query: str) -> dict:
        if not self.serper_api_key:
            return {}
            
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.serper_api_key,
            'Content-Type': 'application/json'
        }
        data = {"q": query, "num": 3}
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Web search error: {e}")
            return {}
        
    @backoff.on_exception(backoff.expo, (RateLimitError, APIError, APIConnectionError), max_tries=3)
    def generate_search_queries(self, fact_json: str) -> str:
        system_prompt = """
        You are an expert fact-checking assistant. Your task is to analyze the provided JSON content and generate relevant queries that should be searched on the internet (e.g., using Google) to validate the facts.
        - Carefully examine the JSON content and identify the main topics or themes that emerge from the claims.
        - Propose precise and actionable search queries that can help verify the claims.
        - Your response should only include the search queries in a clear and concise list.
        """
        
        user_prompt = f"""
        Analyze the following JSON content and generate search queries to validate the claims:
        {fact_json}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in query generation: {e}")
            return ""

    def verify_facts(self, search_results: dict, fact_json: str) -> str:
        system_prompt = """
        You are an expert fact-checking assistant. Your task is to verify the provided facts in the JSON content by analyzing the given search results.

        - For each fact, determine if it is "true," "false," or "unknown" based on the evidence provided in the search results.
        - "true": Strong and reliable evidence supports the fact.
        - "false": Strong and reliable evidence disproves the fact.
        - "unknown": Evidence is insufficient or inconclusive to verify the fact.

        - Be specific and logical in your assessment, focusing on the factual accuracy of each claim.
        - If the search results are empty, rely on your existing knowledge to assess the factual accuracy of the claims.

        Output your analysis in the following JSON format:
        {
            "fact-1": "true/false/unknown",
            "fact-2": "true/false/unknown",
            ...
        }
        """
        
        user_prompt = f"""
        Given the following inputs:

        JSON Content: {fact_json}
        Search Results: {search_results}

        Analyze the search results (or, if empty, use your existing knowledge) to verify the facts in the JSON content. Provide your conclusions in the specified JSON format.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in fact verification: {e}")
            return "{}"

def atomic_save_json(data, file_path):
    """Safely write JSON to file using atomic replacement to prevent corruption."""
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=dir_name, encoding='utf-8') as tf:
            json.dump(data, tf, ensure_ascii=False, indent=2)
            temp_name = tf.name
        # Atomically rename
        os.replace(temp_name, file_path)
    except Exception as e:
        print(f"Error saving results to {file_path}: {e}")
        if 'temp_name' in locals() and os.path.exists(temp_name):
            os.remove(temp_name)

def main():
    print("Initializing InspireScore Evaluator...")
    evaluator = InspireScore(
        openai_api_key=OPENAI_API_KEY,
        serper_api_key=SERPER_API_KEY,
        base_url=BASE_URL,
        model=MODEL
    )

    print(f"Reading input file: {INPUT_FILE}")
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read input file: {e}")
        return

    items_to_process = []
    if isinstance(data, list):
        for entry in data:
            for debate_id, turns in entry.items():
                if isinstance(turns, list):
                    for turn in turns:
                        if turn.get("value", 0) < 75:
                            items_to_process.append(turn)
    
    print(f"Found {len(items_to_process)} items with value < 75.")
    
    # Load existing results to resume if file exists
    subjective_results = []
    objective_results = []
    
    if ENABLE_SUBJECTIVE and os.path.exists(SUBJECTIVE_OUTPUT_FILE):
        try:
            with open(SUBJECTIVE_OUTPUT_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    subjective_results = json.loads(content)
                else:
                    subjective_results = []
            print(f"Loaded {len(subjective_results)} existing subjective results.")
        except Exception as e:
            print(f"Warning: Could not load existing subjective results ({e}). Starting fresh/backup recommended if this is unexpected.")
            # If load fails, we default to empty list, effectively restarting.
            subjective_results = []

    if ENABLE_OBJECTIVE and os.path.exists(OBJECTIVE_OUTPUT_FILE):
        try:
            with open(OBJECTIVE_OUTPUT_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    objective_results = json.loads(content)
                else:
                    objective_results = []
            print(f"Loaded {len(objective_results)} existing objective results.")
        except Exception as e:
            print(f"Warning: Could not load existing objective results ({e}). Starting fresh/backup recommended if this is unexpected.")
            objective_results = []

    # Identify processed items by content (exact match)
    processed_subjective = set(r.get("content") for r in subjective_results)
    processed_objective = set(r.get("content") for r in objective_results)
    
    processed_count_sub = len(processed_subjective)
    processed_count_obj = len(processed_objective)

    print(f"Already processed: {processed_count_sub} subjective, {processed_count_obj} objective.")

    for item in tqdm(items_to_process):
        topic = item.get("辩题", "")
        content = item.get("本轮发言", "")
        if not content:
            continue
            
        # Process Subjective
        if ENABLE_SUBJECTIVE:
            if content not in processed_subjective:
                subjective_json_str = evaluator.evaluate_subjective(topic, content)
                subjective_res = {}
                try:
                    subjective_res = json.loads(subjective_json_str)
                except:
                    subjective_res = {"raw_output": subjective_json_str}
                
                # Helper to get score safely
                def get_score(data, key):
                    try:
                        val = data.get(key, {}).get("score")
                        if val is not None:
                            return float(val)
                    except:
                        pass
                    return None

                scores_list = []
                scores_list.append(get_score(subjective_res, "emotional_appeal"))
                scores_list.append(get_score(subjective_res, "clarity"))
                scores_list.append(get_score(subjective_res, "arrangement"))
                scores_list.append(get_score(subjective_res, "relevance"))
                valid_scores = [s for s in scores_list if s is not None]
                subjective_average = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

                sub_item = {
                    "topic": topic,
                    "side": item.get("持方", ""),
                    "content": content,
                    "ground_truth": item.get("value"),
                    "inspire_evaluation": {
                        "subjective": subjective_res,
                        "subjective_average_score": subjective_average
                    }
                }
                subjective_results.append(sub_item)
                processed_subjective.add(content)
                atomic_save_json(subjective_results, SUBJECTIVE_OUTPUT_FILE)

        # Process Objective
        if ENABLE_OBJECTIVE:
            if content not in processed_objective:
                # 2. Logical Validity Evaluation
                logical_res = evaluator.evaluate_logical_validity(topic, content)
                logical_conclusions = []
                try:
                    matches = re.findall(r'\{([a-zA-Z]+)\}', logical_res)
                    if matches:
                        logical_conclusions = [m.lower() for m in matches]
                except:
                    pass

                # Calculate logical score
                logical_score = 0.0
                if logical_conclusions:
                    true_count = logical_conclusions.count('true')
                    logical_score = (true_count / len(logical_conclusions)) * 100

                # 3. Fact Verification
                facts_json_str = evaluator.extract_atomic_facts(topic, content)
                queries = evaluator.generate_search_queries(facts_json_str)
                
                search_results_combined = {}
                if queries:
                    lines = queries.split('\n')
                    query_lines = []
                    for line in lines:
                        line = line.strip()
                        line = re.sub(r'^[\-\*0-9\.]+\s*', '', line)
                        if line and not line.lower().startswith(('here', 'sure', 'certainly')):
                            query_lines.append(line)
                    
                    if not query_lines and queries.strip():
                        query_lines = [queries[:200]]

                    for q in query_lines[:2]:
                        res = evaluator.web_search(q)
                        if res:
                            search_results_combined[q] = res

                verification_json_str = evaluator.verify_facts(json.dumps(search_results_combined), facts_json_str)
                verification_res = {}
                try:
                    verification_res = json.loads(verification_json_str)
                except:
                    verification_res = {"raw_output": verification_json_str}
                
                # Calculate fact score
                fact_score = 0.0
                fact_items = [v for k, v in verification_res.items() if k.startswith('fact-')]
                if fact_items:
                    true_facts = [v for v in fact_items if str(v).lower() == 'true']
                    fact_score = (len(true_facts) / len(fact_items)) * 100

                # Calculate Objective Average
                objective_average = (logical_score + fact_score) / 2

                obj_item = {
                    "topic": topic,
                    "side": item.get("持方", ""),
                    "content": content,
                    "ground_truth": item.get("value"),
                    "inspire_evaluation": {
                        "logical": {
                            "full_analysis": logical_res,
                            "conclusions": logical_conclusions,
                            "score": logical_score
                        },
                        "fact_check": {
                            "facts": facts_json_str,
                            "queries": queries,
                            "search_results": str(search_results_combined)[:500] + "...", 
                            "verification": verification_res,
                            "score": fact_score
                        },
                        "objective_average_score": objective_average
                    }
                }
                objective_results.append(obj_item)
                processed_objective.add(content)
                atomic_save_json(objective_results, OBJECTIVE_OUTPUT_FILE)

    print(f"All done.")
    if ENABLE_SUBJECTIVE:
        print(f"Saved subjective results to {SUBJECTIVE_OUTPUT_FILE}")
    if ENABLE_OBJECTIVE:
        print(f"Saved objective results to {OBJECTIVE_OUTPUT_FILE}")

if __name__ == "__main__":
    main()
