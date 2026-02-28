import csv
import json
import time
import logging
import statistics
import os
from datetime import datetime

from llm_brain import LLMGo2Brain, ALLOWED_ACTIONS
from metrics_tracker import MetricsTracker

logger = logging.getLogger("Auto-Eval")

def setup_logging(log_filename):
    """Configure the logger to write to both console and a file with timestamps."""
    logger.setLevel(logging.DEBUG)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def parse_expected_move(expected_value_str):
    """Parse the CSV format 'x=0.3;dur=2' into a dictionary."""
    params = {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'duration': 2.0}
    if expected_value_str == 'none':
        return params
    
    parts = expected_value_str.split(';')
    for part in parts:
        if '=' in part:
            k, v = part.split('=')
            k = k.strip()
            if k == 'dur': k = 'duration'
            params[k] = float(v.strip())
    return params

def check_sign_match(val_llm, val_exp):
    """
    For move parameters, we care about the sign (direction) rather than exact value.
    This function checks if the LLM's value has the same sign as the expected value.
    """
    if val_exp == 0.0:
        return val_llm == 0.0
    elif val_exp > 0.0:
        return val_llm > 0.0
    else:
        return val_llm < 0.0

def check_match(llm_actions, expected_type, expected_value, metrics):
    """
    Evaluate semantics and record autocorrect triggers.
    """
    if llm_actions is None:
        metrics.semantic_incorrect += 1
        return False

    if not isinstance(llm_actions, list): 
        llm_actions = []

    if expected_type == "none":
        is_ok = len(llm_actions) == 0
        if is_ok: metrics.semantic_correct += 1
        else: metrics.semantic_incorrect += 1
        return is_ok

    if len(llm_actions) == 0:
        metrics.semantic_incorrect += 1
        return False

    first_action = llm_actions[0]
    llm_type = str(first_action.get("type", "")).lower()
    llm_val = str(first_action.get("value", first_action.get("action", ""))).upper()
    expected_val_upper = str(expected_value).upper()

    is_type_match = (llm_type == expected_type.lower())
    
    if not is_type_match:
        if expected_type.lower() == "action" and expected_val_upper in ["FREE_WALK", "TROT_RUN"] and llm_type == "move":
            is_type_match = True
            logger.debug(f"      ↳ Accepting 'move' as a valid equivalent for action {expected_val_upper}.")
        elif expected_type.lower() == "move" and llm_type == "action" and llm_val in ["FREE_WALK", "TROT_RUN"]:
            is_type_match = True
            logger.debug(f"      ↳ Accepting action '{llm_val}' as a valid equivalent for 'move'.")

    if not is_type_match:
        if llm_type.upper() in ALLOWED_ACTIONS:
            metrics.autocorrect_triggers += 1
        metrics.semantic_incorrect += 1
        return False
    
    if expected_type.lower() == "action":
        if llm_type == "move":
            is_ok = True
        else:
            is_ok = llm_val == expected_val_upper
            if not is_ok and llm_val in ["DANCE1", "DANCE2"] and expected_val_upper in ["DANCE1", "DANCE2"]:
                is_ok = True

    elif expected_type.lower() == "move":
        if llm_type == "action":
            is_ok = True
        else:
            llm_params = first_action.get("params", {})
            exp_dict = parse_expected_move(expected_value)
            
            try:
                dur = float(llm_params.get("duration", 2.0))
                if dur < 1.0: metrics.autocorrect_triggers += 1
                
                llm_x = float(llm_params.get("x", 0))
                llm_y = float(llm_params.get("y", 0))
                llm_yaw = float(llm_params.get("yaw", 0))
                
                is_ok = (check_sign_match(llm_x, exp_dict['x']) and 
                         check_sign_match(llm_y, exp_dict['y']) and 
                         check_sign_match(llm_yaw, exp_dict['yaw']) and
                         dur == exp_dict['duration'])
            except (ValueError, TypeError):
                is_ok = False
            
    if is_ok: metrics.semantic_correct += 1
    else: metrics.semantic_incorrect += 1
    return is_ok

class AutoEvaluator:
    def __init__(self, model="llama3:latest", metrics=None):
        logger.info(f"🧠 Initializing Auto-Evaluator with model: {model}")
        self.brain = LLMGo2Brain(local_model=model)
        self.metrics = metrics if metrics else MetricsTracker()

    def run_test(self, text):
        self.metrics.total_llm_calls += 1
        
        t0_llm = time.time()
        actions, description = self.brain.process(text) 
        llm_lat = time.time() - t0_llm
        
        if actions is None:
            self.metrics.invalid_json_count += 1
        else:
            self.metrics.valid_json_count += 1
            if not actions:
                self.metrics.llm_no_action_decisions += 1
                
        return actions, description, llm_lat

def run_auto_experiment(input_csv="dataset.csv", output_csv="auto_results.csv", num_repetitions=10):
    """Runs the automatic evaluation experiment on the provided dataset CSV, repeating each command multiple times for robustness. Saves results to a new CSV and prints a summary report."""
    metrics = MetricsTracker()
    evaluator = AutoEvaluator(metrics=metrics)
    results = []
    
    logger.info(f"📂 Loading dataset: {input_csv}")
    
    try:
        with open(input_csv, mode='r', encoding='utf-8') as infile:
            reader = list(csv.DictReader(infile))
            total = len(reader)
            
            for i, row in enumerate(reader):
                command_test = row['command']

                logger.info(f"\n[{i+1}/{total}] 🤖 Processing: '{command_test}' (Running {num_repetitions} times)")
                
                correct_count = 0
                latencies = []
                obtained_actions_history = []
                
                for j in range(num_repetitions):
                    logger.debug(f"   ↳ Iteration {j+1}/{num_repetitions}...")
                    
                    llm_actions, llm_description, llm_lat = evaluator.run_test(command_test)
                    is_correct = check_match(llm_actions, row['expected_type'], row['expected_value'], metrics)
                    
                    if is_correct:
                        correct_count += 1
                    latencies.append(llm_lat)
                    
                    act_str = json.dumps(llm_actions) if llm_actions is not None else "Structural Error"
                    obtained_actions_history.append(act_str)
                    
                    resultado_str = "✅ CORRECT" if is_correct else "❌ INCORRECT"
                    logger.debug(f"      ⏱️ Latency: {llm_lat:.2f}s | Semantics: {resultado_str}")
                    
                    if not is_correct:
                        logger.debug(f"      Expected : {row['expected_type']} -> {row['expected_value']}")
                        logger.debug(f"      Obtained : {llm_actions}")
                    
                    metrics.session_history.append({
                        "id": row['id'],
                        "input": command_test,
                        "iteration": j + 1,
                        "actions": llm_actions,
                        "match": is_correct
                    })

                accuracy = (correct_count / num_repetitions) * 100
                avg_latency = statistics.mean(latencies) if latencies else 0.0
                
                logger.info(f"   📊 Results for '{command_test}': Accuracy: {accuracy}% | Avg Latency: {avg_latency:.2f}s")

                results.append({
                    "id": row['id'],
                    "command": command_test,
                    "accuracy_%": f"{accuracy:.1f}",
                    "avg_latency": round(avg_latency, 2),
                    "Expected Action": row['expected_value'],
                    "Obtained Actions": json.dumps(obtained_actions_history)
                })
                
    except FileNotFoundError:
        logger.error(f"❌ Error: File not found: {input_csv}")
        return
    except KeyboardInterrupt:
        logger.info("\n⏹️ Test interrupted.")

    if results:
        with open(output_csv, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            
        metrics.print_summary()

if __name__ == "__main__":
    DATASET_PATH = "LLM-WebRTC/dataset/dataset.csv"
    
    os.makedirs("LLM-WebRTC/results", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    OUTPUT_CSV_PATH = f"LLM-WebRTC/results/auto_experiment_results_{timestamp}.csv"
    OUTPUT_LOG_PATH = f"LLM-WebRTC/results/auto_experiment_logs_{timestamp}.txt"
    
    setup_logging(OUTPUT_LOG_PATH)
    
    logger.info(f"📝 CSV Results will be saved to: {OUTPUT_CSV_PATH}")
    logger.info(f"📝 Raw Logs will be saved to: {OUTPUT_LOG_PATH}")
    
    NUM_REPETITIONS = 10 
    
    run_auto_experiment(input_csv=DATASET_PATH, output_csv=OUTPUT_CSV_PATH, num_repetitions=NUM_REPETITIONS)