from dataclasses import dataclass, field

@dataclass
class MetricsTracker:
    """Class to collect the four experimental metrics for the thesis."""
    
    semantic_correct: int = 0
    semantic_incorrect: int = 0
    
    total_llm_calls: int = 0
    valid_json_count: int = 0
    invalid_json_count: int = 0
    autocorrect_triggers: int = 0
    
    session_history: list = field(default_factory=list)
    
    ignored_no_wake: int = 0
    llm_no_action_decisions: int = 0

    def print_summary(self):
        print("\n" + "█"*60)
        print("📊 EXPERIMENTAL EVALUATION REPORT - JONAY")
        print("█"*60)
        
        total_eval = self.semantic_correct + self.semantic_incorrect
        accuracy = (self.semantic_correct / total_eval * 100) if total_eval > 0 else 0
        print(f"\n🔹 Metric 1 – Semantic accuracy:")
        print(f"   - Accuracy (manual/offline): {accuracy:.2f}% ({self.semantic_correct}/{total_eval})")
        print(f"   - Note: Detailed data available in session_history.")

        total_json = self.total_llm_calls
        json_valid_perc = (self.valid_json_count / total_json * 100) if total_json > 0 else 0
        print(f"\n🔹 Metric 2 – Structural JSON validity:")
        print(f"   - Correctly formatted JSONs: {self.valid_json_count}/{total_json} ({json_valid_perc:.1f}%)")
        print(f"   - JSONs with critical errors: {self.invalid_json_count}")
        print(f"   - Autocorrect triggers (duration/type): {self.autocorrect_triggers}")

        print(f"\n🔹 Metric 3 – Linguistic robustness:")
        print(f"   - Samples recorded for analysis: {len(self.session_history)}")
        print(f"   - (Compare consistency between synonyms in the session history)")

        print(f"\n🔹 Metric 4 – Safety (passive):")
        print(f"   - Wake-word filter (ignored commands): {self.ignored_no_wake}")
        print(f"   - Brain 'No Action' decisions: {self.llm_no_action_decisions}")
        
        print("\n" + "█"*60 + "\n")
