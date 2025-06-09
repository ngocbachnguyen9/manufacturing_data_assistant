from typing import Dict


class CostTracker:
    """A simple utility to track API token usage and monetary cost."""

    def __init__(self, cost_config: Dict):
        self.cost_config = cost_config
        self.reset()

    def reset(self):
        """Resets all counters to zero for a new task."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    def log_transaction(
        self, input_tokens: int, output_tokens: int, model_name: str
    ):
        """Logs tokens and calculates cost for a single API call."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        model_cost_info = self.cost_config.get(model_name, {})
        cost_per_1m_in = model_cost_info.get("cost_per_1m_tokens_input", 0)
        cost_per_1m_out = model_cost_info.get("cost_per_1m_tokens_output", 0)

        cost = ((input_tokens / 1_000_000) * cost_per_1m_in) + (
            (output_tokens / 1_000_000) * cost_per_1m_out
        )
        self.total_cost += cost

    def get_summary(self) -> Dict:
        """Returns a summary of the tracked metrics."""
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens
            + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 6),
        }