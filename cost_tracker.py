"""Spending estimation, limits, and abort logic for UpskillScan."""


class CostLimitExceeded(Exception):
    """Raised when estimated spend exceeds the configured limit."""
    pass


class CostTracker:
    """Tracks API spending and enforces budget limits."""

    BATCH_PRICING = {
        "claude-haiku-4-5-20251001": {"input": 0.50, "output": 2.50},
        "claude-sonnet-4-5-20250929": {"input": 1.50, "output": 7.50},
        "claude-opus-4-5-20251101": {"input": 2.50, "output": 12.50},
    }
    SYNC_PRICING = {
        "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
        "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
        "claude-opus-4-5-20251101": {"input": 5.00, "output": 25.00},
    }

    def __init__(self, model: str, cost_limit: float, use_batch: bool = True):
        self.model = model
        self.cost_limit = cost_limit
        self.use_batch = use_batch
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _get_pricing(self) -> dict:
        pricing_table = self.BATCH_PRICING if self.use_batch else self.SYNC_PRICING
        if self.model in pricing_table:
            return pricing_table[self.model]
        # Fallback: use Haiku pricing as conservative default for unknown models
        print(f"  Warning: No pricing data for {self.model}, using Haiku rates as estimate")
        return pricing_table["claude-haiku-4-5-20251001"]

    def estimate_batch_cost(
        self,
        num_requests: int,
        avg_input_tokens: int = 1500,
        avg_output_tokens: int = 300,
    ) -> float:
        """Pre-flight cost estimate before submitting batch."""
        pricing = self._get_pricing()
        total_input = num_requests * avg_input_tokens
        total_output = num_requests * avg_output_tokens
        cost = (total_input / 1_000_000 * pricing["input"]) + (
            total_output / 1_000_000 * pricing["output"]
        )

        if cost > self.cost_limit * 0.8:
            print(
                f"  WARNING: Estimated cost ${cost:.2f} is >{80}% of limit ${self.cost_limit:.2f}"
            )
        return cost

    def update_from_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Update actual token counts from a single API response."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def update_from_batch_results(self, results_iter) -> None:
        """Update actual token counts from batch response usage data.

        Args:
            results_iter: Iterable of batch result entries with .result.message.usage
        """
        for entry in results_iter:
            if entry.result.type == "succeeded":
                usage = entry.result.message.usage
                self.total_input_tokens += usage.input_tokens
                self.total_output_tokens += usage.output_tokens

    @property
    def estimated_cost(self) -> float:
        """Current estimated total spend based on actual token counts."""
        pricing = self._get_pricing()
        return (self.total_input_tokens / 1_000_000 * pricing["input"]) + (
            self.total_output_tokens / 1_000_000 * pricing["output"]
        )

    @property
    def budget_remaining(self) -> float:
        return self.cost_limit - self.estimated_cost

    def check_limit(self) -> None:
        """Raise CostLimitExceeded if estimated cost exceeds limit."""
        if self.estimated_cost > self.cost_limit:
            raise CostLimitExceeded(
                f"ABORT: Estimated spend ${self.estimated_cost:.2f} exceeds limit "
                f"${self.cost_limit:.2f}. Increase COST_LIMIT_USD in .env to continue."
            )

    def print_estimate(self, num_requests: int) -> None:
        """Print a formatted cost estimate for user confirmation."""
        est = self.estimate_batch_cost(num_requests)
        mode = "Batch API" if self.use_batch else "Synchronous"
        print(f"\n--- Cost Estimate ---")
        print(f"Requests: {num_requests:,}")
        print(f"Model: {self.model} ({mode})")
        print(f"Estimated cost: ${est:.2f}")
        print(f"Budget remaining: ${self.budget_remaining:.2f} / ${self.cost_limit:.2f}")

    def print_actual(self) -> None:
        """Print actual spend so far."""
        print(f"\n--- Actual Spend ---")
        print(f"Input tokens: {self.total_input_tokens:,}")
        print(f"Output tokens: {self.total_output_tokens:,}")
        print(f"Estimated cost: ${self.estimated_cost:.2f}")
        print(f"Budget remaining: ${self.budget_remaining:.2f} / ${self.cost_limit:.2f}")
