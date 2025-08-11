# utils/timebudget.py
import torch

class TimeBudget:
    def __init__(self, budget_ms: float):
        self.budget_ms = budget_ms
        self.start = torch.cuda.Event(enable_timing=True)
        self.end   = torch.cuda.Event(enable_timing=True)

    def begin(self):
        self.start.record()

    def left_ms(self):
        self.end.record()
        torch.cuda.synchronize()
        elapsed = self.start.elapsed_time(self.end)  # ms
        left = max(0.0, self.budget_ms - elapsed)
        return left
