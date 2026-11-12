"""Simulated on-call engineer who approves or rejects high-risk actions."""
from __future__ import annotations

import random

from .actions import Action


class SimulatedHuman:
    def __init__(self, seed: int = 0, accuracy: float = 0.9, delay_ticks: int = 4):
        self.rng = random.Random(seed)
        self.accuracy = accuracy
        self.delay_ticks = delay_ticks      # time to wake up and look (4 ticks = 2 min)
        self.requests = 0

    def review(self, action: Action, correct: Action | None) -> bool:
        self.requests += 1
        right = (correct is not None and action == correct)
        return right if self.rng.random() < self.accuracy else not right
