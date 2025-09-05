"""
RL Policy Stub for Task 4 Day 2
Simple reinforcement learning system for quality retry decisions
"""

import random
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    """Available actions for RL agent"""
    ACCEPT = "accept"
    RETRY_HIGHER_QUALITY = "retry_higher_quality"
    RETRY_LOWER_COST = "retry_lower_cost"
    ESCALATE_TIER = "escalate_tier"


@dataclass
class Experience:
    """RL experience tuple"""
    state: Dict[str, Any]
    action: Action
    reward: float
    next_state: Dict[str, Any]
    timestamp: float


@dataclass
class State:
    """Current state representation"""
    vmaf_score: float
    latency_ms: float
    cost_usd: float
    tier: str
    quality_preset: str
    device_class: str
    task_complexity: str


class RLPolicy:
    """Simple RL policy for adaptive video generation"""

    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Simple Q-table: state -> action -> q_value
        self.q_table: Dict[str, Dict[Action, float]] = {}

        # Experience replay buffer
        self.experience_buffer: List[Experience] = []
        self.max_buffer_size = 1000

        # Policy parameters
        self.vmaf_threshold = 70.0  # Minimum acceptable VMAF
        self.cost_budget_usd = 0.05
        self.latency_budget_ms = 30000

        # Exploration rate
        self.epsilon = 0.1

    def _get_state_key(self, state: State) -> str:
        """Convert state to hashable key"""
        return f"{state.device_class}_{state.task_complexity}_{state.tier}_{state.quality_preset}"

    def _get_reward(self, state: State, action: Action, next_state: Optional[State] = None) -> float:
        """Calculate reward for state-action pair"""
        reward = 0.0

        # VMAF quality reward
        if state.vmaf_score >= self.vmaf_threshold:
            reward += 10.0
        else:
            reward -= 5.0

        # Cost penalty
        if state.cost_usd > self.cost_budget_usd:
            reward -= (state.cost_usd - self.cost_budget_usd) * 100
        else:
            reward += (self.cost_budget_usd - state.cost_usd) * 50

        # Latency penalty
        if state.latency_ms > self.latency_budget_ms:
            reward -= (state.latency_ms - self.latency_budget_ms) / 1000
        else:
            reward += (self.latency_budget_ms - state.latency_ms) / 2000

        # Action-specific rewards
        if action == Action.RETRY_HIGHER_QUALITY and state.vmaf_score >= self.vmaf_threshold:
            reward += 5.0  # Successful quality improvement
        elif action == Action.RETRY_LOWER_COST and state.cost_usd <= self.cost_budget_usd:
            reward += 3.0  # Successful cost reduction

        return reward

    def _get_available_actions(self, state: State) -> List[Action]:
        """Get available actions for current state"""
        actions = [Action.ACCEPT]

        # Can retry for higher quality if current VMAF is low
        if state.vmaf_score < self.vmaf_threshold:
            actions.append(Action.RETRY_HIGHER_QUALITY)

        # Can retry for lower cost if current cost is high
        if state.cost_usd > self.cost_budget_usd:
            actions.append(Action.RETRY_LOWER_COST)

        # Can escalate tier if on edge/local
        if state.tier in ['edge', 'local']:
            actions.append(Action.ESCALATE_TIER)

        return actions

    def choose_action(self, state: State) -> Action:
        """Choose action using epsilon-greedy policy"""
        state_key = self._get_state_key(state)
        available_actions = self._get_available_actions(state)

        # Initialize Q-values for new state
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in available_actions}

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore: random action
            return random.choice(available_actions)
        else:
            # Exploit: best action
            action_values = self.q_table[state_key]
            best_action = max(action_values.items(), key=lambda x: x[1])[0]

            # Ensure best action is still available
            if best_action in available_actions:
                return best_action
            else:
                return random.choice(available_actions)

    def update_policy(self, experience: Experience):
        """Update Q-values using experience"""
        state_key = self._get_state_key(State(**experience.state))
        next_state_key = self._get_state_key(State(**experience.next_state))

        # Initialize Q-values if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {}

        # Q-learning update
        current_q = self.q_table[state_key].get(experience.action, 0.0)

        # Max Q-value for next state
        next_q_values = self.q_table[next_state_key].values()
        max_next_q = max(next_q_values) if next_q_values else 0.0

        # Update Q-value
        new_q = current_q + self.learning_rate * (
            experience.reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[state_key][experience.action] = new_q

        # Add to experience buffer
        self.experience_buffer.append(experience)
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)

    def should_retry(self, current_state: State) -> Tuple[bool, Optional[Action], str]:
        """
        Decide whether to retry generation and with what action
        Returns: (should_retry, action, reason)
        """
        action = self.choose_action(current_state)

        if action == Action.ACCEPT:
            return False, None, "Quality and cost acceptable"

        # Calculate expected improvement
        reward = self._get_reward(current_state, action)

        if action == Action.RETRY_HIGHER_QUALITY:
            if current_state.vmaf_score < self.vmaf_threshold:
                return True, action, f"VMAF {current_state.vmaf_score:.1f} below threshold {self.vmaf_threshold}"
            else:
                return False, None, f"VMAF {current_state.vmaf_score:.1f} already acceptable"

        elif action == Action.RETRY_LOWER_COST:
            if current_state.cost_usd > self.cost_budget_usd:
                return True, action, f"Cost ${current_state.cost_usd:.3f} exceeds budget ${self.cost_budget_usd:.3f}"
            else:
                return False, None, f"Cost ${current_state.cost_usd:.3f} within budget"

        elif action == Action.ESCALATE_TIER:
            if current_state.tier in ['edge', 'local']:
                return True, action, f"Can escalate from {current_state.tier} tier for better performance"
            else:
                return False, None, f"Already on {current_state.tier} tier"

        return False, None, "No retry needed"

    def get_policy_stats(self) -> Dict[str, Any]:
        """Get policy statistics"""
        total_experiences = len(self.experience_buffer)
        avg_reward = sum(exp.reward for exp in self.experience_buffer) / max(total_experiences, 1)

        action_counts = {}
        for exp in self.experience_buffer:
            action_counts[exp.action] = action_counts.get(exp.action, 0) + 1

        return {
            'total_experiences': total_experiences,
            'average_reward': avg_reward,
            'action_distribution': action_counts,
            'q_table_size': len(self.q_table),
            'exploration_rate': self.epsilon
        }

    def reset_policy(self):
        """Reset policy to initial state"""
        self.q_table.clear()
        self.experience_buffer.clear()

    def set_thresholds(self, vmaf_threshold: float = None, cost_budget: float = None, latency_budget: int = None):
        """Update policy thresholds"""
        if vmaf_threshold is not None:
            self.vmaf_threshold = vmaf_threshold
        if cost_budget is not None:
            self.cost_budget_usd = cost_budget
        if latency_budget is not None:
            self.latency_budget_ms = latency_budget


# Global RL policy instance
_rl_policy = None

def get_rl_policy() -> RLPolicy:
    """Get global RL policy instance"""
    global _rl_policy
    if _rl_policy is None:
        _rl_policy = RLPolicy()
    return _rl_policy