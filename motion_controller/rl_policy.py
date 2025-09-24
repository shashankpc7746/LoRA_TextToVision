"""
RL Policy for Task-7 Quality Leap
Reinforcement learning for parameter optimization and quality control
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import random
import json
from datetime import datetime


class QualityState:
    """Represents the current state of video generation quality"""

    def __init__(self, vmqf_score: float = 0.0, lip_sync_score: float = 0.0,
                 temporal_consistency: float = 0.0, resolution: Tuple[int, int] = (512, 512),
                 generation_time: float = 0.0, cost: float = 0.0):
        self.vmaf_score = vmqf_score
        self.lip_sync_score = lip_sync_score
        self.temporal_consistency = temporal_consistency
        self.resolution = resolution
        self.generation_time = generation_time
        self.cost = cost

    def to_tensor(self) -> torch.Tensor:
        """Convert state to tensor representation"""
        return torch.tensor([
            self.vmaf_score,
            self.lip_sync_score,
            self.temporal_consistency,
            self.resolution[0] / 1920.0,  # Normalize width
            self.resolution[1] / 1080.0,  # Normalize height
            self.generation_time / 300.0,  # Normalize time (max 5 min)
            self.cost / 1.0  # Normalize cost (max $1)
        ], dtype=torch.float32)

    def get_quality_score(self) -> float:
        """Calculate overall quality score"""
        # Weighted combination of quality metrics
        weights = {
            'vmaf': 0.3,
            'lip_sync': 0.3,
            'temporal': 0.2,
            'resolution': 0.1,
            'efficiency': 0.1  # Time/cost efficiency
        }

        resolution_score = min(1.0, (self.resolution[0] * self.resolution[1]) / (1920 * 1080))
        efficiency_score = max(0, 1 - (self.generation_time / 300.0 + self.cost / 1.0) / 2)

        return (
            weights['vmaf'] * self.vmaf_score +
            weights['lip_sync'] * self.lip_sync_score +
            weights['temporal'] * self.temporal_consistency +
            weights['resolution'] * resolution_score +
            weights['efficiency'] * efficiency_score
        )


class QualityRLAgent(nn.Module):
    """Reinforcement learning agent for quality optimization"""

    def __init__(self, state_dim: int = 7, action_dim: int = 8, hidden_dim: int = 128):
        super(QualityRLAgent, self).__init__()

        # Q-network
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Target network (for stable learning)
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Copy initial weights
        self.update_target_network()

        # Action space: parameter adjustments
        self.actions = [
            "increase_fps",      # 0
            "decrease_fps",      # 1
            "increase_quality",  # 2
            "decrease_quality",  # 3
            "enable_interpolation",  # 4
            "disable_interpolation", # 5
            "switch_to_local_gpu",   # 6
            "switch_to_cloud"        # 7
        ]

    def update_target_network(self):
        """Update target network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through Q-network"""
        return self.q_network(state)

    def select_action(self, state: torch.Tensor, epsilon: float = 0.1) -> int:
        """Select action using epsilon-greedy policy"""
        if random.random() < epsilon:
            return random.randint(0, len(self.actions) - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(state.unsqueeze(0))
                return q_values.argmax().item()

    def get_action_name(self, action_idx: int) -> str:
        """Get action name from index"""
        return self.actions[action_idx] if 0 <= action_idx < len(self.actions) else "unknown"


class RLPolicy:
    """Reinforcement learning policy for video generation optimization"""

    def __init__(self, model_path: str = "motion_controller/rl_model.pt"):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(exist_ok=True)

        # RL parameters
        self.gamma = 0.99  # Discount factor
        self.learning_rate = 1e-3
        self.batch_size = 32
        self.memory_size = 10000
        self.target_update_freq = 100

        # Initialize agent
        self.agent = QualityRLAgent()
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate)

        # Experience replay buffer
        self.memory = deque(maxlen=self.memory_size)

        # Training stats
        self.training_steps = 0
        self.episodes_completed = 0

        # Load existing model if available
        self.load_model()

    def optimize_parameters(self, current_state: QualityState,
                          target_quality: float = 0.8) -> Dict[str, Any]:
        """Optimize generation parameters based on current state"""

        # Convert state to tensor
        state_tensor = current_state.to_tensor()

        # Select action
        action_idx = self.agent.select_action(state_tensor, epsilon=0.1)
        action_name = self.agent.get_action_name(action_idx)

        # Generate parameter recommendations based on action
        recommendations = self._action_to_parameters(action_name, current_state, target_quality)

        # Calculate expected improvement
        expected_score = self._predict_quality_improvement(state_tensor, action_idx)

        return {
            "action": action_name,
            "recommendations": recommendations,
            "expected_improvement": expected_score,
            "confidence": self._calculate_confidence(state_tensor, action_idx)
        }

    def _action_to_parameters(self, action: str, current_state: QualityState,
                            target_quality: float) -> Dict[str, Any]:
        """Convert RL action to parameter recommendations"""

        recommendations = {}

        if action == "increase_fps":
            recommendations.update({
                "target_fps": min(30, current_state.generation_time * 1.2),
                "interpolation_method": "rife",
                "expected_quality_impact": 0.05
            })

        elif action == "decrease_fps":
            recommendations.update({
                "target_fps": max(12, current_state.generation_time * 0.8),
                "interpolation_method": "none",
                "expected_quality_impact": -0.05
            })

        elif action == "increase_quality":
            recommendations.update({
                "upscale_enabled": True,
                "denoise_strength": min(1.0, current_state.vmaf_score + 0.1),
                "cinematic_polish": True,
                "expected_quality_impact": 0.15
            })

        elif action == "decrease_quality":
            recommendations.update({
                "upscale_enabled": False,
                "denoise_strength": max(0.0, current_state.vmaf_score - 0.1),
                "cinematic_polish": False,
                "expected_quality_impact": -0.15
            })

        elif action == "enable_interpolation":
            recommendations.update({
                "interpolation_enabled": True,
                "interpolation_method": "rife",
                "target_fps": 24,
                "expected_quality_impact": 0.1
            })

        elif action == "disable_interpolation":
            recommendations.update({
                "interpolation_enabled": False,
                "target_fps": 12,
                "expected_quality_impact": -0.1
            })

        elif action == "switch_to_local_gpu":
            recommendations.update({
                "gpu_device": "local",
                "cost_optimization": True,
                "expected_quality_impact": 0.0  # Neutral
            })

        elif action == "switch_to_cloud":
            recommendations.update({
                "gpu_device": "cloud",
                "high_quality_mode": True,
                "expected_quality_impact": 0.05
            })

        return recommendations

    def _predict_quality_improvement(self, state: torch.Tensor, action: int) -> float:
        """Predict quality improvement from action"""
        with torch.no_grad():
            # Simple prediction based on current Q-values
            q_values = self.agent.q_network(state.unsqueeze(0))
            return q_values[0, action].item() * 0.1  # Scale to reasonable improvement

    def _calculate_confidence(self, state: torch.Tensor, action: int) -> float:
        """Calculate confidence in action recommendation"""
        with torch.no_grad():
            q_values = self.agent.q_network(state.unsqueeze(0))
            max_q = q_values.max().item()
            action_q = q_values[0, action].item()

            # Confidence based on how close action Q is to max Q
            if max_q > 0:
                return action_q / max_q
            else:
                return 0.5  # Neutral confidence

    def learn_from_experience(self, state: QualityState, action: int,
                            reward: float, next_state: QualityState, done: bool):
        """Learn from experience tuple"""

        # Store experience
        self.memory.append((state, action, reward, next_state, done))

        # Train if enough experiences
        if len(self.memory) >= self.batch_size:
            self._train_step()

        self.training_steps += 1

        # Update target network periodically
        if self.training_steps % self.target_update_freq == 0:
            self.agent.update_target_network()

    def _train_step(self):
        """Perform one training step"""

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        state_tensors = torch.stack([s.to_tensor() for s in states])
        action_tensors = torch.tensor(actions, dtype=torch.long)
        reward_tensors = torch.tensor(rewards, dtype=torch.float32)
        next_state_tensors = torch.stack([s.to_tensor() for s in next_states])
        done_tensors = torch.tensor(dones, dtype=torch.float32)

        # Compute Q targets
        with torch.no_grad():
            next_q_values = self.agent.target_network(next_state_tensors)
            max_next_q = next_q_values.max(dim=1)[0]
            targets = reward_tensors + self.gamma * max_next_q * (1 - done_tensors)

        # Compute current Q values
        current_q_values = self.agent.q_network(state_tensors)
        current_q = current_q_values.gather(1, action_tensors.unsqueeze(1)).squeeze()

        # Compute loss
        loss = nn.MSELoss()(current_q, targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self):
        """Save RL model"""
        torch.save({
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'episodes_completed': self.episodes_completed,
            'memory_size': len(self.memory)
        }, self.model_path)

        print(f"RL model saved: {self.model_path}")

    def load_model(self):
        """Load RL model if exists"""
        if self.model_path.exists():
            try:
                checkpoint = torch.load(self.model_path)
                self.agent.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.training_steps = checkpoint.get('training_steps', 0)
                self.episodes_completed = checkpoint.get('episodes_completed', 0)

                print(f"RL model loaded: {self.model_path}")
                print(f"Training steps: {self.training_steps}")

            except Exception as e:
                print(f"Warning: Could not load RL model: {e}")

    def get_policy_stats(self) -> Dict[str, Any]:
        """Get policy statistics"""
        return {
            "training_steps": self.training_steps,
            "episodes_completed": self.episodes_completed,
            "memory_size": len(self.memory),
            "model_path": str(self.model_path),
            "learning_rate": self.learning_rate,
            "gamma": self.gamma
        }


# Global RL policy instance
_rl_policy = None


def get_rl_policy() -> RLPolicy:
    """Get global RL policy instance"""
    global _rl_policy
    if _rl_policy is None:
        _rl_policy = RLPolicy()
    return _rl_policy


def optimize_generation_parameters(current_quality: Dict[str, Any],
                                 target_quality: float = 0.8) -> Dict[str, Any]:
    """Convenience function for parameter optimization"""

    # Convert dict to QualityState
    state = QualityState(
        vmqf_score=current_quality.get('vmaf_score', 0.0),
        lip_sync_score=current_quality.get('lip_sync_score', 0.0),
        temporal_consistency=current_quality.get('temporal_consistency', 0.0),
        resolution=current_quality.get('resolution', (512, 512)),
        generation_time=current_quality.get('generation_time', 0.0),
        cost=current_quality.get('cost', 0.0)
    )

    policy = get_rl_policy()
    return policy.optimize_parameters(state, target_quality)


def quick_test_rl_policy():
    """Quick test of RL policy components"""
    print("Testing RL policy...")

    try:
        policy = get_rl_policy()
        agent = policy.agent

        # Test with dummy state
        dummy_state = QualityState(vmaf_score=0.7, lip_sync_score=0.8)
        state_tensor = dummy_state.to_tensor()

        action = agent.select_action(state_tensor, epsilon=0.0)  # Greedy
        action_name = agent.get_action_name(action)

        print("✅ RL policy components initialized")
        print(f"   Selected action: {action_name}")
        print(f"   Policy stats: {policy.get_policy_stats()}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    quick_test_rl_policy()