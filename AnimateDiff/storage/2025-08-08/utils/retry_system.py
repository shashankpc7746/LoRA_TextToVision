#!/usr/bin/env python3
"""
Phase 3: Automatic Retry & Recovery System
Intelligent retry system with parameter adjustment and fallback strategies
"""

import os
import time
import random
import shutil
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from pathlib import Path

class RetryStrategy:
    """Defines retry strategies for different failure types"""
    
    def __init__(self):
        self.max_retries = 1  # Reduced from 3 to 1 for faster generation
        self.base_delay = 1.0  # seconds - reduced for faster generation
        
        # Parameter adjustment strategies
        self.parameter_adjustments = {
            'low_quality': {
                'guidance_scale': lambda x: min(25, x + 2),
                'num_inference_steps': lambda x: min(50, x + 5),
                'controlnet_weight': lambda x: min(1.0, x + 0.1)
            },
            'poor_consistency': {
                'controlnet_weight': lambda x: min(1.0, x + 0.2),
                'guidance_scale': lambda x: min(25, x + 3),
                'strength': lambda x: max(0.5, x - 0.1)
            },
            'generation_failure': {
                'guidance_scale': lambda x: max(7, x - 2),
                'num_inference_steps': lambda x: max(15, x - 5),
                'strength': lambda x: min(1.0, x + 0.1)
            },
            'face_inconsistency': {
                'guidance_scale': lambda x: min(25, x + 4),
                'controlnet_weight': lambda x: min(1.0, x + 0.15),
                'num_inference_steps': lambda x: min(40, x + 8)
            }
        }
        
        # Model fallback hierarchy
        self.model_fallbacks = {
            'anything-v4.0': 'Realistic_Vision_V5.1_noVAE',
            'deliberate_v2': 'Realistic_Vision_V5.1_noVAE',
            'dreamshaper_8': 'Realistic_Vision_V5.1_noVAE',
            'Realistic_Vision_V5.1_noVAE': 'runwayml/stable-diffusion-v1-5'  # Ultimate fallback
        }
    
    def should_retry(self, failure_info: Dict, attempt: int) -> bool:
        """Determine if we should retry based on failure info and attempt count"""
        
        if attempt >= self.max_retries:
            return False
        
        failure_type = failure_info.get('type', 'unknown')
        severity = failure_info.get('severity', 'medium')
        
        # Don't retry critical system failures
        if severity == 'critical':
            return False
        
        # Always retry quality issues (they're often fixable)
        if failure_type in ['low_quality', 'poor_consistency', 'face_inconsistency']:
            return True
        
        # Retry generation failures with decreasing probability
        if failure_type == 'generation_failure':
            retry_probability = 1.0 - (attempt * 0.3)
            return random.random() < retry_probability
        
        return attempt < 2  # Default: retry once
    
    def adjust_parameters(self, config: Dict, failure_info: Dict) -> Dict:
        """Adjust generation parameters based on failure type"""
        
        failure_type = failure_info.get('type', 'unknown')
        adjusted_config = config.copy()
        
        if failure_type in self.parameter_adjustments:
            adjustments = self.parameter_adjustments[failure_type]
            
            for param, adjustment_func in adjustments.items():
                if param in adjusted_config:
                    old_value = adjusted_config[param]
                    new_value = adjustment_func(old_value)
                    adjusted_config[param] = new_value
                    print(f"   📊 Adjusted {param}: {old_value} → {new_value}")
        
        # Add randomization to seed
        if 'seed' in adjusted_config:
            adjusted_config['seed'] = random.randint(1000, 999999)
            print(f"   🎲 New random seed: {adjusted_config['seed']}")
        
        return adjusted_config
    
    def get_fallback_model(self, current_model: str) -> Optional[str]:
        """Get fallback model for current model"""
        
        return self.model_fallbacks.get(current_model)
    
    def calculate_retry_delay(self, attempt: int) -> float:
        """Calculate delay before retry (exponential backoff)"""
        
        delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
        return min(delay, 30.0)  # Cap at 30 seconds

class AutoRetrySystem:
    """Automatic retry and recovery system for video generation"""
    
    def __init__(self):
        self.strategy = RetryStrategy()
        self.retry_history = []
        self.success_rate = 1.0
        
    def execute_with_retry(self, 
                          generation_func: Callable,
                          config: Dict,
                          quality_checker: Callable,
                          identity_checker: Optional[Callable] = None) -> Dict:
        """Execute generation with automatic retry on failure"""
        
        attempt = 0
        last_failure = None
        
        while attempt <= self.strategy.max_retries:
            try:
                print(f"\n🔄 Generation attempt {attempt + 1}/{self.strategy.max_retries + 1}")
                
                # Execute generation
                start_time = time.time()
                result = generation_func(config)
                generation_time = time.time() - start_time
                
                if not result or not result.get('success', False):
                    failure_info = {
                        'type': 'generation_failure',
                        'severity': 'medium',
                        'message': result.get('error', 'Unknown generation failure'),
                        'attempt': attempt
                    }
                    last_failure = failure_info
                    
                else:
                    # Check quality
                    quality_result = quality_checker(result['output_path'])
                    
                    # Check identity consistency if available
                    identity_result = None
                    if identity_checker:
                        identity_result = identity_checker(result['output_path'])
                    
                    # Determine if retry is needed
                    failure_info = self._analyze_results(quality_result, identity_result)
                    
                    if failure_info is None:
                        # Success!
                        self._record_success(attempt, generation_time)
                        return {
                            'success': True,
                            'result': result,
                            'attempts': attempt + 1,
                            'quality': quality_result,
                            'identity': identity_result,
                            'generation_time': generation_time
                        }
                    else:
                        last_failure = failure_info
                
                # Check if we should retry
                if not self.strategy.should_retry(last_failure, attempt):
                    break
                
                # Adjust parameters for retry
                config = self.strategy.adjust_parameters(config, last_failure)
                
                # Try fallback model if needed
                if last_failure['type'] == 'generation_failure' and attempt > 0:
                    fallback_model = self.strategy.get_fallback_model(config.get('base_model', ''))
                    if fallback_model:
                        config['base_model'] = fallback_model
                        print(f"   🔄 Falling back to model: {fallback_model}")
                
                # Wait before retry
                delay = self.strategy.calculate_retry_delay(attempt)
                print(f"   ⏳ Waiting {delay:.1f}s before retry...")
                time.sleep(delay)
                
                attempt += 1
                
            except Exception as e:
                failure_info = {
                    'type': 'generation_failure',
                    'severity': 'critical',
                    'message': str(e),
                    'attempt': attempt
                }
                last_failure = failure_info
                
                if not self.strategy.should_retry(failure_info, attempt):
                    break
                
                attempt += 1
        
        # All retries failed
        self._record_failure(attempt, last_failure)
        return {
            'success': False,
            'failure_info': last_failure,
            'attempts': attempt,
            'message': f"Generation failed after {attempt} attempts: {last_failure['message']}"
        }
    
    def _analyze_results(self, quality_result: Dict, identity_result: Optional[Dict]) -> Optional[Dict]:
        """Analyze results to determine if retry is needed"""

        # STORY PRESERVATION: Check if story should be preserved
        if quality_result.get('story_preserved', False):
            print("📖 Story preservation mode - accepting clip to maintain continuity")
            return None  # No retry needed

        # Check quality issues
        if quality_result.get('should_retry', False):
            return {
                'type': 'low_quality',
                'severity': 'medium',
                'message': f"Quality score {quality_result['overall_score']:.3f} below threshold",
                'issues': quality_result.get('issues', [])
            }
        
        # Check identity consistency issues
        if identity_result and not identity_result.get('is_consistent', True):
            similarity = identity_result.get('similarity', 0.0)
            if similarity < 0.4:  # Very poor identity match
                return {
                    'type': 'face_inconsistency',
                    'severity': 'high',
                    'message': f"Face similarity {similarity:.3f} too low",
                    'similarity': similarity
                }
        
        # Check for poor consistency (multiple quality issues)
        if quality_result.get('individual_scores', {}):
            scores = quality_result['individual_scores']
            poor_scores = [k for k, v in scores.items() if v < 0.3]
            
            if len(poor_scores) >= 2:
                return {
                    'type': 'poor_consistency',
                    'severity': 'medium',
                    'message': f"Multiple poor scores: {', '.join(poor_scores)}",
                    'poor_metrics': poor_scores
                }
        
        return None  # No retry needed
    
    def _record_success(self, attempts: int, generation_time: float):
        """Record successful generation"""
        
        self.retry_history.append({
            'success': True,
            'attempts': attempts + 1,
            'generation_time': generation_time,
            'timestamp': time.time()
        })
        
        # Update success rate
        recent_history = self.retry_history[-20:]  # Last 20 generations
        successes = sum(1 for h in recent_history if h['success'])
        self.success_rate = successes / len(recent_history)
        
        print(f"✅ Generation successful after {attempts + 1} attempt(s)")
        print(f"📊 Current success rate: {self.success_rate:.1%}")
    
    def _record_failure(self, attempts: int, failure_info: Dict):
        """Record failed generation"""
        
        self.retry_history.append({
            'success': False,
            'attempts': attempts,
            'failure_info': failure_info,
            'timestamp': time.time()
        })
        
        # Update success rate
        recent_history = self.retry_history[-20:]
        successes = sum(1 for h in recent_history if h['success'])
        self.success_rate = successes / len(recent_history) if recent_history else 0.0
        
        print(f"❌ Generation failed after {attempts} attempt(s)")
        print(f"📊 Current success rate: {self.success_rate:.1%}")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        
        if not self.retry_history:
            return {'no_data': True}
        
        recent_history = self.retry_history[-50:]  # Last 50 generations
        
        total_attempts = sum(h['attempts'] for h in recent_history)
        successful_gens = [h for h in recent_history if h['success']]
        
        avg_attempts = total_attempts / len(recent_history)
        avg_generation_time = np.mean([h['generation_time'] for h in successful_gens]) if successful_gens else 0
        
        return {
            'success_rate': self.success_rate,
            'avg_attempts_per_generation': avg_attempts,
            'avg_generation_time': avg_generation_time,
            'total_generations': len(recent_history),
            'successful_generations': len(successful_gens)
        }

# Global retry system instance
auto_retry_system = AutoRetrySystem()
