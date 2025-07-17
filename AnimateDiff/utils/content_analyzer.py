#!/usr/bin/env python3
"""
Content-Aware Model Selection for Phase 2 improvements
"""

import re
import numpy as np
from typing import Dict, List, Tuple

class ContentAnalyzer:
    """Analyzes content to select optimal models and parameters"""
    
    def __init__(self):
        # Define content patterns
        self.human_keywords = [
            'person', 'man', 'woman', 'girl', 'boy', 'scientist', 'teacher', 
            'student', 'doctor', 'nurse', 'programmer', 'artist', 'chef',
            'face', 'eyes', 'hands', 'walking', 'running', 'sitting',
            'standing', 'talking', 'smiling', 'looking', 'wearing', 'wizard'
        ]
        
        self.animal_keywords = [
            'eagle', 'bird', 'dog', 'cat', 'horse', 'lion', 'tiger',
            'elephant', 'bear', 'wolf', 'deer', 'rabbit', 'fish',
            'flying', 'swimming', 'hunting', 'feeding', 'nest',
            'wings', 'talons', 'fur', 'feathers', 'paws'
        ]
        
        self.object_keywords = [
            'machine', 'robot', 'car', 'building', 'house', 'tree',
            'mountain', 'ocean', 'sky', 'cloud', 'sun', 'moon',
            'book', 'computer', 'phone', 'table', 'chair'
        ]
        
        self.educational_keywords = [
            'laboratory', 'classroom', 'blackboard', 'experiment',
            'chemical', 'reaction', 'beaker', 'microscope', 'formula',
            'equation', 'geometry', 'mathematics', 'science', 'study'
        ]
        
        self.action_keywords = [
            'moving', 'dynamic', 'fast', 'slow', 'gentle', 'aggressive',
            'precise', 'careful', 'sudden', 'gradual', 'smooth'
        ]
    
    def analyze_content_type(self, prompts: List[str]) -> Dict:
        """Analyze content type from prompts"""
        
        # Combine all prompts for analysis
        combined_text = ' '.join(prompts).lower()
        
        # Count keyword occurrences
        human_score = sum(1 for keyword in self.human_keywords if keyword in combined_text)
        animal_score = sum(1 for keyword in self.animal_keywords if keyword in combined_text)
        object_score = sum(1 for keyword in self.object_keywords if keyword in combined_text)
        educational_score = sum(1 for keyword in self.educational_keywords if keyword in combined_text)
        action_score = sum(1 for keyword in self.action_keywords if keyword in combined_text)
        
        # Determine primary content type
        scores = {
            'human': human_score,
            'animal': animal_score,
            'object': object_score,
            'educational': educational_score,
            'action': action_score
        }
        
        primary_type = max(scores, key=scores.get)
        confidence = scores[primary_type] / max(1, sum(scores.values()))
        
        # Analyze complexity
        complexity = self.analyze_complexity(combined_text)
        
        # Analyze motion requirements
        motion_intensity = self.analyze_motion_intensity(combined_text)
        
        return {
            'primary_type': primary_type,
            'confidence': confidence,
            'scores': scores,
            'complexity': complexity,
            'motion_intensity': motion_intensity,
            'has_humans': human_score > 0,
            'has_animals': animal_score > 0,
            'is_educational': educational_score > 2,
            'original_prompts': prompts  # Include original prompts for style detection
        }
    
    def analyze_complexity(self, text: str) -> str:
        """Analyze scene complexity"""
        
        # Count objects and actions
        object_count = len(re.findall(r'\b(?:a|an|the)\s+\w+', text))
        action_count = len(re.findall(r'\b\w+ing\b', text))
        
        # Count descriptive words
        descriptive_count = len(re.findall(r'\b(?:beautiful|detailed|complex|intricate|simple|clear)\b', text))
        
        total_complexity = object_count + action_count + descriptive_count
        
        if total_complexity > 20:
            return 'high'
        elif total_complexity > 10:
            return 'medium'
        else:
            return 'low'
    
    def analyze_motion_intensity(self, text: str) -> str:
        """Analyze required motion intensity"""
        
        high_motion_words = ['fast', 'quick', 'rapid', 'sudden', 'jumping', 'running', 'flying', 'diving']
        medium_motion_words = ['walking', 'moving', 'turning', 'lifting', 'reaching']
        low_motion_words = ['slow', 'gentle', 'calm', 'peaceful', 'still', 'quiet']
        
        high_count = sum(1 for word in high_motion_words if word in text)
        medium_count = sum(1 for word in medium_motion_words if word in text)
        low_count = sum(1 for word in low_motion_words if word in text)
        
        if high_count > medium_count and high_count > low_count:
            return 'high'
        elif low_count > medium_count and low_count > high_count:
            return 'low'
        else:
            return 'medium'

    def get_content_specific_models(self, content_analysis: Dict) -> Dict:
        """Get content-specific models compatible with AnimateDiff"""

        primary_type = content_analysis.get('primary_type', 'object')
        complexity = content_analysis.get('complexity', 'medium')
        is_educational = content_analysis.get('is_educational', False)

        # PHASE 3: MULTIPLE MODEL SUPPORT - AnimateDiff Compatible Models
        model_configs = {
            'human': {
                'base_model': 'Realistic_Vision_V5.1_noVAE',  # Best for realistic humans
                'motion_adapter': 'animatediff-motion-adapter-v1-5-2',
                'vae': 'stabilityai/sd-vae-ft-mse',
                'lora_models': ['add_detail', 'more_details'],
                'description': 'Optimized for realistic human characters'
            },
            'anime': {
                'base_model': 'anything-v4.0',  # Anime-specific model
                'motion_adapter': 'animatediff-motion-adapter-v1-5-3',
                'vae': 'vae-ft-mse-840000-ema-pruned',
                'lora_models': ['anime_style', 'detailed_anime'],
                'description': 'Optimized for anime and stylized characters'
            },
            'educational': {
                'base_model': 'deliberate_v2',  # Clear, educational content
                'motion_adapter': 'animatediff-motion-adapter-v1-5-2',
                'vae': 'stabilityai/sd-vae-ft-mse',
                'lora_models': ['clarity_enhancement', 'educational_style'],
                'description': 'Optimized for educational and instructional content'
            },
            'animal': {
                'base_model': 'dreamshaper_8',  # Natural scenes and animals
                'motion_adapter': 'animatediff-motion-adapter-v1-5-3',
                'vae': 'vae-ft-mse-840000-ema-pruned',
                'lora_models': ['natural_details', 'wildlife_enhancement'],
                'description': 'Optimized for animals and nature content'
            },
            'mathematics': {
                'base_model': 'deliberate_v2',  # Clear, precise rendering
                'motion_adapter': 'animatediff-motion-adapter-v1-5-2',
                'vae': 'stabilityai/sd-vae-ft-mse',
                'lora_models': ['clarity_enhancement', 'diagram_style'],
                'description': 'Optimized for mathematical diagrams and concepts'
            },
            'object': {
                'base_model': 'Realistic_Vision_V5.1_noVAE',  # General purpose
                'motion_adapter': 'animatediff-motion-adapter-v1-5-2',
                'vae': 'stabilityai/sd-vae-ft-mse',
                'lora_models': ['object_details'],
                'description': 'General purpose for objects and scenes'
            }
        }

        # Smart model selection based on content analysis
        # FORCE ANIME MODEL SELECTION - Always use anime model regardless of content
        # This ensures true anime style output every time
        selected_config = model_configs['anime']
        print(f"🎌 FORCED ANIME MODEL SELECTION - Always using anime model")
        print(f"🎌 Using model: {model_configs['anime']['base_model']} (true anime style)")

        # All other model selection logic removed - always use anime model
        # This ensures consistent anime style output

        print(f"🎨 Selected model: {selected_config['description']}")
        print(f"   • Base model: {selected_config['base_model']}")
        print(f"   • Motion adapter: {selected_config['motion_adapter']}")

        return selected_config

    def select_optimal_config(self, content_analysis: Dict) -> Dict:
        """Select optimal model configuration based on content analysis"""

        # PHASE 3: Get content-specific models
        model_config = self.get_content_specific_models(content_analysis)

        # Base configuration with selected models
        config = {
            'base_model': model_config['base_model'],
            'motion_adapter': model_config['motion_adapter'],
            'vae': model_config['vae'],
            'lora_models': model_config.get('lora_models', []),
            'model_description': model_config['description'],
            'controlnet_weight': 0.8,
            'guidance_scale': 15,
            'num_inference_steps': 25,
            'strength': 0.8
        }
        
        primary_type = content_analysis['primary_type']
        complexity = content_analysis['complexity']
        motion_intensity = content_analysis['motion_intensity']
        
        # Adjust based on content type
        if primary_type == 'human' or content_analysis['has_humans']:
            # Optimize for human consistency
            config.update({
                'controlnet_weight': 0.9,  # Higher control for humans
                'guidance_scale': 18,      # More guidance for face consistency
                'num_inference_steps': 30  # More steps for quality
            })
            
        elif primary_type == 'animal' or content_analysis['has_animals']:
            # Optimize for animal movement
            config.update({
                'controlnet_weight': 0.7,  # Less rigid control for natural movement
                'guidance_scale': 12,      # Less guidance for more natural look
                'motion_adapter': 'animatediff-motion-adapter-v1-5-3'  # Better for animals
            })
            
        elif content_analysis['is_educational']:
            # Optimize for educational content
            config.update({
                'controlnet_weight': 0.8,
                'guidance_scale': 16,
                'num_inference_steps': 28
            })
        
        # Adjust based on complexity
        if complexity == 'high':
            config.update({
                'num_inference_steps': config['num_inference_steps'] + 5,
                'guidance_scale': config['guidance_scale'] + 2
            })
        elif complexity == 'low':
            config.update({
                'num_inference_steps': max(20, config['num_inference_steps'] - 3),
                'guidance_scale': max(10, config['guidance_scale'] - 2)
            })
        
        # Adjust based on motion intensity
        if motion_intensity == 'high':
            config.update({
                'controlnet_weight': max(0.5, config['controlnet_weight'] - 0.1),
                'strength': 0.9
            })
        elif motion_intensity == 'low':
            config.update({
                'controlnet_weight': min(1.0, config['controlnet_weight'] + 0.1),
                'strength': 0.7
            })
        
        return config
    
    def get_consistency_strategy(self, content_analysis: Dict) -> Dict:
        """Get consistency strategy based on content type"""
        
        strategy = {
            'use_character_reference': False,
            'reference_weight': 0.5,
            'pose_consistency_weight': 0.8,
            'color_consistency_weight': 0.6,
            'retry_threshold': 0.4
        }
        
        if content_analysis['has_humans']:
            strategy.update({
                'use_character_reference': True,
                'reference_weight': 0.8,
                'pose_consistency_weight': 0.9,
                'color_consistency_weight': 0.8,
                'retry_threshold': 0.6
            })
        
        elif content_analysis['has_animals']:
            strategy.update({
                'use_character_reference': True,
                'reference_weight': 0.6,
                'pose_consistency_weight': 0.7,
                'color_consistency_weight': 0.9,  # Animals rely more on color consistency
                'retry_threshold': 0.5
            })
        
        return strategy

# Global analyzer instance
content_analyzer = ContentAnalyzer()
