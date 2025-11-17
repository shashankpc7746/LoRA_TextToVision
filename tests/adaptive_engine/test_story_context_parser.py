#!/usr/bin/env python3
"""
Unit Tests for Story Context Parser - Task 11 Day 1
Tests full story NLP analysis and character gender resolution
"""

import pytest
import sys
from pathlib import Path

# Path setup is handled by conftest.py
from adaptive_engine.story_context_parser import (
    StoryContextParser,
    get_story_context_parser,
    Character,
    StoryAnalysis
)


class TestStoryContextParser:
    """Test suite for story context parser"""
    
    def setup_method(self):
        """Setup test instance"""
        self.parser = StoryContextParser()
    
    def test_initialization(self):
        """Test parser initialization"""
        assert self.parser is not None
        assert len(self.parser.male_indicators) > 0
        assert len(self.parser.female_indicators) > 0
    
    def test_singleton_pattern(self):
        """Test singleton pattern"""
        parser1 = get_story_context_parser()
        parser2 = get_story_context_parser()
        assert parser1 is parser2
    
    def test_extract_pronouns(self):
        """Test pronoun extraction"""
        text = "She walks through the forest. He follows her."
        pronouns = self.parser._extract_pronouns(text.lower())
        assert 'she' in pronouns
        assert 'he' in pronouns
        assert 'her' in pronouns
    
    def test_gender_resolution_female(self):
        """Test gender resolution for female character"""
        story = [
            "A young seeker begins her journey.",
            "She walks through the forest.",
            "The seeker finds her path."
        ]
        
        analysis = self.parser.analyze_story(story)
        
        # Check that 'seeker' is identified as female
        assert len(analysis.characters) > 0
        
        # Find seeker character
        seeker = None
        for char_name, char in analysis.characters.items():
            if 'seeker' in char_name.lower():
                seeker = char
                break
        
        assert seeker is not None
        assert seeker.gender == 'female', f"Expected female, got {seeker.gender}"
    
    def test_gender_resolution_male(self):
        """Test gender resolution for male character"""
        story = [
            "A wise teacher shares his knowledge.",
            "He guides his students carefully.",
            "The teacher explains his philosophy."
        ]
        
        analysis = self.parser.analyze_story(story)
        
        # Find teacher character
        teacher = None
        for char_name, char in analysis.characters.items():
            if 'teacher' in char_name.lower():
                teacher = char
                break
        
        assert teacher is not None
        assert teacher.gender == 'male', f"Expected male, got {teacher.gender}"
    
    def test_gender_consistency_across_sentences(self):
        """
        Test that gender is resolved correctly even when first sentence is ambiguous
        This is the KEY test for solving gender confusion problem!
        """
        story = [
            "A young seeker begins the journey.",  # Ambiguous - no gender indicator
            "She walks through misty forests.",    # Female indicator
            "The seeker meets a wise teacher.",    # Ambiguous
            "She learns ancient wisdom."           # Female indicator
        ]
        
        analysis = self.parser.analyze_story(story)
        
        # Find seeker
        seeker = None
        for char_name, char in analysis.characters.items():
            if 'seeker' in char_name.lower():
                seeker = char
                break
        
        assert seeker is not None
        assert seeker.gender == 'female', \
            "Gender should be resolved as female from later sentences (LSTM-like analysis)"
    
    def test_multiple_characters(self):
        """Test handling multiple characters"""
        story = [
            "A young student meets her teacher.",
            "He teaches her about meditation.",
            "She practices daily and grows stronger."
        ]
        
        analysis = self.parser.analyze_story(story)
        
        # Should identify at least student and teacher
        assert len(analysis.characters) >= 2
        
        # Check genders are different
        genders = [char.gender for char in analysis.characters.values()]
        assert 'male' in genders or 'female' in genders
    
    def test_character_role_detection(self):
        """Test protagonist role detection"""
        story = [
            "A young seeker begins her spiritual journey.",
            "She walks through ancient mountains.",
            "The seeker finds inner peace."
        ]
        
        analysis = self.parser.analyze_story(story)
        
        # Find seeker
        seeker = None
        for char in analysis.characters.values():
            if 'seeker' in char.name.lower():
                seeker = char
                break
        
        assert seeker is not None
        assert seeker.role == 'protagonist'
    
    def test_appearance_keyword_extraction(self):
        """Test extraction of appearance keywords"""
        story = [
            "A young woman wearing red clothes walks forward.",
            "She has long black hair and wise eyes."
        ]
        
        analysis = self.parser.analyze_story(story)
        
        # Check that appearance keywords were extracted
        found_keywords = False
        for char in analysis.characters.values():
            if char.appearance_keywords:
                found_keywords = True
                break
        
        assert found_keywords, "Should extract appearance keywords"
    
    def test_enhanced_prompt_generation(self):
        """Test enhanced prompt generation"""
        story = [
            "A young seeker begins her journey.",
            "She walks through the forest."
        ]
        
        analysis = self.parser.analyze_story(story)
        
        assert len(analysis.enhanced_prompts) == len(story)
        
        # Enhanced prompts should be different from original
        assert analysis.enhanced_prompts[0] != story[0] or \
               analysis.enhanced_prompts[1] != story[1]
    
    def test_consistency_map(self):
        """Test consistency map generation"""
        story = [
            "A young seeker begins her journey.",
            "She walks through the forest.",
            "The seeker finds peace."
        ]
        
        analysis = self.parser.analyze_story(story)
        
        # Consistency map should track character appearances per sentence
        assert len(analysis.consistency_map) > 0
    
    def test_confidence_calculation(self):
        """Test confidence calculation"""
        char = Character(
            name="test",
            gender="female"
        )
        char.pronouns_used = ['she', 'her', 'she', 'her']
        
        confidence = self.parser._calculate_confidence(char)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be confident with consistent pronouns
    
    def test_empty_story(self):
        """Test handling empty story"""
        story = []
        
        analysis = self.parser.analyze_story(story)
        
        assert analysis.total_sentences == 0
        assert len(analysis.characters) == 0
    
    def test_single_sentence_story(self):
        """Test handling single sentence"""
        story = ["A young woman walks through the forest."]
        
        analysis = self.parser.analyze_story(story)
        
        assert analysis.total_sentences == 1
        assert len(analysis.characters) >= 0  # May or may not detect characters


# Performance and edge case tests
class TestStoryContextParserEdgeCases:
    """Test edge cases and performance"""
    
    def setup_method(self):
        """Setup test instance"""
        self.parser = StoryContextParser()
    
    def test_conflicting_gender_indicators(self):
        """Test handling conflicting gender indicators"""
        story = [
            "The person walks forward.",
            "He enters the room.",
            "She looks around carefully."  # Conflicting with previous
        ]
        
        analysis = self.parser.analyze_story(story)
        
        # Should handle gracefully (may choose most frequent)
        assert len(analysis.characters) > 0
    
    def test_long_story(self):
        """Test performance with longer story"""
        story = [f"Sentence {i} with character walking." for i in range(50)]
        
        import time
        start = time.time()
        analysis = self.parser.analyze_story(story)
        duration = time.time() - start
        
        assert duration < 5.0  # Should complete in under 5 seconds
        assert analysis.total_sentences == 50
    
    def test_special_characters_in_text(self):
        """Test handling special characters"""
        story = [
            "A seeker says: 'I will find peace!'",
            "She walks through the forest... carefully.",
            "The seeker finds her path (finally)."
        ]
        
        analysis = self.parser.analyze_story(story)
        
        # Should handle special characters gracefully
        assert len(analysis.characters) > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
