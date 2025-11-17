#!/usr/bin/env python3
"""
Story Context Parser - Task 11 Day 1
Analyzes ENTIRE story before video generation to extract character context
Solves: Gender confusion by analyzing ALL sentences (LSTM-like approach)

Created: November 13, 2025
"""

import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter

# Basic stopwords for text condensation
STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'must', 'can', 'of', 'at', 'by', 'for', 'with', 'about',
    'as', 'in', 'on', 'that', 'which', 'who', 'when', 'where', 'why', 'how',
    'very', 'really', 'quite', 'just', 'so', 'too', 'also'
}


@dataclass
class Character:
    """Character information extracted from story"""
    name: str
    gender: str  # 'male', 'female', or 'neutral'
    mentions: List[Tuple[int, str]] = field(default_factory=list)  # (sentence_idx, mention_text)
    pronouns_used: List[str] = field(default_factory=list)
    appearance_keywords: List[str] = field(default_factory=list)
    role: str = "character"  # 'protagonist', 'antagonist', 'character'
    first_mention_idx: int = 0
    confidence: float = 0.0  # Gender detection confidence


@dataclass
class StoryAnalysis:
    """Complete story analysis results"""
    characters: Dict[str, 'Character']
    total_sentences: int
    enhanced_prompts: List[str]  # Enhanced prompts for IMAGE generation
    condensed_narration: List[str]  # Condensed text for AUDIO/SUBTITLES (reduces looping)
    consistency_map: Dict[int, List[str]]  # sentence_idx -> [character_names]


class StoryContextParser:
    """
    Analyze complete story to extract character context BEFORE generation
    
    This solves the gender confusion problem by:
    1. Processing ALL sentences first (like LSTM forward pass)
    2. Extracting character mentions across entire story
    3. Resolving gender from ALL references (not just current sentence)
    4. Building consistent character descriptions
    """
    
    def __init__(self):
        # Gender indicator keywords
        self.male_indicators = {
            'he', 'him', 'his', 'himself', 'man', 'boy', 'male', 
            'guy', 'father', 'brother', 'son', 'king', 'prince',
            'grandfather', 'uncle', 'nephew', 'husband', 'boyfriend'
        }
        
        self.female_indicators = {
            'she', 'her', 'hers', 'herself', 'woman', 'girl', 'female',
            'lady', 'mother', 'sister', 'daughter', 'queen', 'princess',
            'grandmother', 'aunt', 'niece', 'wife', 'girlfriend'
        }
        
        # Character role indicators
        self.protagonist_indicators = {
            'protagonist', 'hero', 'heroine', 'main character', 'seeker',
            'explorer', 'student', 'young', 'warrior'
        }
        
        # Common character reference patterns
        self.character_patterns = [
            r'\b(?:the )?(?:young |old |wise )?(\w+(?:\s+\w+)?)\b',  # "the young seeker"
            r'\b([A-Z][a-z]+)\b',  # Proper names
        ]
    
    def analyze_story(self, sentences: List[str]) -> StoryAnalysis:
        """
        Analyze complete story to extract character information
        
        Args:
            sentences: List of story sentences
            
        Returns:
            StoryAnalysis with characters, enhanced prompts, etc.
        """
        print(f"\n🧠 Analyzing complete story ({len(sentences)} sentences)...")
        
        # Step 1: Extract all character mentions from all sentences
        print("   📝 Step 1: Extracting character mentions...")
        characters = self._extract_characters(sentences)
        print(f"      ✅ Found {len(characters)} characters")
        
        # Step 2: Resolve gender for each character using ALL sentences
        print("   🎭 Step 2: Resolving character genders...")
        for char_name, char in characters.items():
            char.gender = self._resolve_gender(char, sentences)
            char.confidence = self._calculate_confidence(char)
            print(f"      • {char_name}: {char.gender} (confidence: {char.confidence:.2f})")
        
        # Step 3: Detect character roles
        print("   👤 Step 3: Detecting character roles...")
        self._detect_roles(characters, sentences)
        
        # Step 4: Extract appearance keywords
        print("   🎨 Step 4: Extracting appearance keywords...")
        self._extract_appearance_keywords(characters, sentences)
        
        # Step 5: Generate condensed narration for audio/subtitles (reduces looping!)
        print("   📝 Step 5: Condensing sentences for audio/subtitles...")
        condensed_narration = self._condense_for_narration(sentences, characters)
        
        # Step 6: Generate enhanced prompts with character consistency (for images)
        print("   ✨ Step 6: Generating enhanced prompts for image generation...")
        enhanced_prompts = self._generate_enhanced_prompts(sentences, characters)
        
        # Step 7: Build consistency map
        consistency_map = self._build_consistency_map(sentences, characters)
        
        print("   ✅ Story analysis complete!\n")
        
        return StoryAnalysis(
            characters=characters,
            total_sentences=len(sentences),
            enhanced_prompts=enhanced_prompts,
            condensed_narration=condensed_narration,
            consistency_map=consistency_map
        )
    
    def _extract_characters(self, sentences: List[str]) -> Dict[str, Character]:
        """Extract all character mentions from all sentences"""
        characters = {}
        last_mentioned_char = None  # Track last mentioned character for pronoun resolution
        
        for sent_idx, sentence in enumerate(sentences):
            # Look for pronouns first (most reliable gender indicators)
            pronouns = self._extract_pronouns(sentence.lower())
            
            # Look for character references (NOT including pronouns)
            entities = self._extract_entities(sentence)
            
            # If we found actual character entities (not pronouns)
            if entities:
                for entity in entities:
                    entity_lower = entity.lower()
                    
                    # Create character if new
                    if entity_lower not in characters:
                        characters[entity_lower] = Character(
                            name=entity,
                            gender='neutral',
                            first_mention_idx=sent_idx
                        )
                    
                    # Add mention
                    characters[entity_lower].mentions.append((sent_idx, sentence))
                    
                    # Add pronouns found in same sentence to THIS character
                    characters[entity_lower].pronouns_used.extend(pronouns)
                    
                    # Update last mentioned character
                    last_mentioned_char = entity_lower
            
            # If no entities but we have pronouns, link them to last mentioned character
            elif pronouns and last_mentioned_char and last_mentioned_char in characters:
                characters[last_mentioned_char].pronouns_used.extend(pronouns)
                # Also add this sentence as an implicit mention
                characters[last_mentioned_char].mentions.append((sent_idx, sentence))
        
        return characters
    
    def _extract_pronouns(self, text: str) -> List[str]:
        """Extract gender pronouns from text"""
        words = text.split()
        pronouns = []
        
        all_pronouns = self.male_indicators | self.female_indicators
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in all_pronouns:
                pronouns.append(clean_word)
        
        return pronouns
    
    def _extract_entities(self, sentence: str) -> List[str]:
        """Extract potential character entity names (NOT pronouns)"""
        entities = []
        
        # Get all pronouns to exclude them from entities
        all_pronouns = self.male_indicators | self.female_indicators
        
        # Look for common character patterns
        # "a young seeker", "the wise teacher", etc.
        patterns = [
            r'(?:a|an|the)\s+(?:young|old|wise|brave|kind)?\s*(\w+)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',  # Proper names
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, sentence)
            for match in matches:
                entity = match.group(1)
                # Exclude pronouns and short words
                if entity and len(entity) > 2 and entity.lower() not in all_pronouns:
                    entities.append(entity)
        
        # Also look for common character words (but not pronouns)
        character_words = ['seeker', 'teacher', 'student', 'warrior', 'explorer', 
                          'traveler', 'guide', 'master', 'apprentice', 'hero', 'heroine']
        
        for word in character_words:
            if word in sentence.lower() and word not in all_pronouns:
                entities.append(word)
        
        return list(set(entities))  # Remove duplicates
    
    def _resolve_gender(self, character: Character, all_sentences: List[str]) -> str:
        """
        Resolve character gender by analyzing ALL sentences (not just one)
        This is the key solution to gender confusion problem!
        """
        male_score = 0
        female_score = 0
        
        # Count pronouns used with this character
        for pronoun in character.pronouns_used:
            if pronoun in self.male_indicators:
                male_score += 1
            elif pronoun in self.female_indicators:
                female_score += 1
        
        # CRITICAL FIX: Check ALL sentences for pronouns that might refer to this character
        # Look for pronouns in sentences where this character was last mentioned
        last_mention_idx = -1
        if character.mentions:
            last_mention_idx = character.mentions[-1][0]
        
        # Check subsequent sentences for pronouns (they likely refer to this character)
        for sent_idx in range(len(all_sentences)):
            sentence_lower = all_sentences[sent_idx].lower()
            
            # If this sentence is after character's last mention OR contains character name
            if sent_idx >= character.first_mention_idx:
                # Check if character is mentioned in this sentence
                char_mentioned = character.name.lower() in sentence_lower
                
                # Or if it's the next sentence after character mention (pronoun reference)
                is_pronoun_reference = (sent_idx > 0 and 
                                       any(mention_idx == sent_idx - 1 
                                           for mention_idx, _ in character.mentions))
                
                if char_mentioned or is_pronoun_reference:
                    # Count gender indicators in this sentence
                    for indicator in self.male_indicators:
                        if indicator in sentence_lower:
                            male_score += 1.0  # Full weight for contextual indicators
                    
                    for indicator in self.female_indicators:
                        if indicator in sentence_lower:
                            female_score += 1.0  # Full weight for contextual indicators
        
        # Resolve gender
        if female_score > male_score:
            return 'female'
        elif male_score > female_score:
            return 'male'
        else:
            return 'neutral'
    
    def _calculate_confidence(self, character: Character) -> float:
        """Calculate confidence in gender detection"""
        total_indicators = len(character.pronouns_used)
        
        if total_indicators == 0:
            return 0.0
        
        # Count consistent indicators
        male_count = sum(1 for p in character.pronouns_used if p in self.male_indicators)
        female_count = sum(1 for p in character.pronouns_used if p in self.female_indicators)
        
        max_consistent = max(male_count, female_count)
        confidence = max_consistent / total_indicators if total_indicators > 0 else 0.0
        
        return min(confidence, 1.0)
    
    def _detect_roles(self, characters: Dict[str, Character], sentences: List[str]):
        """Detect character roles (protagonist, etc.)"""
        for char_name, char in characters.items():
            # Check if character appears early (likely protagonist)
            if char.first_mention_idx == 0 or char.first_mention_idx == 1:
                # Check for protagonist indicators
                for sent_idx, sentence in char.mentions:
                    sentence_lower = sentence.lower()
                    for indicator in self.protagonist_indicators:
                        if indicator in sentence_lower:
                            char.role = 'protagonist'
                            break
    
    def _extract_appearance_keywords(self, characters: Dict[str, Character], sentences: List[str]):
        """Extract appearance-related keywords for each character"""
        appearance_words = {
            'clothing': ['wearing', 'dressed', 'clothes', 'outfit', 'robe', 'cloak', 'dress', 'suit'],
            'colors': ['red', 'blue', 'green', 'black', 'white', 'yellow', 'purple', 'orange', 'brown', 'gray'],
            'hair': ['hair', 'blonde', 'brunette', 'black-haired', 'long-haired', 'short-haired'],
            'features': ['tall', 'short', 'young', 'old', 'wise', 'strong', 'beautiful', 'handsome']
        }
        
        for char_name, char in characters.items():
            keywords = []
            
            for sent_idx, sentence in char.mentions:
                sentence_lower = sentence.lower()
                
                # Extract appearance keywords
                for category, words in appearance_words.items():
                    for word in words:
                        if word in sentence_lower:
                            keywords.append(word)
            
            char.appearance_keywords = list(set(keywords))  # Remove duplicates
    
    def _condense_for_narration(self, sentences: List[str], characters: Dict[str, Character]) -> List[str]:
        """
        Condense sentences for audio/subtitles to reduce video looping
        
        Strategy:
        1. Remove redundant adjectives (keep one, remove extras)
        2. Simplify wordy phrases using regex patterns
        3. Remove intensifiers (very, extremely, etc.)
        4. Target 20-30% reduction while preserving grammar
        
        Example:
        Original: "A young spiritual seeker embarks on a sacred journey through ancient mystical mountains to find inner peace"
        Condensed: "A young seeker begins a journey through mystical mountains to find peace"
        """
        condensed = []
        
        # Phrase simplification patterns (apply before word-level processing)
        phrase_patterns = [
            (r'\bembarks? on (?:a|an|the) ', 'begins a '),  # "embarks on a" → "begins a"
            (r'\bshares? (?:profound|deep) knowledge with\b', 'teaches'),  # "shares profound knowledge with" → "teaches"
            (r'\bdevelops? (?:deeper|greater) (?:awareness|understanding) of\b', 'understands'),  # simplify
            (r'\bto find inner peace\b', 'for peace'),  # "to find inner peace" → "for peace"
            (r'\bfor many years\b', 'for years'),  # "for many years" → "for years"
            (r'\bin order to\b', 'to'),  # common verbose pattern
        ]
        
        for sentence in sentences:
            # Apply phrase patterns first
            condensed_sentence = sentence
            for pattern, replacement in phrase_patterns:
                condensed_sentence = re.sub(pattern, replacement, condensed_sentence, flags=re.IGNORECASE)
            
            # Now do word-level processing
            words = condensed_sentence.split()
            filtered_words = []
            i = 0
            
            while i < len(words):
                word = words[i]
                word_lower = word.lower().strip('.,!?;:')
                skip = False
                
                # 1. Remove intensifiers
                if word_lower in {'very', 'really', 'quite', 'extremely', 'highly', 'deeply', 'truly', 'absolutely'}:
                    skip = True
                
                # 2. Remove "young" before character names (often redundant)
                elif word_lower == 'young' and i < len(words) - 1:
                    next_word = words[i + 1].lower().strip('.,!?;:')
                    if next_word in {'seeker', 'student', 'disciple', 'monk', 'teacher'}:
                        skip = True
                
                # 3. Remove one of multiple consecutive adjectives
                elif i < len(words) - 1:
                    next_word = words[i + 1].lower().strip('.,!?;:')
                    
                    redundant_adjectives = {'ancient', 'spiritual', 'sacred', 'mystical', 'divine', 
                                          'holy', 'profound', 'deep', 'eternal', 'infinite', 
                                          'magnificent', 'glorious', 'wonderful', 'wise', 'eager'}
                    
                    if word_lower in redundant_adjectives and next_word in redundant_adjectives:
                        skip = True  # Remove first adjective
                
                # Add word to output
                if not skip:
                    filtered_words.append(word)
                
                i += 1
            
            # Reconstruct sentence
            if filtered_words:
                condensed_sentence = ' '.join(filtered_words)
                condensed_sentence = re.sub(r'\s+', ' ', condensed_sentence).strip()
                condensed_sentence = re.sub(r'\s+([.,!?;:])', r'\1', condensed_sentence)
                
                # Ensure proper ending
                if condensed_sentence and condensed_sentence[-1] not in '.!?':
                    condensed_sentence += '.'
                
                condensed.append(condensed_sentence)
                
                # Show reduction
                orig_len = len(sentence)
                cond_len = len(condensed_sentence)
                reduction = ((orig_len - cond_len) / orig_len * 100) if orig_len > 0 else 0
                if reduction > 0:
                    print(f"      ✂️  Reduced {reduction:.0f}%: {sentence[:50]}... → {condensed_sentence[:50]}...")
                else:
                    print(f"      ✓  Kept as-is: {condensed_sentence[:60]}...")
            else:
                condensed.append(sentence)
        
        return condensed
    
    def _generate_enhanced_prompts(self, sentences: List[str], characters: Dict[str, Character]) -> List[str]:
        """
        Generate enhanced prompts with character consistency injected
        
        Example:
        Original: "A young seeker begins her journey"
        Enhanced: "A young female seeker (main character, consistent across story) begins her journey"
        """
        enhanced_prompts = []
        
        for sent_idx, sentence in enumerate(sentences):
            enhanced = sentence
            
            # Find which characters appear in this sentence
            appearing_chars = []
            for char_name, char in characters.items():
                for mention_idx, mention_sentence in char.mentions:
                    if mention_idx == sent_idx:
                        appearing_chars.append(char)
                        break
            
            # Inject character consistency info
            for char in appearing_chars:
                # Build character descriptor
                descriptor_parts = []
                
                # Add gender
                if char.gender != 'neutral':
                    descriptor_parts.append(char.gender)
                
                # Add role
                if char.role != 'character':
                    descriptor_parts.append(char.role)
                
                # Add "consistent character" tag
                if sent_idx > char.first_mention_idx:
                    descriptor_parts.append('same person as scene {}'.format(char.first_mention_idx + 1))
                else:
                    descriptor_parts.append('main character')
                
                # Add appearance keywords (limit to 2)
                if char.appearance_keywords:
                    descriptor_parts.extend(char.appearance_keywords[:2])
                
                descriptor = ', '.join(descriptor_parts)
                
                # Inject into prompt
                # Replace character mentions with enhanced version
                char_pattern = rf'\b{re.escape(char.name)}\b'
                enhanced = re.sub(
                    char_pattern,
                    f"{char.name} ({descriptor})",
                    enhanced,
                    count=1,
                    flags=re.IGNORECASE
                )
            
            enhanced_prompts.append(enhanced)
        
        return enhanced_prompts
    
    def _build_consistency_map(self, sentences: List[str], characters: Dict[str, Character]) -> Dict[int, List[str]]:
        """Build map of which characters appear in which sentences"""
        consistency_map = defaultdict(list)
        
        for char_name, char in characters.items():
            for sent_idx, _ in char.mentions:
                consistency_map[sent_idx].append(char_name)
        
        return dict(consistency_map)
    
    def print_analysis_summary(self, analysis: StoryAnalysis):
        """Print human-readable summary of story analysis"""
        print("\n" + "="*60)
        print("📊 STORY ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\n📚 Total Sentences: {analysis.total_sentences}")
        print(f"👥 Total Characters: {len(analysis.characters)}")
        
        print("\n" + "-"*60)
        print("CHARACTER DETAILS:")
        print("-"*60)
        
        for char_name, char in analysis.characters.items():
            print(f"\n• {char.name}")
            print(f"  Gender: {char.gender} (confidence: {char.confidence:.2f})")
            print(f"  Role: {char.role}")
            print(f"  First mentioned: Scene {char.first_mention_idx + 1}")
            print(f"  Total mentions: {len(char.mentions)}")
            if char.appearance_keywords:
                print(f"  Appearance: {', '.join(char.appearance_keywords)}")
            if char.pronouns_used:
                pronouns = list(set(char.pronouns_used))[:5]
                print(f"  Pronouns used: {', '.join(pronouns)}")
        
        print("\n" + "-"*60)
        print("ENHANCED PROMPTS (Sample):")
        print("-"*60)
        
        for i, (original, enhanced) in enumerate(zip(
            analysis.enhanced_prompts[:3], 
            analysis.enhanced_prompts[:3]
        ), 1):
            print(f"\nScene {i}:")
            print(f"Enhanced: {enhanced[:150]}...")
        
        print("\n" + "="*60 + "\n")


# Singleton instance
_story_context_parser: Optional[StoryContextParser] = None


def get_story_context_parser() -> StoryContextParser:
    """Get singleton story context parser instance"""
    global _story_context_parser
    if _story_context_parser is None:
        _story_context_parser = StoryContextParser()
    return _story_context_parser


# Example usage and testing
if __name__ == "__main__":
    # Test with example story that has gender confusion
    test_story = [
        "A young seeker begins her journey through the ancient mountains.",
        "She walks through misty forests where sages once meditated.",
        "The seeker meets a wise teacher who guides her path.",
        "He teaches the seeker about inner wisdom and meditation.",
        "Through practice, she develops deeper awareness and peace."
    ]
    
    print("🧪 Testing Story Context Parser")
    print("="*60)
    print("\nTest Story:")
    for i, sentence in enumerate(test_story, 1):
        print(f"{i}. {sentence}")
    
    parser = get_story_context_parser()
    analysis = parser.analyze_story(test_story)
    
    parser.print_analysis_summary(analysis)
    
    print("\n✅ Test complete! Gender confusion should be resolved.")
