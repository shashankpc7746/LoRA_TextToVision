#!/usr/bin/env python3
"""Test text condensation for grammar correctness"""

from adaptive_engine import get_story_context_parser

parser = get_story_context_parser()

# Problematic sentences from user feedback
test_story = [
    'She walks through misty forests where ancient sages once meditated for years.',
    'Days pass as she practices mindfulness and develops deeper awareness of her thoughts.',
    'A young spiritual seeker embarks on a sacred journey through ancient mystical mountains to find inner peace.',
    'The wise teacher shares profound knowledge with eager students in the temple.'
]

print("\n" + "="*80)
print("TESTING CONDENSATION - Grammar Check")
print("="*80)

analysis = parser.analyze_story(test_story)

print("\n=== CONDENSED RESULTS ===\n")
for i, (orig, cond) in enumerate(zip(test_story, analysis.condensed_narration), 1):
    print(f"Sentence {i}:")
    print(f"  Original:  {orig}")
    print(f"  Condensed: {cond}")
    reduction = ((len(orig) - len(cond)) / len(orig) * 100) if len(orig) > 0 else 0
    print(f"  Reduction: {reduction:.0f}%")
    print()
