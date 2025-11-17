# Task 11: Phase III - TTV Studio Core (REVISED PLAN)
## Addressing Real Production Issues

**Created**: November 13, 2025  
**Status**: 🎯 PROBLEM-FOCUSED APPROACH  
**Duration**: 7 Days  
**Purpose**: Solve actual video generation problems + Complete Phase 2 feedback goals

---

## 🔴 **ACTUAL PROBLEMS WE'RE SOLVING**

### Problem 1: Video Looping Issue ❌
**Current Behavior**:
```
Clip 1: 2 sec actual content → Looped 2-3 times to match 5 sec audio → Looks repetitive
Clip 2: 3 sec actual content → Looped 2 times to match 6 sec audio → Looks repetitive
...
```

**Code Location**: `AnimateDiff/unified_video_generator.py:452`
```python
extended_clip = loop(video_clip, duration=audio_duration)  # ❌ BAD: Causes repetitive visuals
```

**Root Cause**: AnimateDiff generates only 16-32 frames (~2 seconds), but audio is 5-10 seconds

**Our Solution (Day 5 + RIFE Enhancement)**:
- **Intelligent Frame Interpolation**: Use RIFE properly with scene-aware interpolation
- **Content Extension**: Generate additional frames based on prompt continuation
- **Smart Transitions**: Blend between clips instead of hard loops
- **Variable FPS**: Slow down motion naturally (24fps → 16fps) to extend duration

---

### Problem 2: Character Consistency (Gender Confusion) ❌
**Current Behavior**:
```
Sentence 1: "A young seeker begins her journey" → Model sees "seeker" → Assumes MALE
Sentence 2: "She walks through misty forest" → Model sees "She" → Switches to FEMALE
Result: Character changes gender mid-video! ❌
```

**Root Cause**: Per-clip prompt processing without forward context

**Our Solution (Day 3: Narrative Sequencer with NLP)**:
- **Full Story Analysis**: Parse entire story BEFORE generation (LSTM/GRU-like approach)
- **Character Identification**: Extract all character references across all sentences
- **Gender Resolution**: Resolve gender from ALL mentions (she/her = female, he/him = male)
- **Consistent Prompts**: Inject resolved character info into EVERY clip prompt

**NLP Implementation**:
```python
# Day 3: narrative_sequencer_v1.py
class NarrativeSequencer:
    def analyze_full_story(self, all_sentences: List[str]) -> Dict:
        """
        Analyze complete story using NLP (similar to LSTM forward pass)
        - Extract all character mentions
        - Resolve gender/appearance from context
        - Build character continuity graph
        """
        # Step 1: Extract entities (spaCy NER)
        characters = self.extract_characters(all_sentences)
        
        # Step 2: Gender resolution (look ahead + look back)
        for char in characters:
            char.gender = self.resolve_gender_from_context(char, all_sentences)
        
        # Step 3: Generate consistent prompts
        enhanced_prompts = self.inject_character_consistency(all_sentences, characters)
        
        return enhanced_prompts
```

---

### Problem 3: Model Can't Generate Educational Diagrams ❌
**Current Issue**: RealisticVision model generates faces well but fails at:
- Mathematical diagrams (geometry shapes, formulas)
- Educational structures (charts, graphs, animations)
- Technical content (blueprints, schematics)

**Our Solution (Day 1: LoRA Training + Hybrid Approach)**:
- **Educational LoRA**: Train LoRA on 500 educational diagram images
- **Hybrid Generation**: 
  - Faces/Characters → RealisticVision base model
  - Diagrams/Educational → RealisticVision + Educational LoRA
- **Smart Detection**: Auto-detect if scene needs diagrams or faces
- **Fallback**: Use stable-diffusion base model for pure educational content

---

### Problem 4: RIFE Black Screen Issue ❌
**Current Behavior**:
```
First half: Actual video content (2-3 seconds)
Second half: BLACK SCREEN while audio continues (3-5 seconds)
```

**Root Cause**: RIFE expects continuous frames, but we're asking it to fill gaps it can't handle

**Our Solution (Day 5: Proper RIFE Integration)**:
- **Scene Continuity**: Extract last meaningful frame + predict next frames
- **Motion Vectors**: Use optical flow to guide RIFE interpolation
- **Freeze Detection**: Detect if RIFE generates static frames → Switch to slow-mo instead
- **Hybrid Approach**: RIFE for motion + Frame freezing for static scenes

---

## 🎯 **REVISED 7-DAY PLAN (Problem-Focused)**

### **Day 1: Smart LoRA Training + Character Context Parser**
**Files**: 
- `AnimateDiff/adaptive_engine/identity_memory.py` (~400 lines)
- `AnimateDiff/adaptive_engine/educational_lora_trainer.py` (~300 lines)

**Problem Addressed**: Character consistency + Educational diagram generation

**What We Build**:

1. **Character Identity Memory** (identity_memory.py):
```python
class IdentityMemory:
    """Track character identity across entire story"""
    
    def analyze_story_context(self, all_sentences: List[str]) -> Dict[str, Character]:
        """
        FULL STORY ANALYSIS (Like LSTM - processes all sentences)
        - Extract all character references ("young seeker", "she", "her")
        - Resolve gender from ALL mentions (not just current sentence)
        - Build appearance profile from descriptive words
        """
        characters = {}
        
        # Step 1: Extract entities using NLP
        for sent_idx, sentence in enumerate(all_sentences):
            entities = self.extract_entities(sentence)
            
            # Look ahead to next sentences for gender clues
            forward_context = all_sentences[sent_idx:sent_idx+3]
            
            # Look back to previous sentences for consistency
            backward_context = all_sentences[max(0, sent_idx-3):sent_idx]
            
            for entity in entities:
                char_id = self.resolve_character_id(entity, forward_context, backward_context)
                
                if char_id not in characters:
                    characters[char_id] = Character(
                        name=entity,
                        gender=self.resolve_gender(entity, all_sentences),  # ALL sentences
                        appearance_keywords=self.extract_appearance(entity, all_sentences),
                        first_mention=sent_idx
                    )
        
        return characters
    
    def resolve_gender(self, entity: str, all_sentences: List[str]) -> str:
        """
        Resolve gender by analyzing ALL sentences (not just current)
        Similar to LSTM: processes sequence to understand context
        """
        male_score = 0
        female_score = 0
        
        for sentence in all_sentences:
            if entity.lower() in sentence.lower():
                # Count gender indicators in this sentence
                male_score += self.count_keywords(sentence, ["he", "him", "his", "man", "boy"])
                female_score += self.count_keywords(sentence, ["she", "her", "hers", "woman", "girl"])
        
        return "female" if female_score > male_score else "male"
```

2. **Educational LoRA Trainer** (educational_lora_trainer.py):
```python
class EducationalLoRATrainer:
    """Train LoRA on educational diagrams for better generation"""
    
    def __init__(self):
        self.educational_images_dir = "datasets/educational_diagrams/"
        self.lora_output_dir = "adapters/educational_lora/"
    
    def detect_content_type(self, prompt: str) -> str:
        """Detect if prompt needs diagrams or faces"""
        educational_keywords = [
            "geometry", "diagram", "chart", "graph", "formula", "equation",
            "triangle", "circle", "square", "blueprint", "schematic", "animation"
        ]
        
        face_keywords = ["person", "man", "woman", "face", "character", "seeker"]
        
        edu_score = sum(1 for kw in educational_keywords if kw in prompt.lower())
        face_score = sum(1 for kw in face_keywords if kw in prompt.lower())
        
        if edu_score > face_score:
            return "educational"  # Use LoRA
        return "character"  # Use base model
    
    def train_educational_lora(self, images_dir: str, num_epochs: int = 10):
        """Train LoRA on 500 educational diagram images"""
        # Prepare dataset
        image_paths = self.prepare_educational_dataset(images_dir)
        
        # Train LoRA (actual training code)
        lora_path = self.train_lora_model(
            base_model="SG161222/Realistic_Vision_V5.1_noVAE",
            training_images=image_paths,
            output_dir=self.lora_output_dir,
            concepts=["educational diagram", "geometry shapes", "mathematical illustration"]
        )
        
        return lora_path
```

**Deliverables**:
- [ ] `identity_memory.py` with full story context analysis
- [ ] `educational_lora_trainer.py` with LoRA training
- [ ] Character consistency across all clips
- [ ] Educational LoRA model trained on 500 images
- [ ] Unit tests

---

### **Day 2: Scene Graph + Story Understanding**
**File**: `AnimateDiff/adaptive_engine/scene_memory_core.py` (~500 lines)

**Problem Addressed**: Track characters, objects, and relationships across scenes

**What We Build**:
```python
class SceneMemoryCore:
    """Scene graph with temporal character/object tracking"""
    
    def build_scene_graph(self, story_sentences: List[str], characters: Dict) -> nx.DiGraph:
        """
        Build scene graph from story
        Nodes: Characters, Objects, Locations
        Edges: Relationships (who appears where, when)
        """
        graph = nx.DiGraph()
        
        for idx, sentence in enumerate(story_sentences):
            # Add scene node
            scene_id = f"scene_{idx}"
            graph.add_node(scene_id, type="scene", text=sentence, timestamp=idx)
            
            # Link characters to scene
            for char_id, character in characters.items():
                if character.appears_in_sentence(sentence):
                    graph.add_edge(char_id, scene_id, relationship="appears_in")
            
            # Temporal linking (scene A → scene B)
            if idx > 0:
                prev_scene = f"scene_{idx-1}"
                graph.add_edge(prev_scene, scene_id, relationship="temporal_next")
        
        return graph
```

**🎯 PHASE 2 GOAL #1: Scene Graph Module - COMPLETED HERE**

---

### **Day 3: Narrative Sequencer with NLP (LSTM-like Context)**
**File**: `AnimateDiff/adaptive_engine/narrative_sequencer_v1.py` (~500 lines)

**Problem Addressed**: Character gender confusion + Lack of forward context

**What We Build**:
```python
class NarrativeSequencer:
    """
    Story understanding engine with LSTM-like sequential processing
    Processes entire story to understand context before generation
    """
    
    def __init__(self):
        # Load NLP model (spaCy for entity recognition)
        import spacy
        self.nlp = spacy.load("en_core_web_sm")
    
    def analyze_narrative(self, story: List[str]) -> NarrativeAnalysis:
        """
        Full story analysis (similar to LSTM forward pass)
        - Processes all sentences to build context
        - Resolves character identities
        - Detects story beats (setup, conflict, resolution)
        """
        
        # Step 1: Entity extraction across ALL sentences
        all_entities = []
        for sent in story:
            doc = self.nlp(sent)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            all_entities.extend(entities)
        
        # Step 2: Character resolution with forward+backward context
        characters = self.resolve_characters_globally(story, all_entities)
        
        # Step 3: Story beat detection
        beats = self.detect_story_beats(story, characters)
        
        # Step 4: Generate enhanced prompts with character consistency
        enhanced_prompts = self.generate_consistent_prompts(story, characters)
        
        return NarrativeAnalysis(
            characters=characters,
            story_beats=beats,
            enhanced_prompts=enhanced_prompts
        )
    
    def resolve_characters_globally(self, story: List[str], entities: List) -> Dict:
        """
        Resolve character identities from ENTIRE story (not per-sentence)
        This is the LSTM-like approach you mentioned!
        """
        characters = {}
        
        # Build character co-reference resolution
        for i, sentence in enumerate(story):
            # Look at surrounding sentences for context
            context_window = story[max(0, i-2):min(len(story), i+3)]
            
            # Extract pronouns and link to characters
            pronouns = self.extract_pronouns(sentence)
            for pronoun in pronouns:
                # Resolve pronoun to actual character using context
                char = self.resolve_pronoun(pronoun, context_window, characters)
                if char:
                    characters[char.id].mentions.append((i, pronoun))
        
        return characters
    
    def generate_consistent_prompts(self, story: List[str], characters: Dict) -> List[str]:
        """
        Generate prompts with character consistency
        Example:
        Original: "A young seeker begins her journey"
        Enhanced: "A young female seeker (consistent character, brown hair, determined expression) begins her journey"
        """
        enhanced = []
        
        for sent_idx, sentence in enumerate(story):
            # Find which characters appear in this sentence
            appearing_chars = [c for c in characters.values() if c.appears_in(sent_idx)]
            
            # Inject character consistency info
            enhanced_sentence = sentence
            for char in appearing_chars:
                # Add character details from global resolution
                char_desc = f"{char.gender} {char.role}, same person as scene {char.first_mention}"
                enhanced_sentence = enhanced_sentence.replace(char.name, f"{char.name} ({char_desc})")
            
            enhanced.append(enhanced_sentence)
        
        return enhanced
```

**Example Output**:
```
Input Story:
1. "A young seeker begins her journey"
2. "She walks through misty forest"
3. "The seeker finds ancient wisdom"

After NLP Analysis:
1. "A young female seeker (main character, first appearance) begins her journey"
2. "She (same female seeker from scene 1) walks through misty forest"
3. "The female seeker (consistent character across all scenes) finds ancient wisdom"
```

**🎯 PHASE 2 GOAL #4: Narrative Engine - COMPLETED HERE**

---

### **Day 4: Emotion Controller + Character Arc Tracking**
**File**: `AnimateDiff/adaptive_engine/emotion_controller.py` (~400 lines)

**Problem Addressed**: Emotional continuity across clips

**What We Build**:
- Track emotion state per character
- Smooth emotion transitions (joy → surprise, not joy → anger suddenly)
- Sync emotions with narrative beats

---

### **Day 5: Intelligent Frame Extension (Fix Looping + RIFE)**
**File**: `AnimateDiff/adaptive_engine/intelligent_frame_extender.py` (~450 lines)

**Problem Addressed**: Video looping + RIFE black screen issues

**What We Build**:
```python
class IntelligentFrameExtender:
    """
    Smart video extension without repetitive looping
    Fixes: Looping issue + RIFE black screen problem
    """
    
    def extend_clip_to_audio(self, video_path: str, target_duration: float) -> str:
        """
        Extend video to match audio WITHOUT obvious looping
        
        Strategies:
        1. RIFE Interpolation (for motion scenes)
        2. Slow Motion (reduce FPS 24→16)
        3. Smart Freeze (for static scenes)
        4. Optical Flow Extension (predict next frames)
        """
        
        # Load video
        clip = VideoFileClip(video_path)
        current_duration = clip.duration
        
        if current_duration >= target_duration:
            return video_path  # Already long enough
        
        # Analyze motion in video
        motion_score = self.analyze_motion(video_path)
        
        if motion_score > 0.5:
            # High motion → Use RIFE interpolation
            extended_clip = self.extend_with_rife(clip, target_duration)
        elif motion_score > 0.2:
            # Medium motion → Use slow motion
            extended_clip = self.extend_with_slowmo(clip, target_duration)
        else:
            # Static scene → Use smart freeze
            extended_clip = self.extend_with_freeze(clip, target_duration)
        
        return extended_clip
    
    def extend_with_rife(self, clip: VideoClip, target_duration: float) -> VideoClip:
        """
        Use RIFE properly with motion continuity
        Fixes black screen issue by ensuring frame continuity
        """
        from interpolator.rife_interpolator import RIFEInterpolator
        
        rife = RIFEInterpolator()
        
        # Extract frames
        frames = list(clip.iter_frames(fps=24))
        
        # Calculate needed frame count
        current_frames = len(frames)
        needed_frames = int(target_duration * 24)
        
        if needed_frames <= current_frames:
            return clip
        
        # IMPORTANT: Use last frame + optical flow to predict continuation
        last_frame = frames[-1]
        second_last_frame = frames[-2] if len(frames) > 1 else last_frame
        
        # Calculate motion vector
        motion_vector = self.calculate_optical_flow(second_last_frame, last_frame)
        
        # Generate continuation frames using RIFE + motion guidance
        extended_frames = frames.copy()
        
        for i in range(needed_frames - current_frames):
            # Predict next frame using RIFE
            next_frame = rife.interpolate_with_motion_guidance(
                extended_frames[-1],
                motion_vector,
                timestep=0.5
            )
            
            # Check if frame is valid (not black)
            if self.is_valid_frame(next_frame):
                extended_frames.append(next_frame)
            else:
                # Fallback: freeze last good frame
                extended_frames.append(extended_frames[-1])
        
        # Create video from extended frames
        extended_clip = ImageSequenceClip(extended_frames, fps=24)
        return extended_clip
    
    def extend_with_slowmo(self, clip: VideoClip, target_duration: float) -> VideoClip:
        """
        Extend by slowing down motion (24fps → 16fps)
        More natural than looping
        """
        slowdown_factor = target_duration / clip.duration
        return clip.fx(vfx.speedx, slowdown_factor)
    
    def extend_with_freeze(self, clip: VideoClip, target_duration: float) -> VideoClip:
        """
        For static scenes, freeze last frame elegantly
        Better than looping which looks repetitive
        """
        freeze_duration = target_duration - clip.duration
        
        # Freeze last frame with subtle zoom effect for visual interest
        frozen = clip.fx(freeze, t='end', freeze_duration=freeze_duration)
        zoomed_frozen = frozen.fx(vfx.resize, lambda t: 1 + 0.05 * (t / freeze_duration))
        
        return zoomed_frozen
```

**Replace Current Code**:
```python
# OLD (unified_video_generator.py:452)
extended_clip = loop(video_clip, duration=audio_duration)  # ❌ Causes looping

# NEW
from adaptive_engine.intelligent_frame_extender import IntelligentFrameExtender
extender = IntelligentFrameExtender()
extended_clip = extender.extend_clip_to_audio(video_clip, audio_duration)  # ✅ Smart extension
```

---

### **Day 6: Telemetry v3 + Performance Tracking**
**File**: `AnimateDiff/analytics/telemetry_v3.py` (~400 lines)

**What We Track**:
- Character consistency score
- Frame extension quality (RIFE vs slowmo vs freeze usage)
- Narrative coherence metrics
- Educational LoRA effectiveness

**🎯 PHASE 2 GOAL #2: Real-time Dashboard Backend - COMPLETED HERE**

---

### **Day 7: Integration + Demo Video**
**Deliverable**: `TTV_Studio_v1_demo.mp4` (3-scene demo showing all fixes)

**Demo Structure**:
```
Scene 1: "A young seeker begins her journey"
  ✅ Character identified as FEMALE from full story analysis
  ✅ Video extended to 6s using smart slow-mo (not looping)

Scene 2: "She walks through misty forest" 
  ✅ Same female character (consistent)
  ✅ Smooth transition from Scene 1
  ✅ Video extended using RIFE (no black screen)

Scene 3: "The seeker studies geometry diagrams"
  ✅ Educational LoRA kicks in for diagrams
  ✅ Same female character present
  ✅ Perfect audio-video sync
```

---

## 📊 **SUCCESS METRICS (Real Problems Solved)**

### ✅ Problem 1 Fixed: No More Looping
- **Before**: Video loops 2-3 times per clip (looks repetitive)
- **After**: Intelligent extension (RIFE/slowmo/freeze) based on content
- **Metric**: User satisfaction score: "Does video look repetitive?" → NO

### ✅ Problem 2 Fixed: Character Consistency
- **Before**: "Young seeker" → male, then "She" → female (inconsistent)
- **After**: Full story NLP analysis resolves gender BEFORE generation
- **Metric**: Character consistency > 95% across all scenes

### ✅ Problem 3 Fixed: Educational Diagrams
- **Before**: RealisticVision fails on geometry/diagrams
- **After**: Educational LoRA trained on 500 diagram images
- **Metric**: Diagram quality score > 8/10 (user evaluation)

### ✅ Problem 4 Fixed: RIFE Black Screen
- **Before**: RIFE generates black frames → audio continues on black screen
- **After**: Motion-guided RIFE + validity checking + freeze fallback
- **Metric**: Zero black screen occurrences

---

## 🛠️ **TECHNICAL IMPLEMENTATION NOTES**

### NLP Approach (Like LSTM/GRU):
```python
# Traditional Per-Clip Processing (WRONG ❌)
for sentence in story:
    gender = detect_gender(sentence)  # Only looks at current sentence
    generate_video(sentence, gender)  # Inconsistent!

# Our Approach: Full Story Analysis (CORRECT ✅)
characters = analyze_full_story(story)  # Process ALL sentences first
for sentence in story:
    char_info = characters[get_character_in_sentence(sentence)]
    generate_video(sentence, char_info)  # Consistent!
```

This is similar to LSTM because:
- **Sequential Processing**: We process sentences in order
- **Context Window**: We look at previous + next sentences
- **Hidden State**: Character information carries forward
- **Bidirectional**: We analyze forward AND backward context

### Educational LoRA Training:
```bash
# Prepare dataset of 500 educational images
datasets/educational_diagrams/
├── geometry_001.png (triangle with labels)
├── geometry_002.png (circle with radius)
├── formula_001.png (quadratic equation)
└── ... (497 more)

# Train LoRA
python -m adapters.educational_lora_trainer \
    --images_dir datasets/educational_diagrams \
    --base_model SG161222/Realistic_Vision_V5.1_noVAE \
    --output_dir adapters/educational_lora \
    --num_epochs 10
```

---

## 📅 **EXECUTION TIMELINE**

```
Day 1 (Nov 13): Identity Memory + Educational LoRA Trainer
Day 2 (Nov 14): Scene Graph Module
Day 3 (Nov 15): Narrative Sequencer with NLP (LSTM-like)
Day 4 (Nov 16): Emotion Controller
Day 5 (Nov 17): Intelligent Frame Extender (Fix looping + RIFE)
Day 6 (Nov 18): Telemetry v3
Day 7 (Nov 19): Integration + Demo
```

---

## ✅ **FINAL DELIVERABLE**

**Demo Video**: Shows before/after comparison
- **Before**: Looping video, inconsistent character, no diagrams, black screens
- **After**: Smooth video, consistent character, perfect diagrams, no black screens

**Metrics Report**:
```json
{
  "character_consistency": "98%",
  "video_looping_eliminated": true,
  "educational_diagram_quality": "8.5/10",
  "rife_black_screen_occurrences": 0,
  "user_satisfaction": "9.2/10"
}
```

---

**Ready to start Day 1?** 🚀

This revised plan directly addresses your actual production problems instead of building theoretical features!
