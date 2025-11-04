# Diverse Educational Dataset Generator - Quick Guide

## 📋 Overview

This script generates **300 diverse educational images** using SDXL across all learning domains for general-purpose Gurukul training.

## 🎯 What Gets Generated

### Categories (17 domains, ~18 images each):

1. **STEM - Mathematics** (10 prompts)
   - Equations, geometry, algebra, calculus, statistics, etc.

2. **STEM - Physics** (10 prompts)
   - Lab experiments, mechanics, optics, thermodynamics, etc.

3. **STEM - Chemistry** (10 prompts)
   - Molecules, periodic table, reactions, lab equipment, etc.

4. **STEM - Biology** (10 prompts)
   - Cell diagrams, anatomy, DNA, ecosystems, microscopy, etc.

5. **STEM - Computer Science** (10 prompts)
   - Programming, algorithms, data structures, networks, AI, etc.

6. **Humanities - History** (10 prompts)
   - Timelines, maps, historical figures, civilizations, etc.

7. **Humanities - Geography** (10 prompts)
   - World maps, climate, topography, countries, rivers, etc.

8. **Humanities - Literature** (10 prompts)
   - Books, poetry, grammar, authors, story structure, etc.

9. **Languages** (10 prompts)
   - Alphabets, flashcards, grammar, phonetics, vocabulary, etc.

10. **Arts - Music** (10 prompts)
    - Instruments, music theory, notes, chords, composition, etc.

11. **Arts - Visual** (10 prompts)
    - Painting, drawing, color theory, photography, sculpture, etc.

12. **Professional - Business** (10 prompts)
    - Presentations, charts, marketing, finance, meetings, etc.

13. **Professional - Technology** (10 prompts)
    - Engineering, CAD, robotics, circuits, blueprints, etc.

14. **Professional - Medical** (10 prompts)
    - Anatomy, diagnosis, first aid, equipment, X-rays, etc.

15. **General - Classroom** (10 prompts)
    - Modern classrooms, lecture halls, study areas, labs, etc.

16. **General - Digital Learning** (10 prompts)
    - Educational apps, VR, smartboards, e-learning, etc.

**Total: 170 unique prompts → Repeated/shuffled to create 300 images**

## 🚀 Usage

### Quick Start (Generate 300 images)
```bash
cd adapters/gurukul_lora
python generate_training_dataset.py
```

### Custom Number of Images
```bash
# Generate 200 images
python generate_training_dataset.py --num_images 200

# Generate 500 images
python generate_training_dataset.py --num_images 500
```

### Custom Output Directory
```bash
python generate_training_dataset.py --output_dir "datasets/my_custom_dataset"
```

### All Options
```bash
python generate_training_dataset.py \
    --num_images 300 \
    --output_dir "datasets/gurukul_keyframes" \
    --device "cuda" \
    --batch_size 1
```

## ⏱️ Time Estimates

| Images | Time (RTX 3060 Ti) | Time (High-end GPU) |
|--------|-------------------|---------------------|
| 100    | ~40 minutes       | ~20 minutes         |
| 200    | ~1.5 hours        | ~40 minutes         |
| 300    | ~2 hours          | ~1 hour             |
| 500    | ~3.5 hours        | ~1.5 hours          |

*Time based on ~25-30 seconds per image at 30 inference steps*

## 📊 Output

### Generated Files
```
datasets/gurukul_keyframes/
├── keyframe_0001.png      # Math equation diagram
├── keyframe_0002.png      # Physics experiment
├── keyframe_0003.png      # Chemistry molecules
├── ...
├── keyframe_0300.png      # Digital learning
└── captions.json          # All captions
```

### Captions Format
```json
{
  "keyframe_0001.png": "clean educational diagram of mathematical equation on whiteboard",
  "keyframe_0002.png": "physics laboratory experiment setup with equipment",
  ...
}
```

## ✅ What Happens Next

After generation completes:

1. **Review Images**: Check `datasets/gurukul_keyframes/` folder
2. **Verify Quality**: Ensure images are educational and diverse
3. **Start Training**: Run 100-epoch training with new dataset

```bash
cd adapters/gurukul_lora
# Edit train_optimized.py: Change range(10) to range(100)
python train_optimized.py
```

## 🎯 Expected Results

With this diverse dataset:

✅ Model can generate **any educational content**
✅ Works for math, science, history, arts, coding, etc.
✅ Not limited to ancient Gurukul theme
✅ General-purpose educational image generation
✅ High quality with 100 epochs training

## 🔧 Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size (already at 1)
# Or reduce resolution in the script (change 1024 to 768)
```

### Slow Generation
```bash
# This is normal - ~30 seconds per image
# Run overnight or use background process
```

### SDXL Not Found
```bash
# Remove local_files_only=True from script
# It will download SDXL automatically (first time only)
```

## 💡 Tips

1. **Run Overnight**: 300 images takes ~2 hours, perfect for overnight run
2. **Check Progress**: Script shows progress every 50 images
3. **Quality Check**: Review first 10-20 images to ensure quality
4. **Backup**: The script won't overwrite existing images by default

## 📈 Comparison

| Dataset | Old (Current) | New (Generated) |
|---------|--------------|-----------------|
| **Images** | 50 | 300 |
| **Type** | Placeholders | Real SDXL |
| **Diversity** | Low (20 themes) | High (170 themes) |
| **Domains** | 1 (Ancient Gurukul) | 17 (All education) |
| **Generalization** | Limited | Excellent |
| **Quality** | Text on colored bg | Photorealistic SDXL |

## 🎯 Ready to Start?

```bash
cd c:\Shashank\LoRA_TextToVision\adapters\gurukul_lora
python generate_training_dataset.py --num_images 300
```

**Estimated completion**: 2 hours  
**Output**: 300 diverse educational images  
**Next step**: 100-epoch training for production quality!
