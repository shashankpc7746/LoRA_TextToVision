# LoRA\_TextToVision

📘 **Project**: From Language to Light - A LoRA Bootcamp

---

This repository documents the progress for the "From Language to Light" onboarding task. It starts with learning the fundamentals of Low-Rank Adaptation (LoRA) on text models and progressively builds towards text-to-image and text-to-video synthesis.

This README will be updated as each phase of the project is completed.

---

## ✅ Phase 1: LoRA Basics - Text Fine-tuning (Completed)

### 🎯 Objective

Understand, implement, and evaluate the efficiency of LoRA by comparing it with traditional full fine-tuning on a text classification task.

### 📝 Task

Fine-tune the `distilbert-base-uncased` model for sentiment analysis on the IMDb dataset.

### 📌 What is LoRA?

LoRA (Low-Rank Adaptation) is a Parameter-Efficient Fine-Tuning (PEFT) technique that updates only small trainable adapters in a model while freezing the rest of the weights.

### ⚡ Why is LoRA Efficient?

* Trains only \~1% of the parameters
* Reduces training time and model size
* Requires less GPU memory
* Easily sharable adapters

### 📊 Results Summary

| Metric               | Full Fine-Tuning    | LoRA Fine-Tuning  | Key Insight            |
| -------------------- | ------------------- | ----------------- | ---------------------- |
| Accuracy             | 87.5%               | 87.0%             | Comparable performance |
| Training Time        | 1359s (\~22.6 mins) | 1143s (\~19 mins) | \~15% faster           |
| Trainable Parameters | \~67M (100%)        | \~740K (1.09%)    | Huge reduction         |
| Model Size           | \~256 MB            | \~3 MB            | Tiny adapter           |

### 🧪 How to Run

```bash
pip install transformers==4.40.0 datasets==3.6.0 peft==0.10.0 accelerate torch wandb numpy
gurukul-lora-env\Scripts\activate
python LoRA_Text/train_distilbert_full_vs_lora.py
```

📂 Outputs:

* `./results/`
* `./logs/`
* `./LoRA_Text/`
* `./wandb/`

---

## ✅ Phase 2: LoRA for Vision – Image Fine-tuning with Stable Diffusion (Completed)

### 🎯 Objective

Apply LoRA to fine-tune the Stable Diffusion v1.4 model and generate stylized Gurukul-themed images.

### 📝 Task

Train a LoRA adapter on a small dataset (10 images). Use the adapter to generate visuals based on specific prompts.

### 📌 LoRA for Diffusion Models

LoRA inserts trainable adapters into UNet attention layers (`attn2.to_q`, `to_k`, `to_v`), updating only them.

### ⚡ Efficiency Insights

| Metric               | Value              | Insight                          |
| -------------------- | ------------------ | -------------------------------- |
| Trainable Parameters | 297,984 (\~0.035%) | Only adapters updated            |
| Total Parameters     | 859M               | Original model remains untouched |
| Training Steps       | 10                 | Minimal training required        |
| Output Adapter Size  | \~3 MB             | Easily deployable                |

### 📸 Prompts Used

* "AI robot teaching students in a village school"
* "A traditional Gurukul reimagined with VR and AR"
* "Students learning under a hologram tree"

### 🧪 How to Run

```bash
pip install diffusers transformers peft accelerate xformers
```

Activate environment:

```bash
gurukul-lora-env\Scripts\activate
```

Train LoRA adapter:

```bash
python LoRA_StableDiffusion/train_lora_sd.py
```

Generate inference from LoRA adapter:

```bash
python LoRA_StableDiffusion/inference_generate.py
```

Optional: direct prompt to image:

```bash
python LoRA_StableDiffusion/text2img_generate.py
```

📂 Outputs:

* Trained Adapter: `./LoRA_StableDiffusion/lora_output/`
* Generated Images: `./LoRA_StableDiffusion/outputs/`
* Training Dataset: `./LoRA_StableDiffusion/training_data/gurukul_hologram_style/`

⚠️ Notes:

* xFormers/triton not available
* AMP warnings, but successful convergence
* LoRA adapter saved for reuse (`adapter_model.safetensors`)

---

## ✅ Phase 3: Visual Sequencing - Image-to-Video Integration (Completed)

### 🎯 Objective

Convert short textual scenes into sequential image frames, enabling visual storytelling through text-to-image pipelines.

### 📝 Tasks

* **Automated Folder Handling**

  * Each run saves images in `frames_scene_X` (auto-incremented).
* **Prompt Optimization**

  * Paragraphs split into shorter parts.
  * Maintained object consistency.
* **Sequential Testing**

  * Example 1: Bunny with Magic Hat
  * Example 2: Alien Explorer on Mars
* **Future Idea**

  * Consistent character tracking via image-to-image chaining using `img2img` or `ControlNet`.

### 📁 Input & Output

#### Inputs

* **Text Prompts**: Paragraph or sentence list.
* **Style**: anime, fantasy, realistic, etc.
* **Scene Number**: Optional; automatically increments.

#### Outputs

* **Frames**: Saved to:

  ```
  VideoMaker/frames_scene_X/
  ```
* **Final Video**: Stored in:

  ```
  VideoMaker/video_outputs/
  ```

### 🔧 Tools Used

* Python, HuggingFace Diffusers
* Stable Diffusion (v1.4 + SDXL)
* Prompt Engineering
* File System Automation (os, pathlib)
* MoviePy for image-to-video conversion

---

## 📁 Subfolder Overview

| Folder                  | Purpose                                                           |
| ----------------------- | ----------------------------------------------------------------- |
| `LoRA_Text/`            | Scripts for text-based LoRA fine-tuning using DistilBERT          |
| `LoRA_StableDiffusion/` | Scripts for image LoRA training and inference on Stable Diffusion |
| `VideoMaker/`           | Frame sequencing and video generation using MoviePy               |
| `lora_output/`          | Contains LoRA adapter weights for image model                     |
| `outputs/`              | Stores generated images after inference                           |
| `training_data/`        | Dataset used for training the image LoRA model                    |

---

## 📌 Future Scope

* Integrate `ControlNet` or `img2img` for better temporal coherence
* Add more scenes with evolving object states
* Animate transitions between frames
* Build a web UI using Streamlit or Gradio

---

## 💡 Challenges Faced

* Limited GPU VRAM: couldn’t use higher batch sizes or longer training
* No access to triton/xformers in training phase
* Visual continuity remains tough without dedicated attention tracking modules

---

## ✅ Installation

To install all required libraries:

```bash
pip install -r requirements.txt
```

> Make sure to activate your environment beforehand using:

```bash
gurukul-lora-env\Scripts\activate
```
