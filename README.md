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
gurukul-lora-env\Scripts\activate
pip install -r requirements.txt
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
gurukul-lora-env\Scripts\activate
python LoRA_StableDiffusion/train_lora_sd.py
python LoRA_StableDiffusion/inference_generate.py
```

📂 Outputs:

* Trained Adapter: `./lora_output/`
* Images: `./outputs/`
* Dataset: `./training_data/gurukul_hologram_style/`

⚠️ Notes:

* xFormers/triton not available
* AMP warnings, but successful convergence

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