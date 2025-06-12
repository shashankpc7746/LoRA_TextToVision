# LoRA_TextToVision

📘 Project: From Language to Light - A LoRA Bootcamp
------------------------------------------------------

This repository documents the progress for the "From Language to Light" onboarding task. It starts with learning the fundamentals of Low-Rank Adaptation (LoRA) on text models and will progressively build towards text-to-image and text-to-video synthesis.

This README will be updated as each phase of the project is completed.

=================================================================================================================================================================

✅ Phase 1: LoRA Basics - Text Fine-tuning (Completed)
🎯 Objective:
To understand, implement, and evaluate the efficiency of LoRA by comparing it against traditional full fine-tuning on a text classification task.

📝 Task:
Fine-tune the distilbert-base-uncased model for sentiment analysis on the IMDb dataset.

📌 What is LoRA?
LoRA (Low-Rank Adaptation) is a Parameter-Efficient Fine-Tuning (PEFT) technique designed to make the fine-tuning of large language models more manageable.

Instead of retraining all the millions or billions of parameters in a large model, LoRA works on a simple but powerful principle: it freezes the original pre-trained weights of the model and injects small, trainable "adapter" layers into the model's architecture (typically into the attention layers). These adapters consist of low-rank matrices that learn the task-specific information. During training, only the weights of these newly added adapter layers are updated, while the vast majority of the original model remains untouched. The final output is a small file containing only the trained adapter weights, which can be loaded on top of the base model to perform the specific task.

⚡ Why is LoRA Efficient? A Practical Comparison
The core of this phase was to practically demonstrate LoRA's efficiency. The script LoRA_Text/train_distilbert_full_vs_lora.py was executed to collect direct comparison metrics.

📊 Results Summary
Metric	                    Full Fine-Tuning	          LoRA Fine-Tuning	          Key Insight
Accuracy	                     87.5%	                      87.0%	                   LoRA achieved nearly identical performance, with only a negligible 0.5% drop.
Training Time	                 1359.41s (~22.6 mins)	      1143.90s (~19.0 mins)	   LoRA was over 15% faster, a significant saving even on this small-scale task.
Trainable Parameters	         ~67 Million (100%)	          ~740 Thousand (1.09%)	   This is the most crucial metric. LoRA updated only 1% of the total parameters.
Final Model Size	             ~256 MB (Full model)	      ~3 MB (Adapter only)	   The resulting LoRA adapter is tiny, making it easy to store, share, and deploy.

📚 What Was Learned
Drastic Reduction in Computational Cost:
By training only ~1% of the parameters, LoRA significantly reduces the GPU memory and processing power required for fine-tuning. This makes fine-tuning large models accessible without high-end, specialized hardware.

Faster Iteration:
The ~15% reduction in training time allows for quicker experiments and hyperparameter tuning. This advantage grows exponentially with larger models and datasets.

Minimal Performance Trade-off:
The experiment confirms that it's possible to achieve performance nearly on par with full fine-tuning while benefiting from massive efficiency gains. The small drop in accuracy is a highly acceptable trade-off.

Portability and Modularity:
The small size of the saved LoRA adapter (~3 MB vs. ~256 MB for the full model) is a game-changer. One can have a single base model and numerous small, plug-and-play adapters for various tasks, which is incredibly efficient for storage and deployment.

🧪 How to Run the Script:

## Ensure you have a Conda/virtual environment set up with the required libraries. For this project, a stable environment was created using:
pip install transformers==4.40.0 datasets==3.6.0 peft==0.10.0 accelerate torch wandb numpy

## Activate the environment:
gurukul-lora-env\Scripts\activate   

## Run the training script from the project's root directory:
python LoRA_Text/train_distilbert_full_vs_lora.py

## Results, logs, and saved models will be generated in the following directories:

./results/

./logs/

./LoRA_Text/

Offline wandb logs: ./wandb/

===================================================================================================================================================

✅ Phase 2: LoRA for Vision – Image Fine-tuning with Stable Diffusion (Completed)

🎯 Objective:
To apply LoRA for parameter-efficient fine-tuning of the Stable Diffusion model and stylize image generation for futuristic "Gurukul"-themed concepts.

📝 Task:
Train a LoRA adapter on CompVis/stable-diffusion-v1-4 using a small 10-image dataset. After training, generate themed images using custom prompts and the learned visual style.

📌 What is LoRA for Diffusion Models?
Just like in text-based models, LoRA works by freezing the original weights of the UNet inside Stable Diffusion and inserting small trainable adapter modules in selected attention layers (attn2.to_q, to_k, to_v). During fine-tuning, only these adapters are updated — allowing the model to learn a new visual style or task without retraining the full base model.

This results in a compact and efficient model adapter that can be plugged into the original Stable Diffusion model to produce stylized images with minimal overhead.

⚡ Why is LoRA Efficient for Vision Tasks?
This phase demonstrates LoRA’s versatility beyond text models by successfully applying it to large vision diffusion models with only ~0.03% trainable parameters.

📊 Training Summary

Metric	                          Value	                       Insight
Trainable Parameters	   297,984 (~0.035%)	          Only a tiny portion of model was updated
Total Parameters	         859 Million	              Most of the model remains frozen
Training Steps	              10 steps	                  Small dataset, quick experimentation
Sample Loss Range	        0.002 – 0.534	              Typical for very small datasets with high variance
Output Adapter Size	            ~3 MB	                  Highly portable and modular

📸 Prompts Used for Image Generation:

"AI robot teaching students in a village school"

"A traditional Gurukul reimagined with VR and AR"

"Students learning under a hologram tree"

"A child meditating in a virtual garden"

"Knowledge as light in a dark classroom"

🧪 How to Run the Scripts:

## Ensure the environment has the required libraries:
pip install diffusers transformers peft accelerate xformers

## Activate the environment:
gurukul-lora-env\Scripts\activate

## Run the LoRA fine-tuning script:
python LoRA_StableDiffusion/train_lora_sd.py

## Then, generate images using the trained adapter:
python LoRA_StableDiffusion/inference_generate.py

Results:
🔧 Trained Adapter: ./lora_output/

🖼️ Final Images: ./outputs/

🧠 Dataset: ./training_data/gurukul_hologram_style/

⚠️ Warnings Noted During Training:

xFormers and triton not available — memory optimizations skipped

AMP deprecated warnings — model still trained correctly and loss converged

📚 What Was Learned

LoRA is Model-Agnostic:
We saw that LoRA can be successfully applied to both text and vision models — requiring minimal compute for meaningful style learning.

Fast and Flexible Style Transfer:
Despite using only 10 training samples, the model could stylize concepts like "hologram tree" and "AI teaching" effectively in under 10 steps.

Tiny but Powerful:
The final adapter (~3MB) gives us a stylized generation ability that normally requires re-training hundreds of millions of parameters.

Easier Deployment:
One base SD model can support multiple adapters for different visual styles — ideal for low-storage apps like mobile or embedded inference.

===================================================================================================================================================

✅ Phase 3: Visual Sequencing - Image-to-Video Integration

🎯 Objective:
To generate coherent sequences of images from short descriptive prompts and automate their organization into video-ready folders, setting the foundation for future image-to-video conversion.

📝 Tasks Performed:
Consistent Folder Management

Automated creation of output directories:
Each new run now saves images in a unique folder like frames_scene_1, frames_scene_2, etc., without overwriting older runs.

Prompt Optimization

Long paragraphs are split into shorter sentences for better granularity in image generation.

Ensured consistent object presence (e.g., "bunny with magic hat") across all prompts to reduce hallucinations in generated frames.

Sequential Prompt Testing

Designed and tested example scenes with a character performing actions step-by-step.

Example 1: Bunny + Magic Hat Journey

Example 2: Alien Explorer on Mars (short, sequential, consistent)

Discussion on Frame Consistency

Explored the idea of image-to-image chaining for consistent character motion across frames.

Identified this as a future direction using Stable Diffusion img2img or ControlNet.

📁 Output:
Scene-wise image folders in /VideoMaker/frames_scene_X/ format

Short, optimized prompts leading to visually coherent image sequences

Example prompts prepared for use in future video generation pipelines

🔧 Tools & Technologies Used:
Python, diffusers, Hugging Face Models

Stable Diffusion / SDXL pipelines

Prompt Engineering for visual storytelling

File system automation (os, pathlib)
