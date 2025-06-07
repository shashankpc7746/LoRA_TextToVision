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

=================================================================================================================================================================