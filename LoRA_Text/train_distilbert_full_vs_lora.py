# train_distilbert_full_vs_lora.py

import torch
import time
import os
import inspect # For inspecting class initializers

# --- Enhanced Library Version and Path Diagnostics ---
try:
    print("--- Library Diagnostics ---")
    import transformers
    print(f"✅ Using Transformers version: {transformers.__version__}")
    print(f"   Transformers library path: {transformers.__file__}")

    from transformers import TrainingArguments
    # Attempt to print recognized arguments for TrainingArguments
    try:
        init_signature = inspect.signature(TrainingArguments.__init__)
        params_list = list(init_signature.parameters.keys())
        print(f"   TrainingArguments __init__ parameters (from inspect): {params_list}")
        
        # More detailed check for 'evaluation_strategy'
        found_param_by_iteration = False
        print("   Detailed check for 'evaluation_strategy' in params_list:")
        for idx, p_name in enumerate(params_list):
            # Print details for the parameter that looks like 'evaluation_strategy' or nearby ones
            if 'eval' in str(p_name) or 'strategy' in str(p_name) or idx < 5 or idx > len(params_list) - 5 : # Log a few params for context
                 print(f"      Param[{idx}]: '{p_name}' (type: {type(p_name)}), repr: {repr(p_name)}")
            if p_name == 'evaluation_strategy':
                print(f"      ✅ Found 'evaluation_strategy' by direct '==' comparison: p_name == 'evaluation_strategy' is True for '{p_name}'")
                found_param_by_iteration = True
                # No break here, let it print if there are duplicates, though unlikely for function args
        
        if found_param_by_iteration:
            print("   ✅ 'evaluation_strategy' WAS found in TrainingArguments parameters by iterating and direct '==' comparison.")
        else:
            print("   ❌ 'evaluation_strategy' WAS NOT found by iterating and direct '==' comparison.")
            print("      This is highly unusual if it visually appears in the printed params_list above.")

        # Original 'in' check for comparison with the iteration method
        if 'evaluation_strategy' in params_list:
            print("   ✅ Original 'in' check: 'evaluation_strategy' IS in TrainingArguments parameters.")
        else:
            print("   ❌ Original 'in' check: 'evaluation_strategy' IS NOT in TrainingArguments parameters.")


        # ATTEMPT IMMEDIATE TEST only if found by iteration (more reliable check)
        if found_param_by_iteration:
            print("   Attempting minimal TrainingArguments instantiation with evaluation_strategy (based on iteration check)...")
            try:
                if not os.path.exists("./test_output_dir_diag"):
                    os.makedirs("./test_output_dir_diag")
                test_args = TrainingArguments(output_dir="./test_output_dir_diag", evaluation_strategy="epoch")
                print("   ✅ Minimal instantiation with evaluation_strategy SUCCEEDED in diagnostic block.")
            except TypeError as te:
                print(f"   ❌ Minimal instantiation with evaluation_strategy FAILED in diagnostic block: {te}")
                print(f"   ❌ This means the TrainingArguments class (version: {transformers.__version__}) is not behaving as inspect or iteration suggests.")
            except Exception as e_diag:
                 print(f"   ❌ An unexpected error occurred during minimal instantiation diagnostic: {e_diag}")
        else:
            print("   Skipping minimal instantiation test as 'evaluation_strategy' was not confirmed by iteration.")


    except Exception as e_inspect:
        print(f"   Could not inspect TrainingArguments.__init__: {e_inspect}")

    import peft
    print(f"✅ Using PEFT version: {peft.__version__}")
    print(f"   PEFT library path: {peft.__file__}")

    import datasets
    print(f"✅ Using Datasets version: {datasets.__version__}")
    print(f"   Datasets library path: {datasets.__file__}")
    
    print(f"✅ Using PyTorch version: {torch.__version__}")
    print(f"   PyTorch library path: {torch.__file__}")
    print("--- End Library Diagnostics ---\n")

except ImportError as e_import:
    print(f"❌ Critical Import Error: {e_import}. One of the core libraries is not installed correctly.")
    print("   Please ensure transformers, peft, datasets, and torch are installed in your environment.")
    exit()
# --- End Diagnostics ---


from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    # TrainingArguments already imported for diagnostics
    Trainer,
    DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType

import wandb
import numpy as np # For metrics calculation

# Function to compute metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = np.mean(predictions == labels)
    return {"accuracy": accuracy}

try:
    # ✅ Offline mode to prevent wandb login issues
    os.environ["WANDB_MODE"] = "offline" 
    if not os.path.exists("./wandb"):
        os.makedirs("./wandb")
    
    wandb.init(project="Gurukul-LoRA-Text", name="distilbert_full_vs_lora_run") 
    print("✅ wandb initialized (offline mode)")

    # ✅ Load IMDb dataset
    try:
        dataset = load_dataset("imdb")
        print("✅ IMDb dataset loaded successfully")
    except Exception as e:
        print(f"❌ Error loading IMDb dataset: {e}")
        print("   Ensure you have an internet connection or the dataset is cached.")
        raise

    # ✅ Load tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    print("✅ Tokenizer loaded")

    # ✅ Tokenize the dataset
    def tokenize_function(batch): 
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.remove_columns(["text"]) 
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"]) 
    print("✅ Tokenization complete. Sample keys:", dataset["train"][0].keys())

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- Full Fine-tuning ---
    print("\n--- Starting Full Fine-tuning Phase ---")
    model_full = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    print("✅ DistilBERT model (for full fine-tuning) loaded")

    print("Instantiating TrainingArguments for full fine-tuning...")
    training_args_full = TrainingArguments(
        output_dir="./results/full_finetuning",
        num_train_epochs=1, 
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch", 
        save_strategy="epoch",       
        logging_dir="./logs/full_finetuning",
        logging_steps=100, 
        report_to="wandb",
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        load_best_model_at_end=True, 
        metric_for_best_model="accuracy", 
    )
    print("✅ TrainingArguments for full fine-tuning instantiated.")


    train_dataset_full = dataset["train"].shuffle(seed=42).select(range(2000)) 
    eval_dataset_full = dataset["test"].shuffle(seed=42).select(range(1000))   

    trainer_full = Trainer(
        model=model_full,
        args=training_args_full,
        train_dataset=train_dataset_full,
        eval_dataset=eval_dataset_full,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics, 
    )

    print("🚀 Starting FULL fine-tuning...")
    start_full_time = time.time()
    train_result_full = trainer_full.train()
    end_full_time = time.time()
    full_tuning_time = end_full_time - start_full_time
    print(f"⏱️ Full fine-tuning time: {full_tuning_time:.2f} seconds")

    metrics_full = train_result_full.metrics
    trainer_full.log_metrics("train_full", metrics_full)
    trainer_full.save_metrics("train_full", metrics_full)
    
    eval_metrics_full = trainer_full.evaluate()
    print(f"📊 Full fine-tuning evaluation metrics: {eval_metrics_full}")
    trainer_full.log_metrics("eval_full", eval_metrics_full)
    trainer_full.save_metrics("eval_full", eval_metrics_full)
    
    wandb.log({
        "full_tuning_time_seconds": full_tuning_time, 
        "full_eval_accuracy": eval_metrics_full.get("eval_accuracy", 0),
        "full_train_loss": metrics_full.get("train_loss") 
    })

    full_model_save_path = "./LoRA_Text/distilbert_full_finetuned"
    model_full.save_pretrained(full_model_save_path)
    tokenizer.save_pretrained(full_model_save_path) 
    print(f"💾 Full fine-tuned model saved to {full_model_save_path}")

    # --- LoRA Fine-tuning ---
    print("\n--- Starting LoRA Fine-tuning Phase ---")
    model_lora_base = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    
    lora_config = LoraConfig(
        r=8,                            
        lora_alpha=16,                   
        target_modules=["q_lin", "v_lin"], 
        lora_dropout=0.1,                
        bias="none",                     
        task_type=TaskType.SEQ_CLS       
    )

    model_lora_peft = get_peft_model(model_lora_base, lora_config)
    print("✅ DistilBERT model with LoRA adapter initialized")
    model_lora_peft.print_trainable_parameters() 

    print("Instantiating TrainingArguments for LoRA fine-tuning...")
    training_args_lora = TrainingArguments(
        output_dir="./results/lora_finetuning",
        num_train_epochs=1, 
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch", 
        logging_dir="./logs/lora_finetuning",
        logging_steps=100,
        report_to="wandb",
        learning_rate=3e-4, 
        weight_decay=0.01,
        warmup_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    print("✅ TrainingArguments for LoRA fine-tuning instantiated.")

    
    train_dataset_lora = dataset["train"].shuffle(seed=42).select(range(2000))
    eval_dataset_lora = dataset["test"].shuffle(seed=42).select(range(1000))

    trainer_lora = Trainer(
        model=model_lora_peft,
        args=training_args_lora,
        train_dataset=train_dataset_lora,
        eval_dataset=eval_dataset_lora,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics, 
    )

    print("🚀 Starting LoRA fine-tuning...")
    start_lora_time = time.time()
    train_result_lora = trainer_lora.train()
    end_lora_time = time.time()
    lora_tuning_time = end_lora_time - start_lora_time
    print(f"⏱️ LoRA fine-tuning time: {lora_tuning_time:.2f} seconds")

    metrics_lora = train_result_lora.metrics
    trainer_lora.log_metrics("train_lora", metrics_lora)
    trainer_lora.save_metrics("train_lora", metrics_lora)

    eval_metrics_lora = trainer_lora.evaluate()
    print(f"📊 LoRA fine-tuning evaluation metrics: {eval_metrics_lora}")
    trainer_lora.log_metrics("eval_lora", eval_metrics_lora)
    
    # A small typo was here, eval_lora_lora, corrected to eval_metrics_lora
    # This would only affect saving metrics, not the core error
    trainer_lora.save_metrics("eval_lora", eval_metrics_lora) 


    wandb.log({
        "lora_tuning_time_seconds": lora_tuning_time, 
        "lora_eval_accuracy": eval_metrics_lora.get("eval_accuracy", 0),
        "lora_train_loss": metrics_lora.get("train_loss")
    })
    
    lora_model_save_path = "./LoRA_Text/distilbert_lora_adapter"
    model_lora_peft.save_pretrained(lora_model_save_path)
    tokenizer.save_pretrained(lora_model_save_path) 
    print(f"💾 LoRA adapter saved to {lora_model_save_path}")

    # --- Comparison ---
    print("\n--- Comparison Summary ---")
    print(f"Full fine-tuning time: {full_tuning_time:.2f}s, Accuracy: {eval_metrics_full.get('eval_accuracy', 'N/A')}, Train Loss: {metrics_full.get('train_loss', 'N/A')}")
    print(f"LoRA fine-tuning time: {lora_tuning_time:.2f}s, Accuracy: {eval_metrics_lora.get('eval_accuracy', 'N/A')}, Train Loss: {metrics_lora.get('train_loss', 'N/A')}")
    
    wandb.summary["full_final_accuracy"] = eval_metrics_full.get('eval_accuracy', 0)
    wandb.summary["lora_final_accuracy"] = eval_metrics_lora.get('eval_accuracy', 0)
    wandb.summary["full_total_time_seconds"] = full_tuning_time
    wandb.summary["lora_total_time_seconds"] = lora_tuning_time

    wandb.finish()
    print("✅ wandb session closed.")
    print("🎉 Training script completed successfully!")

except Exception as e_main:
    print(f"❌ An error occurred in the main script body: {e_main}")
    import traceback
    traceback.print_exc() 
    if wandb.run: # Check if wandb.run is not None
        wandb.finish(exit_code=1)
    else:
        print("wandb.run was None, not attempting to finish wandb.")
