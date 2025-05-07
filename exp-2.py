#!/usr/bin/env python3

# Step 1: Importing the libraries

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU with index 0
import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging)

import transformers, accelerate, trl, triton, bitsandbytes

print(f"transformers version: {transformers.__version__}")
print(f"accelerate version: {accelerate.__version__}")
print(f"trl version: {trl.__version__}")
print(f"triton version: {triton.__version__}")
print(f"bitsandbytes version: {bitsandbytes.__version__}")

# Step 2: Setting up links to Hugging Face datasets and models

model_identifier = "NousResearch/Llama-2-7b-chat-hf"
formatted_dataset = "aboonaji/wiki_medical_terms_llam2_format"
source_dataset = "gamino/wiki_medical_terms"

# Step 3: Setting up all the QLoRA hyperparameters for fine-tuning

lora_hyper_r = 64
lora_hyper_alpha = 16
lora_hyper_dropout = 0.1

# Step 4: Setting up all the bitsandbytes hyperparameters for fine-tuning

enable_4bit = True
compute_dtype_bnb = "float16"
quant_type_bnb = "nf4"
double_quant_flag = True

# Step 5: Setting up all the training arguments hyperparameters for fine-tuning

results_dir = "./results"
epochs_count = 10
enable_fp16 = False
enable_bf16 = False
train_batch_size = 4
eval_batch_size = 4
accumulation_steps = 1
checkpointing_flag = True
grad_norm_limit = 0.3
train_learning_rate = 2e-4
decay_rate = 0.001
optimizer_type = "paged_adamw_32bit"
scheduler_type = "cosine"
steps_limit = 100
warmup_percentage = 0.03
length_grouping = True
checkpoint_interval = 0
log_interval = 25

# Step 6: Setting up all the supervised fine-tuning arguments hyperparameters for fine-tuning

enable_packing = False
sequence_length_max = 512 # Good to specifiy this so you don't run out of GPU VRAM
device_assignment = {"": 0}

# Step 7: Loading the dataset

training_data = load_dataset(formatted_dataset, split = "train")


# Step 8: Defining the QLoRA configuration

dtype_computation = getattr(torch, compute_dtype_bnb)
bnb_setup = BitsAndBytesConfig(load_in_4bit = enable_4bit,
                               bnb_4bit_quant_type = quant_type_bnb,
                               bnb_4bit_use_double_quant = double_quant_flag,
                               bnb_4bit_compute_dtype = dtype_computation)

# Step 9a: Loading the pre-trained LLaMA 3.1 model

llama_model = AutoModelForCausalLM.from_pretrained(model_identifier,
                                                   quantization_config = bnb_setup,
                                                   device_map = device_assignment)
llama_model.config.use_case = False
llama_model.config.pretraining_tp = 1

# Step 9b: Chatting with the pre-trained model

if hasattr(llama_model, "peft_config"):
    print(f"Model already has default PEFT configuration: {llama_model.peft_config}")

print(llama_model.is_loaded_in_4bit)  # should be True

user_prompt = "What is Paracetamol poisoning? List its most common symptoms."

my_tokenizer = AutoTokenizer.from_pretrained(model_identifier, trust_remote_code = True, )
my_tokenizer.pad_token = my_tokenizer.eos_token
my_tokenizer.padding_side = "right"

text_generation_pipe = pipeline(task = "text-generation",
                                model = llama_model,
                                tokenizer = my_tokenizer,
                                max_length = 300)
generation_result = text_generation_pipe(f"<s>[INST] {user_prompt} [/INST]")
print(generation_result[0]["generated_text"])

# Step 10: Loading the pre-trained tokenizer for the LLaMA 3.1 model

llama_tokenizer = AutoTokenizer.from_pretrained(model_identifier, trust_remote_code = True, )
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

# Step 11: Setting up the configuration for the LoRA fine-tuning method

peft_setup = LoraConfig(lora_alpha = lora_hyper_alpha,
                        lora_dropout = lora_hyper_dropout,
                        r = lora_hyper_r,
                        bias = "none",
                        task_type = "CAUSAL_LM",
                        inference_mode=False,
                        target_modules=["q_proj","v_proj"] # for efficient GPU VRAM usage
                        )

llama_model.enable_input_require_grads() # A quirk of LoRA + gradient checkpointing

llama_model = get_peft_model(llama_model, peft_setup) # instead of including it with SFTTrainer

# Step 12: Creating a training configuration by setting the training parameters

train_args = SFTConfig(output_dir = results_dir, # Changed from TrainingArguments
                               num_train_epochs = epochs_count,
                               per_device_train_batch_size = train_batch_size,
                               per_device_eval_batch_size = eval_batch_size,
                               gradient_accumulation_steps = accumulation_steps,
                               learning_rate = train_learning_rate,
                               weight_decay = decay_rate,
                               optim = optimizer_type,
                               save_steps = checkpoint_interval,
                               logging_steps = log_interval,
                               fp16 = enable_fp16,
                               bf16 = enable_bf16,
                               max_grad_norm = grad_norm_limit,
                               max_steps = steps_limit,
                               warmup_ratio = warmup_percentage,
                               group_by_length = length_grouping,
                               lr_scheduler_type = scheduler_type,
                               gradient_checkpointing = checkpointing_flag,
                               dataset_text_field = "text", # Moved from SFTTrainer
                               max_seq_length = sequence_length_max, # Moved from SFTTrainer, changed from "max_length"
                               packing = enable_packing,
                               )

# Step 13: Creating the Supervised Fine-Tuning Trainer

llama_sftt_trainer = SFTTrainer(model = llama_model,
                                args = train_args,
                                train_dataset = training_data,
                                processing_class=llama_tokenizer  # Changed from tokenizer
                                )

# Step 14: Training the model

llama_model.config.use_cache = False       # free key/value cache
llama_model.gradient_checkpointing_enable()  # discard activations
train_args.gradient_checkpointing = True

llama_sftt_trainer.train()

# Step 15: Chatting with the fine-tuned model

user_prompt = "What is Paracetamol poisoning? List its most common symptoms."

text_generation_pipe = pipeline(task = "text-generation",
                                model = llama_model,
                                tokenizer = llama_tokenizer,
                                max_length = 300)
generation_result = text_generation_pipe(f"<s>[INST] {user_prompt} [/INST]")
print(generation_result[0]["generated_text"])
