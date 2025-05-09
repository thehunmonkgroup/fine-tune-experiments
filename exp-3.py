#!/usr/bin/env python3

# Step 1: Importing the libraries

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU with index 0
import torch
from huggingface_hub import login
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

login()

# Step 2: Setting up links to Hugging Face datasets and models

model_identifier = "meta-llama/Llama-3.1-8B-Instruct"
source_dataset = "training_data.jsonl"


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
epochs_count = 3
enable_fp16 = False
enable_bf16 = False
train_batch_size = 4
eval_batch_size = 4
accumulation_steps = 4
checkpointing_flag = True
grad_norm_limit = 0.7
train_learning_rate = 2e-4
decay_rate = 0.001
optimizer_type = "paged_adamw_32bit"
scheduler_type = "cosine"
warmup_percentage = 0.03
length_grouping = False
checkpoint_interval = 0
log_interval = 1

# Step 6: Setting up all the supervised fine-tuning arguments hyperparameters for fine-tuning

enable_packing = True
sequence_length_max = 1024 # Good to specifiy this so you don't run out of GPU VRAM
device_map = "auto"

# Step 7: Loading the dataset

dataset = load_dataset("json", data_files=source_dataset, split = "train")

# Step 8: Defining the QLoRA configuration

dtype_computation = getattr(torch, compute_dtype_bnb)
bnb_setup = BitsAndBytesConfig(load_in_4bit = enable_4bit,
                               bnb_4bit_quant_type = quant_type_bnb,
                               bnb_4bit_use_double_quant = double_quant_flag,
                               bnb_4bit_compute_dtype = dtype_computation)

# Step 9: Loading the pre-trained LLaMA 3.1 model

llama_model = AutoModelForCausalLM.from_pretrained(model_identifier,
                                                   quantization_config = bnb_setup,
                                                   device_map = device_map)
llama_model.config.use_case = False
llama_model.config.pretraining_tp = 1

# Step 10: Chatting with the pre-trained model

if hasattr(llama_model, "peft_config"):
    print(f"Model already has default PEFT configuration: {llama_model.peft_config}")

print(llama_model.is_loaded_in_4bit)  # should be True

chat = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Why is the sky blue?"}
]

my_tokenizer = AutoTokenizer.from_pretrained(model_identifier, use_fast=True, trust_remote_code = True, )
my_tokenizer.padding_side = "left"

# Add a real PAD if none exists
if my_tokenizer.pad_token is None:
    my_tokenizer.add_special_tokens({"pad_token": "<pad>"})  # or "<|finetune_right_pad_id|>"
    pad_id = my_tokenizer.pad_token_id
    llama_model.resize_token_embeddings(len(my_tokenizer))
    llama_model.config.pad_token_id = pad_id

prompt_str = my_tokenizer.apply_chat_template(
    chat,
    tokenize=False,
    add_generation_prompt=True          # <|start_header_id|>assistant … tag
)

inputs = my_tokenizer(prompt_str, return_tensors="pt").to(llama_model.device)
outputs = llama_model.generate(**inputs, max_new_tokens=800, do_sample=True)
print(my_tokenizer.decode(outputs[0], skip_special_tokens=True))

# Step 11: Preparing the dataset

def row_to_text(example: dict[str, str]):
    chat = [
        {"role": "user",   "content": example["user"]},
        {"role": "assistant", "content": example["assistant"]}
    ]
    example["text"] = my_tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=False
    )
    return {"text": example["text"]}

training_data = dataset.map(row_to_text, remove_columns=dataset.column_names)

# Step 12: Setting up the configuration for the LoRA fine-tuning method

peft_setup = LoraConfig(lora_alpha = lora_hyper_alpha,
                        lora_dropout = lora_hyper_dropout,
                        r = lora_hyper_r,
                        bias = "none",
                        task_type = "CAUSAL_LM",
                        inference_mode=False,
                        target_modules=["q_proj","v_proj", "k_proj","o_proj","gate_proj","up_proj","down_proj"] # for efficient GPU VRAM usage
                        )

llama_model.enable_input_require_grads() # A quirk of LoRA + gradient checkpointing

llama_model = get_peft_model(llama_model, peft_setup) # instead of including it with SFTTrainer

# Step 13: Creating a training configuration by setting the training parameters

train_args = SFTConfig(output_dir = results_dir,
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
                               warmup_ratio = warmup_percentage,
                               group_by_length = length_grouping,
                               lr_scheduler_type = scheduler_type,
                               gradient_checkpointing = checkpointing_flag,
                               dataset_text_field = "text",
                               max_seq_length = sequence_length_max,
                               packing = enable_packing,
                               )

# Step 14: Creating the Supervised Fine-Tuning Trainer

llama_sftt_trainer = SFTTrainer(model = llama_model,
                                args = train_args,
                                train_dataset = training_data,
                                processing_class=my_tokenizer
                                )

# Step 15: Training the model

llama_model.config.use_cache = False       # free key/value cache
llama_model.gradient_checkpointing_enable()  # discard activations

llama_sftt_trainer.train()

# Step 16: Chatting with the fine-tuned model

inputs = my_tokenizer(prompt_str, return_tensors="pt").to(llama_model.device)
outputs = llama_model.generate(**inputs, max_new_tokens=800, do_sample=True)
print(my_tokenizer.decode(outputs[0], skip_special_tokens=True))
