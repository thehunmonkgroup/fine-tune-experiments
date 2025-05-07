#!/usr/bin/env python3

import pprint
from datasets import load_dataset
from huggingface_hub import login
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

class CausalCollator:
    def __init__(self, tokenizer):
        self.pad_id  = tokenizer.pad_token_id

    def __call__(self, features):
        ids     = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        labels  = [torch.tensor(f["labels"],    dtype=torch.long) for f in features]

        # left-pad or right-pad as you prefer; here we right-pad
        ids_pad    = pad_sequence(ids,    batch_first=True, padding_value=self.pad_id)
        labels_pad = pad_sequence(labels, batch_first=True, padding_value=-100)

        # attention mask: 1 where ids != pad_id
        attn_mask = (ids_pad != self.pad_id).long()

        return {
            "input_ids": ids_pad,
            "labels":    labels_pad,
            "attention_mask": attn_mask,
        }

login()

data = load_dataset("json", data_files="training_data.jsonl")
dataset = data["train"]
print(f"Total samples: {len(dataset)}")
# Optional: peek at one sample
pprint.pprint(dataset[0])
# Expected keys: ['system', 'user', 'assistant']

dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = dataset["train"]
eval_ds  = dataset["test"]
print(f"Train samples: {len(train_ds)}, Validation samples: {len(eval_ds)}")


# Determine compute dtype: use bfloat16 if supported (e.g. on A100 GPUs), else float16
compute_dtype = torch.float16
if torch.cuda.is_available():
    # Check if GPU supports bfloat16
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:  # Ampere or higher
        compute_dtype = torch.bfloat16

# Set up 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype
)

model_id = "meta-llama/Llama-3.1-8B"
# Load the model in 4-bit mode
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True  # enable if the model repo has custom code
)
model.config.use_cache = False  # Disable cache to allow training (important for gradient checkpointing)
print("Model loaded in 4-bit mode")

model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model.resize_token_embeddings(len(tokenizer))

# Function to format and tokenize one example
def tokenize_example(example):
    user_msg = example["user"].strip()
    assistant_msg = example["assistant"].strip()
    prompt = f"User: {user_msg}\nAssistant: "
    # Tokenize prompt and response separately
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(assistant_msg, add_special_tokens=False)["input_ids"]
    # Append EOS token to the answer (ensure the model knows where the answer ends)
    answer_ids = answer_ids + [tokenizer.eos_token_id]
    # Create labels, with prompt tokens masked out
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return {"input_ids": input_ids, "labels": labels}

# Apply tokenization to the training and validation sets
tokenized_train = train_ds.map(tokenize_example, remove_columns=train_ds.column_names)
tokenized_eval  = eval_ds.map(tokenize_example, remove_columns=eval_ds.column_names)

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # target key query/value projection in Transformer layers
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="llama-3b1-8b-qlora-output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # accumulate gradients to simulate batch size 8
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,  # use FP16 precision (set bf16=True instead if on A100/AMPERE for bfloat16)
    gradient_checkpointing=True,
    optim="adamw_bnb_8bit",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="no",
    report_to="none"   # no third-party logging (like W&B)
)

data_collator = CausalCollator(tokenizer)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
)

# Begin fine-tuning
trainer.train()
