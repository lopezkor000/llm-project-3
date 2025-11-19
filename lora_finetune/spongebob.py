import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from sklearn.model_selection import train_test_split

model_name = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM # Autoregressive Model
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# Load training data from JSON file
with open("../datasets/spongebob_dataset.json", "r", encoding="utf-8") as f:
    all_data = json.load(f)

print(f"Loaded {len(all_data)} total examples", flush=True)

# Split into 80% train, 20% test
train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
print(f"Training examples: {len(train_data)}", flush=True)
print(f"Test examples: {len(test_data)}", flush=True)

def format_chat(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

# Create train and test datasets
train_dataset = Dataset.from_list(train_data)
train_dataset = train_dataset.map(format_chat)

test_dataset = Dataset.from_list(test_data)
test_dataset = test_dataset.map(format_chat)

def tokenize_function(examples):
    result = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    result["labels"] = result["input_ids"].copy()
    return result

# Tokenize both train and test datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text", "messages"])
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text", "messages"])

training_args = TrainingArguments(
    output_dir="./lora-spongebob",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="no",
    eval_strategy="epoch",
    report_to="none",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
)

# Evaluate the model before fine-tuning
print("="*50, flush=True)
print("🔹 BEFORE FINE-TUNING:", flush=True)
print("="*50, flush=True)

test_messages = [
    {"role": "user", "content": "Hello, how are you?"}
]
# add_generation_prompt=True will properly add the assistant turn
prompt = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
print(f"Full prompt:\n{prompt}\n", flush=True)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
print(f"User: Hello, how are you?", flush=True)
print(f"Assistant: {response}\n", flush=True)

trainer.train()

# Evaluate the model after fine-tuning
print("\n" + "="*50, flush=True)
print("🔹 AFTER FINE-TUNING:", flush=True)
print("="*50, flush=True)

model.eval()
with torch.no_grad():
    test_messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    prompt = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    print(f"User: Hello, how are you?", flush=True)
    print(f"Assistant: {response}\n", flush=True)

model.save_pretrained("./spongebob-lora-adapter")
tokenizer.save_pretrained("./spongebob-lora-adapter")
print("✅ Saved LoRA adapter!", flush=True)