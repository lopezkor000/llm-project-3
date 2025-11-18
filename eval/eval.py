from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch

print("\n" + "="*70)
print("🔄 LOADING SAVED LORA ADAPTER...")
print("="*70)

from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("../lora_finetune/spongebob-lora-adapter")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "../lora_finetune/spongebob-lora-adapter")
model.eval()

print("✅ Model loaded successfully!\n")

# Test with DIFFERENT prompts (not in training data)
print("="*70)
print("🏴‍☠️ TESTING WITH NEW PROMPTS:")
print("="*70 + "\n")

test_prompts = [
    "How's the weather today?",
    "What's your favorite color?",
    "Can you help me with my homework?",
    "What did you have for breakfast?",
    "Tell me about your ship",
    "What's the meaning of life?",
    "Do you like cats or dogs?",
    "What's your biggest treasure?",
    "Who is this?"
]

for user_prompt in test_prompts:
    messages = [{"role": "user", "content": user_prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )

    response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

    print(f"👤 User: {user_prompt}")
    print(f"🏴‍☠️ spongebob: {response}")
    print("-" * 70 + "\n")