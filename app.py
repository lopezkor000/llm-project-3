from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = Flask(__name__)

# Load base model and tokenizer
base_model_name = "Qwen/Qwen2-0.5B-Instruct"
print("Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA adapters
print("Loading LoRA adapters...")
models = {}
models["dwight"] = PeftModel.from_pretrained(base_model, "lora_finetune/dwight-lora-adapter")
models["michael"] = PeftModel.from_pretrained(base_model, "lora_finetune/michael-lora-adapter")
models["spongebob"] = PeftModel.from_pretrained(base_model, "lora_finetune/spongebob-lora-adapter")

print("All models loaded successfully!")

# Model personalities
PERSONALITIES = {
    "dwight": {
        "name": "Dwight Schrute",
        "description": "Assistant Regional Manager with beet farm wisdom"
    },
    "michael": {
        "name": "Michael Scott",
        "description": "World's Best Boss with unique management style"
    },
    "spongebob": {
        "name": "SpongeBob SquarePants",
        "description": "Optimistic fry cook from Bikini Bottom"
    }
}


def generate_response(message, personality):
    """
    Generate response using fine-tuned LoRA models
    """
    if personality not in models:
        personality = "dwight"
    
    # Get the appropriate model
    model = models[personality]
    
    # Format the message as a chat
    messages = [
        {"role": "user", "content": message}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize and generate
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "<|im_start|>assistant" in full_response:
        response = full_response.split("<|im_start|>assistant")[-1].strip()
        response = response.replace("<|im_end|>", "").strip()
    else:
        response = full_response[len(text):].strip()
    
    return response


@app.route('/')
def index():
    return render_template('index.html', personalities=PERSONALITIES)


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        personality = data.get('personality', 'dwight')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Generate response
        response = generate_response(message, personality)
        
        return jsonify({
            'response': response,
            'personality': personality,
            'personality_name': PERSONALITIES[personality]['name']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
