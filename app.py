from flask import Flask, render_template, request, jsonify
import random
import time

app = Flask(__name__)

# Simulated LLM personalities - In production, these would call actual LLM APIs
PERSONALITIES = {
    "friendly": {
        "name": "Friendly Assistant",
        "description": "A warm, enthusiastic, and supportive personality",
        "responses": [
            "That's such a great question! {}",
            "I'm so happy to help you with that! {}",
            "Oh, I love talking about this! {}",
            "You're doing amazing! Let me share: {}",
            "What a wonderful thing to ask about! {}"
        ]
    },
    "professional": {
        "name": "Professional Consultant",
        "description": "A formal, concise, and business-oriented personality",
        "responses": [
            "Based on my analysis: {}",
            "From a professional standpoint: {}",
            "The optimal approach would be: {}",
            "To address your inquiry directly: {}",
            "In consideration of your request: {}"
        ]
    },
    "creative": {
        "name": "Creative Thinker",
        "description": "An imaginative, artistic, and unconventional personality",
        "responses": [
            "Ooh, let me paint you a picture! {}",
            "Here's a fresh perspective: {}",
            "Imagine this in a totally new way: {}",
            "Let's think outside the box! {}",
            "What if we looked at it like this? {}"
        ]
    }
}


def generate_response(message, personality):
    """
    Simulate LLM response based on personality
    In production, this would call an actual LLM API (OpenAI, Anthropic, etc.)
    """
    if personality not in PERSONALITIES:
        personality = "friendly"
    
    # Simulate processing time
    time.sleep(0.5)
    
    # Get personality info
    personality_data = PERSONALITIES[personality]
    
    # Generate a contextual response
    base_response = f"I understand you're asking about: '{message}'. "
    
    if "hello" in message.lower() or "hi" in message.lower():
        base_response = f"Hello! Nice to meet you! "
    elif "how are you" in message.lower():
        base_response = f"I'm doing well, thank you for asking! "
    elif "?" in message:
        base_response = f"That's an interesting question. "
    
    # Add personality-specific flair
    template = random.choice(personality_data["responses"])
    response = template.format(base_response)
    
    return response


@app.route('/')
def index():
    return render_template('index.html', personalities=PERSONALITIES)


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        personality = data.get('personality', 'friendly')
        
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
