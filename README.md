# LLM Personality Chat Web App

A Python Flask web application that allows users to chat with 3 different AI personality models: Friendly, Professional, and Creative.

## Features

- 🎨 Modern, responsive chat interface
- 🤖 Three distinct AI personalities to choose from
- 💬 Real-time message exchange
- 🎯 Simple and intuitive user experience
- ⚡ Fast response times

## Personalities

1. **Friendly Assistant** 😊 - A warm, enthusiastic, and supportive personality
2. **Professional Consultant** 💼 - A formal, concise, and business-oriented personality
3. **Creative Thinker** 🎨 - An imaginative, artistic, and unconventional personality

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd /workspaces/llm-project-3
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

3. **Start chatting!**
   - Select a personality using the buttons at the top
   - Type your message in the text box
   - Press Enter or click Send

## Project Structure

```
llm-project-3/
├── app.py                 # Flask backend server
├── templates/
│   └── index.html        # Frontend chat interface
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## How It Works

- **Backend (app.py)**: Flask server that handles chat requests and generates responses based on the selected personality
- **Frontend (index.html)**: Single-page application with a chat interface that communicates with the backend via AJAX

## Future Enhancements

- Integrate actual LLM APIs (OpenAI, Anthropic, etc.)
- Add conversation history persistence
- Implement user authentication
- Add more personality types
- Deploy to cloud platform

## Technologies Used

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript
- **Styling**: Custom CSS with gradient backgrounds and animations

## Note

This is a demo application with simulated AI responses. To integrate real LLM models, you would need to:
1. Sign up for an LLM API service (OpenAI, Anthropic, etc.)
2. Install the appropriate SDK (e.g., `pip install openai`)
3. Replace the `generate_response()` function with actual API calls
4. Add API keys to environment variables
