# LLM Chat Application

## Prerequisites
- Python 3.8+
- OpenAI API Key

## Setup Instructions
1. Clone the repository
2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up OpenAI API Key
- Open `.env` file
- Replace `your_openai_api_key_here` with your actual OpenAI API key

5. Run the application
```bash
python app.py
```

6. Open a web browser and navigate to `http://localhost:5000`

## Features
- Context-aware chat interface
- AI-powered responses using Langchain
- Conversation reset functionality
- Responsive, stylish UI with Tailwind CSS