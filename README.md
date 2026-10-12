# AI-Customer-Support-Assistant

OpenRouter-powered e-commerce support assistant built with LangChain.

## Setup

1. Create and activate a virtual environment from this directory:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies and create your local credentials file:

	```bash
	pip install -r requirements.txt
	cp .env.example .env
	```

3. Put your OpenRouter API key in `.env`. You can optionally change
	`OPENROUTER_MODEL` there to another model available to your account.

4. Start the Streamlit app:

	```bash
	streamlit run app.py
	```

Conversation history is kept in memory for the current browser session. The
assistant does not have access to order systems and will not claim to issue
refunds or make account changes.

Try questions about a delayed order, a damaged product, return eligibility, or
a refund, then ask a follow-up to check that conversational context is retained.