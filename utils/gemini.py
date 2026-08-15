import os
from dotenv import load_dotenv
from google import genai

# Load variables from .env
load_dotenv()

# Get API Key
api_key = os.getenv("GEMINI_API_KEY")

# Create the Gemini client (replaces the old genai.configure() pattern)
client = genai.Client(api_key=api_key)

# Model name (kept as a constant so it's easy to change later)
MODEL_NAME = "gemini-3.6-flash"


# How many previous messages (user + assistant combined) to include as
# memory. Keeps the prompt from growing unbounded in very long chats.
MAX_HISTORY_MESSAGES = 10


def format_history(history):
    """
    Convert stored chat messages into a plain-text transcript for the prompt.
    Only role + content are used; the 'sources' field on assistant messages
    is ignored here since it's not needed for conversational context.
    """

    if not history:
        return "No previous conversation."

    # Keep only the most recent messages so the prompt stays a reasonable size
    recent_history = history[-MAX_HISTORY_MESSAGES:]

    lines = []

    for message in recent_history:

        role_label = "User" if message["role"] == "user" else "Assistant"

        lines.append(f"{role_label}: {message['content']}")

    return "\n".join(lines)


def ask_gemini(question, chunks, history=None):
    """
    Generate an answer using the retrieved chunks as context, taking prior
    conversation turns (history) into account for follow-up questions.
    """

    context = ""

    for item in chunks:
        context += f"""
File: {item['file']}
Page: {item['page']}
Chunk: {item['chunk']}

{item['text']}
--------------------
"""

    conversation_history = format_history(history)

    prompt = f"""
You are an AI assistant that answers questions only from the uploaded PDF(s).

Use ONLY the provided context to answer the current question.

If the answer is not present in the context, reply exactly:

"I couldn't find the answer in the uploaded PDF."

When possible, mention the page number where the information was found.

You are given the recent conversation history below. Use it ONLY to
understand what the user is referring to in follow-up questions (for
example, resolving "it" or "that" to something mentioned earlier).
Never use the conversation history itself as a source of facts — every
factual claim in your answer must still come from the Context section.

Conversation History:
{conversation_history}

Context:
{context}

Current Question:
{question}

Answer:
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text

    except Exception as e:
        return f"⚠️ Gemini API Error:\n\n{str(e)}"