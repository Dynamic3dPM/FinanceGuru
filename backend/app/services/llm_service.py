import httpx
import sqlite3
import json # To store raw_response dictionary as a string
from datetime import datetime 
from app.core.config import settings
from app.models.schemas import FinancialAnalysis # Assuming you want to structure the output

# Database setup
DB_PATH = "llm_interactions.sqlite" # Consider making this configurable

def init_db():
    """Initializes the database and creates the llm_responses table if it doesn't exist."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                prompt TEXT,
                model TEXT,
                raw_response TEXT,
                generated_text TEXT,
                summary TEXT,
                suggestions TEXT,
                spoken_response TEXT
            )
        """)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database initialization error: {e}") # Log this properly in a real app
    finally:
        if conn:
            conn.close()

# Initialize DB when the module is loaded
init_db()

def save_llm_interaction(prompt: str, model: str, raw_response: dict, generated_text: str, summary: str, suggestions: list[str], spoken_response: str):
    """Saves the LLM interaction details to the SQLite database, including spoken_response."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO llm_responses (timestamp, prompt, model, raw_response, generated_text, summary, suggestions, spoken_response)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (datetime.now(), prompt, model, json.dumps(raw_response), generated_text, summary, json.dumps(suggestions), spoken_response))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error saving LLM interaction to database: {e}") # Log this
    finally:
        if conn:
            conn.close()

async def analyze_text_with_llm(text_content: str) -> FinancialAnalysis:
    """
    Sends text to Ollama for analysis and returns a structured response.
    """
    if not settings.OLLAMA_API_URL:
        raise ValueError("OLLAMA_API_URL is not configured.")

    current_prompt = f"""
Act as a friendly and personal financial assistant. Review the provided financial statement and respond in a warm, conversational, first-person tone suitable for text-to-speech.

Instructions:
- Carefully review all the transactions in the statement.
- Summarize the user's spending by category (e.g., dining, groceries, travel, entertainment, etc.) in a conversational way. Mention which categories had the most spending and any notable patterns you see.
- Offer friendly, practical advice for budgeting and saving based on these patterns (e.g., "I noticed you spent quite a bit on dining out this month. Maybe try setting a dining budget for next month!").
- Your 'spoken_response' should sound like a real conversation, using "I" and "you", and should NOT list individual transactions, copy tables, or output any markdown or code blocks. Instead, talk about the overall spending habits and categories in a natural, flowing paragraph.
- Be positive, supportive, and easy to understand.
- Output a single JSON object with:
  - "summary": A brief, plain-language summary of the statement.
  - "insights_and_suggestions": A list of 2-3 actionable, friendly financial tips or observations based on the spending categories.
  - "spoken_response": A conversational, first-person paragraph for text-to-speech, referencing spending by category and giving advice in a friendly, personal way. This should sound like a natural voice, not a report or a list.

Example output:
{{
  "summary": "You spent the most this month on dining and groceries, with some purchases in entertainment and travel.",
  "insights_and_suggestions": [
    "Consider setting a budget for dining out next month.",
    "You might be able to save more by planning grocery trips in advance.",
    "Great job keeping your travel expenses reasonable!"
  ],
  "spoken_response": "Hi! I looked over your recent statement and noticed you spent quite a bit on dining out and groceries, with a few purchases in entertainment and travel. If you're looking to save a bit more, maybe try setting a dining budget or planning your grocery trips ahead of time. You're doing well—just a few small changes could help you save even more!"
}}

Text:
{text_content}
"""
    payload = {
        "model": "mistral", # Or make this configurable
        "prompt": current_prompt,
        "stream": False # Get the full response at once
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(settings.OLLAMA_API_URL, json=payload, timeout=60.0) # Increased timeout
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            
            ollama_response = response.json()
            generated_text = ollama_response.get("response", "")

            # Try to parse the generated_text as JSON to extract structured fields
            try:
                llm_json = json.loads(generated_text)
                summary = llm_json.get("summary", "")
                suggestions_list = llm_json.get("insights_and_suggestions", [])
                transactions_identified = llm_json.get("transactions", [])
                spoken_response = llm_json.get("spoken_response", "")
            except Exception:
                # Fallback to previous placeholder logic if parsing fails
                summary = "Summary from LLM: " + generated_text[:100]
                suggestions_list = ["Suggestion from LLM: " + generated_text[100:200]]
                transactions_identified = []
                spoken_response = generated_text

            # Save to database
            save_llm_interaction(
                prompt=current_prompt,
                model=payload["model"],
                raw_response=ollama_response,
                generated_text=generated_text,
                summary=summary,
                suggestions=suggestions_list,
                spoken_response=spoken_response
            )

            return FinancialAnalysis(
                summary=summary,
                suggestions=suggestions_list,
                transactions_identified=transactions_identified,
                spoken_response=spoken_response
            )

        except httpx.ReadTimeout:
            # Handle timeout specifically
            return FinancialAnalysis(summary="Error: The request to Ollama timed out.", suggestions=[])
        except httpx.HTTPStatusError as e:
            # Handle HTTP errors (e.g., Ollama server down, model not found)
            error_detail = e.response.text
            # Potentially save error information to DB as well, or a different table
            save_llm_interaction(
                prompt=current_prompt,
                model=payload.get("model", "unknown"),
                raw_response={"error": str(e), "status_code": e.response.status_code, "details": error_detail},
                generated_text="",
                summary=f"Error communicating with LLM: {e.response.status_code}",
                suggestions=[f"Details: {error_detail}"],
                spoken_response=""
            )
            return FinancialAnalysis(summary=f"Error communicating with LLM: {e.response.status_code}. Details: {error_detail}", suggestions=[])
        except Exception as e:
            # Catch other potential errors (e.g., network issues, JSON parsing errors)
            save_llm_interaction(
                prompt=current_prompt,
                model=payload.get("model", "unknown"), # payload might not be defined if error is early
                raw_response={"error": str(e)},
                generated_text="",
                summary=f"An unexpected error occurred: {str(e)}",
                suggestions=[],
                spoken_response=""
            )
            return FinancialAnalysis(summary=f"An unexpected error occurred: {str(e)}", suggestions=[])

# Example of a more specific prompt for transaction extraction if you want the LLM to output structured data:
# PROMPT_TEMPLATE = """
# Extract all financial transactions from the following text.
# For each transaction, provide the date, description, amount, and category.
# If a value is not present, use "N/A".
# Format the output as a JSON list of objects, where each object represents a transaction.
# Example:
# [
#   { "date": "YYYY-MM-DD", "description": "Grocery Store", "amount": 50.25, "category": "Groceries" },
#   { "date": "YYYY-MM-DD", "description": "Salary", "amount": 2000.00, "category": "Income" }
# ]
# Text:
# {text_content}
# """
