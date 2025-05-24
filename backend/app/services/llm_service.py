import httpx
import sqlite3
import json # To store raw_response dictionary as a string
from datetime import datetime 
from app.core.config import settings
from app.models.schemas import FinancialAnalysis # Assuming you want to structure the output
from gtts import gTTS

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

def synthesize_spoken_response_to_mp3(spoken_response: str, output_path: str):
    """Synthesizes the spoken_response to an MP3 file using gTTS."""
    # Clean up text for natural TTS
    spoken_response = spoken_response.replace('—', '-')  # Replace em-dash with hyphen
    spoken_response = spoken_response.replace('..', '.')
    spoken_response = spoken_response.replace('  ', ' ')
    spoken_response = spoken_response.strip()

    tts = gTTS(text=spoken_response, lang='en', slow=False)
    tts.save(output_path)
    return output_path

async def analyze_text_with_llm(text_content: str) -> FinancialAnalysis:
    """
    Sends text to Ollama for analysis, returns a structured response, and synthesizes spoken_response to MP3.
    """
    if not settings.OLLAMA_API_URL:
        raise ValueError("OLLAMA_API_URL is not configured.")

    current_prompt = f"""
Act as a friendly, personal financial assistant. I'm looking over your financial statement, and I want to talk about what you spent on in a warm, conversational, first-person tone, perfect for text-to-speech.

Instructions:
- Review all transactions in the statement and group spending into categories like dining, groceries, travel, entertainment, shopping, or others. Estimate amounts or proportions if exact numbers aren't clear.
- Summarize the spending by category in a brief, plain-language way, highlighting the top 1-2 categories and any patterns (e.g., frequent dining out).
- Provide 2-3 actionable, friendly budgeting tips based on the spending patterns, using a supportive tone (e.g., "You might save a bit by trying meal prep for groceries!").
- For the 'spoken_response', write a natural, flowing paragraph as if I'm chatting with you. Use "I" and "you", focus on category spending (not individual transactions, dates, or technical details like APR), and include a budgeting tip. Avoid lists, tables, markdown, or jargon.
- Be positive, engaging, and easy to understand for TTS.
- Output a single JSON object with:
  - "summary": A short summary of spending by category (e.g., "You spent mostly on dining and groceries, with some shopping.").
  - "insights_and_suggestions": A list of 2-3 friendly, actionable budgeting tips or observations.
  - "spoken_response": A conversational paragraph for TTS, summarizing category spending and giving one tip in a personal tone.

Example output:
{{
  "summary": "You spent mostly on dining and groceries, with some shopping and travel.",
  "insights_and_suggestions": [
    "Try setting a monthly dining budget to keep things in check.",
    "Planning grocery trips could help you save a bit.",
    "Great job keeping travel costs low!"
  ],
  "spoken_response": "Hey there! I looked over your statement, and it seems you spent quite a bit on dining out and groceries, with a little on shopping and travel too. You’re doing great, but maybe try setting a dining budget next month to save a bit more. Keep it up!"
}}

Text:
{text_content}
"""
    payload = {
        "model": "dolphin3:latest",
        "prompt": current_prompt,
        "stream": False
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(settings.OLLAMA_API_URL, json=payload, timeout=60.0)
            response.raise_for_status()
            
            ollama_response = response.json()
            generated_text = ollama_response.get("response", "")

            # Try to parse the generated_text as JSON
            try:
                llm_json = json.loads(generated_text)
                summary = llm_json.get("summary", "")
                suggestions_list = llm_json.get("insights_and_suggestions", [])
                transactions_identified = llm_json.get("transactions", [])
                spoken_response = llm_json.get("spoken_response", "")
            except Exception:
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

            # Synthesize spoken_response to MP3
            output_path = f"financial_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            synthesize_spoken_response_to_mp3(spoken_response, output_path)

            return FinancialAnalysis(
                summary=summary,
                suggestions=suggestions_list,
                transactions_identified=transactions_identified,
                spoken_response=spoken_response
            )

        except httpx.ReadTimeout:
            save_llm_interaction(
                prompt=current_prompt,
                model=payload["model"],
                raw_response={"error": "Request timed out"},
                generated_text="",
                summary="Error: The request to Ollama timed out.",
                suggestions=[],
                spoken_response=""
            )
            return FinancialAnalysis(summary="Error: The request to Ollama timed out.", suggestions=[])
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            save_llm_interaction(
                prompt=current_prompt,
                model=payload["model"],
                raw_response={"error": str(e), "status_code": e.response.status_code, "details": error_detail},
                generated_text="",
                summary=f"Error communicating with LLM: {e.response.status_code}",
                suggestions=[f"Details: {error_detail}"],
                spoken_response=""
            )
            return FinancialAnalysis(summary=f"Error communicating with LLM: {e.response.status_code}. Details: {error_detail}", suggestions=[])
        except Exception as e:
            save_llm_interaction(
                prompt=current_prompt,
                model=payload.get("model", "unknown"),
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
