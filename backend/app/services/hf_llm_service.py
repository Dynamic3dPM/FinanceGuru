import torch
import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    pipeline,
    BitsAndBytesConfig
)
from app.core.config import settings
from app.models.schemas import FinancialAnalysis
from app.services.rag_service import rag_service
from gtts import gTTS

# Database setup
DB_PATH = "llm_interactions.sqlite"

class HuggingFaceLLMService:
    """Service for interacting with Hugging Face models with RAG support."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialize_model()
        self._init_db()
    
    def _initialize_model(self):
        """Initialize the Hugging Face model and tokenizer."""
        try:
            print(f"Loading model {settings.HF_MODEL_NAME} on {self.device}...")
            
            # Configure quantization for better memory usage
            quantization_config = None
            if torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.HF_MODEL_NAME,
                cache_dir=settings.HF_CACHE_DIR,
                trust_remote_code=True
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model based on type
            model_config = {
                "cache_dir": settings.HF_CACHE_DIR,
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            }
            
            if quantization_config:
                model_config["quantization_config"] = quantization_config
            else:
                model_config["device_map"] = "auto"
            
            # Try to determine model type from config first
            try:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(settings.HF_MODEL_NAME, trust_remote_code=True)
                is_encoder_decoder = getattr(config, 'is_encoder_decoder', False)
                
                if is_encoder_decoder:
                    # Seq2seq models like BlenderBot, T5, BART
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        settings.HF_MODEL_NAME,
                        **model_config
                    )
                    model_type = "seq2seq"
                else:
                    # Causal LM models like GPT, Phi-2, DialoGPT
                    self.model = AutoModelForCausalLM.from_pretrained(
                        settings.HF_MODEL_NAME,
                        **model_config
                    )
                    model_type = "causal"
                    
            except Exception as config_error:
                print(f"Could not determine model type from config: {config_error}")
                # Fallback: try causal first (most common), then seq2seq
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        settings.HF_MODEL_NAME,
                        **model_config
                    )
                    model_type = "causal"
                except:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        settings.HF_MODEL_NAME,
                        **model_config
                    )
                    model_type = "seq2seq"
            
            # Create text generation pipeline
            pipeline_kwargs = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "device_map": "auto" if not quantization_config else None
            }
            
            # Only add return_full_text for causal models
            if model_type == "causal":
                pipeline_kwargs["return_full_text"] = False
            # text2text-generation doesn't use return_full_text parameter
            
            self.pipeline = pipeline(
                "text-generation" if model_type == "causal" else "text2text-generation",
                **pipeline_kwargs
            )
            
            print(f"Model loaded successfully as {model_type} model")
            print(f"Pipeline task: {self.pipeline.task}")
            print(f"Model name: {settings.HF_MODEL_NAME}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to a smaller model
            try:
                print("Attempting to load fallback model...")
                self._load_fallback_model()
            except Exception as fallback_error:
                print(f"Fallback model also failed: {fallback_error}")
                raise
    
    def _load_fallback_model(self):
        """Load a smaller fallback model if the main model fails."""
        fallback_model = "facebook/blenderbot-400M-distill"  # Better conversational model
        print(f"Loading fallback model: {fallback_model}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                fallback_model,
                cache_dir=settings.HF_CACHE_DIR
            )
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                fallback_model,
                cache_dir=settings.HF_CACHE_DIR,
                torch_dtype=torch.float32
            )
            
            self.pipeline = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device
                # Note: text2text-generation doesn't use return_full_text parameter
            )
            print(f"BlenderBot loaded successfully as seq2seq model")
            print(f"Pipeline task: text2text-generation")
        except:
            # Final fallback to DialoGPT if BlenderBot fails
            fallback_model = "microsoft/DialoGPT-small"
            print(f"Loading final fallback model: {fallback_model}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                fallback_model,
                cache_dir=settings.HF_CACHE_DIR
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                fallback_model,
                cache_dir=settings.HF_CACHE_DIR,
                torch_dtype=torch.float32
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                return_full_text=False
            )
            print(f"DialoGPT loaded successfully as causal model")
            print(f"Pipeline task: text-generation")
    
    def _init_db(self):
        """Initialize the database and create the llm_responses table if it doesn't exist."""
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
                    spoken_response TEXT,
                    rag_context TEXT,
                    rag_chunks_used INTEGER
                )
            """)
            
            # Add new columns if they don't exist (for existing databases)
            try:
                cursor.execute("ALTER TABLE llm_responses ADD COLUMN rag_context TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists
                
            try:
                cursor.execute("ALTER TABLE llm_responses ADD COLUMN rag_chunks_used INTEGER")
            except sqlite3.OperationalError:
                pass  # Column already exists
                
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database initialization error: {e}")
        finally:
            if conn:
                conn.close()
    
    def save_llm_interaction(self, prompt: str, model: str, raw_response: dict, 
                           generated_text: str, summary: str, suggestions: list, 
                           spoken_response: str, rag_context: str = "", rag_chunks_used: int = 0):
        """Save the LLM interaction details to the SQLite database."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO llm_responses 
                (timestamp, prompt, model, raw_response, generated_text, summary, suggestions, 
                 spoken_response, rag_context, rag_chunks_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(), prompt, model, json.dumps(raw_response), 
                generated_text, summary, json.dumps(suggestions), 
                spoken_response, rag_context, rag_chunks_used
            ))
            conn.commit()
        except sqlite3.Error as e:
            print(f"Error saving LLM interaction to database: {e}")
        finally:
            if conn:
                conn.close()
    
    def synthesize_spoken_response_to_mp3(self, spoken_response: str, output_path: str):
        """Synthesize the spoken_response to an MP3 file using gTTS."""
        # Clean up text for natural TTS
        spoken_response = spoken_response.replace('—', '-')
        spoken_response = spoken_response.replace('..', '.')
        spoken_response = spoken_response.replace('  ', ' ')
        spoken_response = spoken_response.strip()

        tts = gTTS(text=spoken_response, lang='en', slow=False)
        tts.save(output_path)
        return output_path
    
    def generate_response(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.8) -> str:
        """Generate a response using the Hugging Face model with optimized parameters for natural speech."""
        try:
            # Truncate prompt if it's too long for the model
            max_prompt_length = 100  # Much shorter to avoid token limit issues
            if len(prompt.split()) > max_prompt_length:
                prompt_words = prompt.split()[:max_prompt_length]
                prompt = " ".join(prompt_words) + "..."
            
            # Use parameters optimized for natural, conversational text generation
            generation_kwargs = {
                "max_new_tokens": min(max_new_tokens, 128),  # Reduced to stay within limits
                "temperature": temperature,
                "do_sample": True,
                "top_p": 0.92,
                "top_k": 40,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.eos_token_id,
                "clean_up_tokenization_spaces": True
            }
            
            # Don't include return_full_text in generation kwargs - it's already set in pipeline config
            
            response = self.pipeline(prompt, **generation_kwargs)
            
            if isinstance(response, list) and len(response) > 0:
                generated_text = response[0].get('generated_text', '').strip()
                # Clean up the response for better TTS
                generated_text = self._clean_text_for_speech(generated_text)
                return generated_text
            return str(response).strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def _clean_text_for_speech(self, text: str) -> str:
        """Clean generated text to make it more suitable for text-to-speech."""
        import re
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common issues with generated text
        text = text.replace(' ,', ',')
        text = text.replace(' .', '.')
        text = text.replace(' !', '!')
        text = text.replace(' ?', '?')
        
        # Remove excessive punctuation
        text = re.sub(r'\.\.+', '.', text)
        text = re.sub(r'\!\!+', '!', text)
        text = re.sub(r'\?\?+', '?', text)
        
        # Ensure sentences end properly
        if text and not text[-1] in '.!?':
            text += '.'
            
        return text.strip()
    
    async def analyze_text_with_llm_and_rag(self, text_content: str, use_rag: bool = True) -> FinancialAnalysis:
        """
        Analyze text using Hugging Face model with optional RAG context.
        Generates comprehensive temporal analysis including month-to-month comparisons.
        """
        try:
            # Get temporal context for date-aware analysis
            temporal_context = self._get_temporal_context()
            
            # Get RAG context if enabled - expand for comprehensive analysis
            rag_context = ""
            rag_chunks_used = 0
            all_previous_statements = ""
            
            if use_rag:
                # Get broader context for comprehensive analysis
                query = f"financial analysis spending categories budgeting tips transactions: {text_content[:200]}"
                rag_context = rag_service.get_context_for_query(query, max_context_length=3000)  # Increased for more context
                relevant_chunks = rag_service.retrieve_relevant_chunks(query, top_k=10)  # Get more chunks
                rag_chunks_used = len(relevant_chunks)
                
                # Extract all previous statement data for comprehensive analysis
                all_previous_statements = self._get_all_previous_statements_summary()
            
            # Generate temporal analysis with month-to-month comparisons
            summary, suggestions_list, comprehensive_spoken_response = self._generate_temporal_analysis(
                text_content, rag_context, temporal_context, all_previous_statements
            )

            # Save to database
            self.save_llm_interaction(
                prompt=f"Comprehensive analysis of current and all previous statements",
                model=settings.HF_MODEL_NAME,
                raw_response={"comprehensive_analysis": True, "rag_chunks_used": rag_chunks_used},
                generated_text=comprehensive_spoken_response,
                summary=summary,
                suggestions=suggestions_list,
                spoken_response=comprehensive_spoken_response,
                rag_context=rag_context[:500],  # Truncate for storage
                rag_chunks_used=rag_chunks_used
            )

            # Synthesize comprehensive spoken response to MP3
            if comprehensive_spoken_response:
                output_path = "latest_financial_analysis.mp3"  # Fixed filename - always overwrites
                self.synthesize_spoken_response_to_mp3(comprehensive_spoken_response, output_path)

            return FinancialAnalysis(
                summary=summary,
                suggestions=suggestions_list,
                transactions_identified=[],
                spoken_response=comprehensive_spoken_response
            )

        except Exception as e:
            error_msg = f"An unexpected error occurred: {str(e)}"
            print(error_msg)
            
            # Save error to database
            self.save_llm_interaction(
                prompt=f"Comprehensive analysis failed for: {text_content[:100]}",
                model=settings.HF_MODEL_NAME,
                raw_response={"error": str(e)},
                generated_text="",
                summary=error_msg,
                suggestions=[],
                spoken_response="I apologize, but I encountered an error while analyzing your financial statements. Please try again.",
                rag_context="",
                rag_chunks_used=0
            )
            
            return FinancialAnalysis(
                summary=error_msg,
                suggestions=["Please try again or check your input."],
                transactions_identified=[],
                spoken_response="I apologize, but I encountered an error while analyzing your financial statements. Please try again."
            )
    
    def _extract_summary_from_text(self, text: str) -> str:
        """Extract summary from unstructured text response."""
        if not text:
            return "Unable to generate summary"
            
        # Look for lines that contain spending information
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 20 and any(word in line.lower() for word in ['spent', 'spending', 'category', 'categories', 'dining', 'grocery', 'groceries', 'coffee', 'gas', 'netflix', 'amazon']):
                return line[:200]
                
        # Try to extract from the beginning of the response
        first_lines = ' '.join(lines[:3]).strip()
        if len(first_lines) > 20:
            return first_lines[:200]
                
        # Fallback: analyze the input to create a summary
        if 'starbucks' in text.lower() or 'coffee' in text.lower():
            return "You have spending on coffee and dining establishments"
        if 'grocery' in text.lower() or 'whole foods' in text.lower():
            return "You have grocery and food-related expenses"
        if 'gas' in text.lower() or 'shell' in text.lower():
            return "You have transportation and fuel costs"
            
        return "Analysis completed on your financial statement"
    
    def _extract_suggestions_from_text(self, text: str) -> List[str]:
        """Extract suggestions from unstructured text response."""
        suggestions = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(word in line.lower() for word in ['try', 'consider', 'suggest', 'tip', 'budget', 'save', 'plan', 'reduce', 'limit']):
                if len(line) > 10:
                    # Clean up the suggestion
                    cleaned = line.replace('*', '').replace('-', '').replace('•', '').strip()
                    if cleaned and not cleaned.lower().startswith(('you', 'your', 'i')):
                        suggestions.append(cleaned[:150])
                    if len(suggestions) >= 3:
                        break
        
        # Provide smart default suggestions based on common patterns
        if not suggestions:
            suggestions = [
                "Track your spending patterns to identify areas for improvement",
                "Set monthly budgets for major expense categories",
                "Consider reducing discretionary spending on dining and entertainment"
            ]
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _extract_spoken_response_from_text(self, text: str) -> str:
        """Extract spoken response from unstructured text response."""
        if not text:
            return "I've analyzed your financial statement and found some interesting patterns in your spending."
            
        # Look for conversational text that would work well for TTS
        lines = text.split('\n')
        best_line = ""
        
        for line in lines:
            line = line.strip()
            # Look for lines with personal pronouns and reasonable length
            if (len(line) > 30 and len(line) < 200 and 
                any(word in line.lower() for word in ['you', 'your']) and
                any(word in line.lower() for word in ['spend', 'budget', 'money', 'financial', 'expenses'])):
                best_line = line
                break
        
        if not best_line:
            # Create a contextual spoken response based on the generated text
            if any(word in text.lower() for word in ['coffee', 'starbucks', 'dining']):
                best_line = "I noticed you have some dining and coffee expenses. Consider setting a monthly budget for these categories to help manage your spending."
            elif any(word in text.lower() for word in ['grocery', 'food', 'whole foods']):
                best_line = "You have grocery expenses which are essential. Try planning your meals ahead to optimize your food budget."
            elif any(word in text.lower() for word in ['gas', 'fuel', 'transportation']):
                best_line = "Your transportation costs are showing up. Consider tracking your fuel expenses to find potential savings."
            else:
                best_line = "I've reviewed your financial statement and can help you identify patterns in your spending to improve your budget."
        
        return best_line[:300]

    def _get_all_previous_statements_summary(self) -> str:
        """Get a summary of all previous financial statements from RAG system."""
        try:
            # Get all documents from RAG
            documents = rag_service.list_documents()
            
            if not documents:
                return "This is your first financial statement analysis."
            
            # Create summary of previous statements
            summary_parts = []
            summary_parts.append(f"Based on your {len(documents)} previous statement(s):")
            
            # Get spending patterns from previous documents
            patterns_query = "spending categories dining groceries transportation entertainment"
            previous_patterns = rag_service.retrieve_relevant_chunks(patterns_query, top_k=5)
            
            if previous_patterns:
                summary_parts.append("Previous spending patterns include:")
                for i, chunk in enumerate(previous_patterns[:3]):
                    # Extract key spending info from chunk
                    content = chunk.get('content', '')[:200]
                    if any(word in content.lower() for word in ['$', 'restaurant', 'grocery', 'gas', 'amazon']):
                        summary_parts.append(f"- {content}")
            
            return " ".join(summary_parts)
            
        except Exception as e:
            print(f"Error getting previous statements: {e}")
            return "Unable to retrieve previous statement data."
    
    def _generate_comprehensive_analysis(self, current_text: str, rag_context: str, previous_summary: str) -> tuple:
        """Generate comprehensive analysis including current and all previous statements."""
        
        # Extract transaction categories from current statement
        current_categories = self._extract_spending_categories(current_text)
        
        # Create comprehensive summary
        summary = f"Current statement analysis: {self._analyze_current_statement(current_text)}"
        
        # Generate detailed suggestions based on current + historical data
        suggestions = self._generate_comprehensive_suggestions(current_text, rag_context, previous_summary)
        
        # Generate comprehensive spoken response (this is the key enhancement)
        spoken_response = self._generate_comprehensive_spoken_response(
            current_text, rag_context, previous_summary, current_categories
        )
        
        return summary, suggestions, spoken_response
    
    def _extract_spending_categories(self, text: str) -> dict:
        """Extract and categorize spending from statement text."""
        categories = {
            'dining': 0,
            'groceries': 0,
            'transportation': 0,
            'entertainment': 0,
            'shopping': 0,
            'utilities': 0,
            'other': 0
        }
        
        lines = text.split('\n')
        for line in lines:
            line = line.lower()
            
            # Look for dollar amounts and categorize
            import re
            amounts = re.findall(r'\$?(\d+\.?\d*)', line)
            
            for amount_str in amounts:
                try:
                    amount = float(amount_str)
                    if amount > 0:
                        # Categorize based on keywords
                        if any(word in line for word in ['restaurant', 'dining', 'starbucks', 'mcdonald', 'pizza']):
                            categories['dining'] += amount
                        elif any(word in line for word in ['grocery', 'market', 'food', 'walmart', 'target']):
                            categories['groceries'] += amount
                        elif any(word in line for word in ['gas', 'fuel', 'shell', 'exxon', 'uber', 'lyft']):
                            categories['transportation'] += amount
                        elif any(word in line for word in ['netflix', 'spotify', 'amazon prime', 'movie']):
                            categories['entertainment'] += amount
                        elif any(word in line for word in ['amazon', 'shopping', 'store', 'mall']):
                            categories['shopping'] += amount
                        else:
                            categories['other'] += amount
                except ValueError:
                    continue
        
        return categories
    
    def _analyze_current_statement(self, text: str) -> str:
        """Analyze the current statement for key insights."""
        categories = self._extract_spending_categories(text)
        total = sum(categories.values())
        
        if total == 0:
            return "No specific spending amounts detected in this statement."
        
        # Find top spending categories
        sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        top_cats = [cat for cat, amount in sorted_cats if amount > 0][:3]
        
        return f"Total spending: ${total:.2f}, primarily in {', '.join(top_cats)}"
    
    def _generate_comprehensive_suggestions(self, current_text: str, rag_context: str, previous_summary: str) -> list:
        """Generate comprehensive budgeting suggestions based on all available data."""
        suggestions = []
        categories = self._extract_spending_categories(current_text)
        
        # Current statement suggestions
        if categories['dining'] > 100:
            suggestions.append("Your dining expenses are quite high this period. Consider meal planning and cooking at home more often to reduce restaurant spending.")
        
        if categories['transportation'] > 200:
            suggestions.append("Transportation costs are significant. Look into carpooling, public transit, or combining errands to reduce fuel expenses.")
        
        # Historical pattern suggestions
        if rag_context and "previous" in previous_summary.lower():
            suggestions.append("Based on your spending history, consider setting monthly budgets for your top spending categories to maintain better control.")
        
        # Default suggestions
        if not suggestions:
            suggestions.extend([
                "Track your spending patterns across all statements to identify trends and opportunities for savings.",
                "Consider using the 50/30/20 budgeting rule: 50% for needs, 30% for wants, and 20% for savings.",
                "Review your monthly subscriptions and recurring charges to eliminate unused services."
            ])
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _generate_comprehensive_spoken_response(self, current_text: str, rag_context: str, 
                                              previous_summary: str, categories: dict) -> str:
        """Generate a comprehensive spoken response covering current and all previous statements."""
        
        # Start building the comprehensive spoken response
        response_parts = []
        
        # Opening
        response_parts.append("Hello! I've completed a comprehensive analysis of your current financial statement along with all your previous statements.")
        
        # Current statement analysis
        total_current = sum(categories.values())
        if total_current > 0:
            response_parts.append(f"For your current statement, I found total spending of ${total_current:.2f}.")
            
            # Break down by categories
            top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
            spending_breakdown = []
            for cat, amount in top_categories:
                if amount > 0:
                    percentage = (amount / total_current) * 100
                    spending_breakdown.append(f"{cat} at ${amount:.2f} which is {percentage:.0f}% of your total")
            
            if spending_breakdown:
                response_parts.append(f"Your spending breaks down as follows: {'. '.join(spending_breakdown[:3])}.")
        
        # Historical context from previous statements
        if previous_summary and "previous" in previous_summary:
            response_parts.append("Looking at your spending history from all previous statements, I can see some interesting patterns.")
            response_parts.append(previous_summary)
        
        # Spending trends and comparisons
        if rag_context:
            response_parts.append("Comparing this statement to your previous ones, I notice some trends in your spending habits.")
            # Extract key insights from RAG context
            if "restaurant" in rag_context.lower() or "dining" in rag_context.lower():
                response_parts.append("You have a consistent pattern of dining expenses across multiple statements.")
            if "grocery" in rag_context.lower() or "market" in rag_context.lower():
                response_parts.append("Your grocery spending appears regularly, which shows good budgeting for essential food purchases.")
        
        # Detailed recommendations
        response_parts.append("Based on this comprehensive analysis, here are my detailed recommendations:")
        
        if categories['dining'] > categories['groceries'] and categories['dining'] > 50:
            response_parts.append("I notice your restaurant spending exceeds your grocery spending significantly. Consider shifting some of that dining budget toward home cooking. You could save hundreds of dollars per month by preparing just a few more meals at home.")
        
        if categories['transportation'] > 150:
            response_parts.append("Your transportation costs are quite substantial. Look into ways to optimize your travel, such as combining trips, using public transportation when possible, or considering a more fuel-efficient vehicle if you're driving frequently.")
        
        # Long-term financial health advice
        response_parts.append("For your overall financial health, I recommend reviewing these statements monthly to track your progress. Set specific dollar amount goals for each spending category.")
        
        # Positive reinforcement and encouragement
        if total_current > 0:
            response_parts.append("You're doing great by tracking your expenses and seeking analysis. This level of awareness is the first step toward excellent financial management.")
        
        # Closing with action items
        response_parts.append("Moving forward, try to implement one or two of these suggestions this month and track how they impact your next statement. I'm here to help you analyze your progress as you continue building better financial habits.")
        
        # Join all parts into a comprehensive response
        full_response = " ".join(response_parts)
        
        # Ensure the response is substantial (aim for 2-3 minutes of speech)
        if len(full_response) < 800:  # Add more content if too short
            full_response += " Remember, financial wellness is a journey, not a destination. Each statement you analyze brings you closer to your financial goals. Keep up the excellent work in monitoring your spending patterns, and don't hesitate to make adjustments as needed. Your future self will thank you for the financial discipline you're building today."
        
        return full_response

    def _create_financial_analysis_prompt(self, text_content: str, rag_context: str = "") -> str:
        """Create a well-structured prompt for financial analysis that encourages natural, spoken responses."""
        
        # Extract key financial elements from the text
        has_transactions = any(word in text_content.lower() for word in ['$', 'spent', 'charge', 'payment', 'debit', 'credit'])
        has_categories = any(word in text_content.lower() for word in ['restaurant', 'grocery', 'gas', 'amazon', 'netflix', 'starbucks'])
        
        base_prompt = f"""As a friendly financial advisor, analyze this financial statement and provide insights in a conversational, natural speaking tone.

Financial Statement Content:
{text_content[:1500]}

"""
        
        if rag_context:
            base_prompt += f"""Previous Financial Context:
{rag_context[:1000]}

"""
        
        base_prompt += """Please provide a comprehensive analysis that includes:

1. SPENDING SUMMARY: A conversational overview of spending patterns in natural language
2. KEY INSIGHTS: Notable trends, unusual transactions, or patterns worth mentioning
3. PRACTICAL SUGGESTIONS: 3 actionable tips for improving financial health
4. SPOKEN RESPONSE: A natural, conversational summary (2-3 sentences) suitable for text-to-speech

Format your response as natural, flowing text that sounds good when spoken aloud. Use phrases like "I noticed that you spent...", "Your spending shows...", "I'd recommend..." to make it conversational.

Response:"""

        return base_prompt

    def _create_comprehensive_analysis_prompt(self, current_text: str, all_previous: str) -> str:
        """Create a prompt for comprehensive analysis including all previous statements."""
        
        prompt = f"""As a financial advisor, provide a comprehensive analysis covering both current and historical spending patterns.

CURRENT FINANCIAL STATEMENT:
{current_text[:1000]}

ALL PREVIOUS STATEMENTS SUMMARY:
{all_previous[:2000]}

Please analyze:
1. Current statement highlights and spending breakdown
2. Trends and patterns across all statements
3. Comparative analysis (how current spending compares to historical patterns)
4. Comprehensive recommendations based on complete financial picture
5. A detailed spoken summary (3-4 sentences) that covers both current and historical insights

Make the response conversational and suitable for text-to-speech. Use natural language that flows well when spoken.

Comprehensive Analysis:"""

        return prompt

    def _get_temporal_context(self, current_document_date: str = None) -> Dict[str, Any]:
        """Get documents organized by time periods for temporal analysis."""
        try:
            all_docs = rag_service.get_all_documents()
            temporal_context = {
                "current_month": None,
                "previous_months": [],
                "all_months_summary": {},
                "chronological_order": []
            }
            
            documents_with_dates = []
            
            for doc in all_docs:
                metadata = doc.get("metadata", {})
                statement_date = metadata.get("statement_date")
                month_year = metadata.get("month_year")
                
                if statement_date and month_year:
                    try:
                        parsed_date = datetime.fromisoformat(statement_date.replace('Z', '+00:00'))
                        documents_with_dates.append({
                            "document": doc,
                            "date": parsed_date,
                            "month_year": month_year,
                            "filename": doc.get("filename", ""),
                            "statement_date": statement_date
                        })
                    except ValueError:
                        continue
            
            # Sort documents by date (newest first)
            documents_with_dates.sort(key=lambda x: x["date"], reverse=True)
            temporal_context["chronological_order"] = documents_with_dates
            
            # Identify current month (most recent) and previous months
            if documents_with_dates:
                temporal_context["current_month"] = documents_with_dates[0]
                temporal_context["previous_months"] = documents_with_dates[1:]
                
                # Create month summaries
                for doc_info in documents_with_dates:
                    month_year = doc_info["month_year"]
                    if month_year not in temporal_context["all_months_summary"]:
                        temporal_context["all_months_summary"][month_year] = []
                    temporal_context["all_months_summary"][month_year].append(doc_info)
            
            return temporal_context
            
        except Exception as e:
            print(f"Error getting temporal context: {e}")
            return {
                "current_month": None,
                "previous_months": [],
                "all_months_summary": {},
                "chronological_order": []
            }

    def _create_temporal_analysis_prompt(self, text_content: str, temporal_context: Dict[str, Any]) -> str:
        """Create a detailed financial analysis prompt that forces specific financial insights."""
        
        current_month = temporal_context.get("current_month")
        previous_months = temporal_context.get("previous_months", [])
        
        # Create a comprehensive financial analysis prompt
        prompt = f"""FINANCIAL ANALYSIS TASK:
You are a professional financial advisor analyzing spending patterns. Provide specific financial insights, not conversational responses.

FINANCIAL STATEMENT DATA:
{text_content[:800]}

ANALYSIS REQUIREMENTS:
1. CATEGORIZE all spending: dining, groceries, travel, entertainment, shopping, utilities, other
2. CALCULATE approximate total amounts per category (estimate from data)
3. IDENTIFY the top 2-3 spending categories with specific amounts
4. PROVIDE 3 specific financial recommendations based on spending patterns
5. COMPARE to previous months if data available

OUTPUT FORMAT: Provide analysis in this structure:
- Summary: "Total spending was $X across Y categories. Highest: Category ($amount). Notable: specific observation."
- Recommendation 1: Specific action with dollar impact
- Recommendation 2: Specific behavioral change
- Recommendation 3: Future planning advice

FOCUS ON: Actual spending amounts, specific categories, actionable financial advice.
AVOID: Generic responses, conversational filler, non-financial content.

Begin analysis:</prompt>

        return prompt

    def _generate_temporal_analysis(self, text_content: str, rag_context: str, 
                                  temporal_context: Dict[str, Any], all_previous_statements: str) -> tuple:
        """Generate comprehensive temporal analysis with month-to-month comparisons."""
        try:
            # Create temporal analysis prompt (improved version)
            temporal_prompt = self._create_temporal_analysis_prompt(text_content, temporal_context)
            
            # Generate response with increased token limits for detailed analysis
            print("Generating detailed financial analysis...")
            response = self.generate_response(
                temporal_prompt, 
                max_new_tokens=256,  # Increased for more detailed analysis
                temperature=0.3,     # Lower temperature for more focused financial analysis
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            # Extract components from response
            summary = self._extract_summary_from_text(response)
            suggestions = self._extract_suggestions_from_text(response)
            
            # Generate comprehensive spoken response with temporal insights
            spoken_response = self._generate_temporal_spoken_response(
                text_content, temporal_context, summary, suggestions, response
            )
            
            return summary, suggestions, spoken_response
            
        except Exception as e:
            print(f"Error in temporal analysis: {e}")
            # Fallback to basic analysis
            return self._generate_basic_analysis(text_content, rag_context)
    
    def _generate_temporal_spoken_response(self, text_content: str, temporal_context: Dict[str, Any], 
                                         summary: str, suggestions: List[str], full_analysis: str = "") -> str:
        """Generate a comprehensive spoken response that highlights temporal patterns and financial insights."""
        try:
            current_month = temporal_context.get("current_month")
            previous_months = temporal_context.get("previous_months", [])
            
            spoken_parts = []
            
            # Start with current month analysis
            if current_month:
                month_name = current_month.get("month_year", "this month")
                spoken_parts.append(f"I've analyzed your {month_name} financial statement.")
            else:
                spoken_parts.append("I've analyzed your current financial statement.")
            
            # Add the main analysis if available, otherwise use summary
            if full_analysis and len(full_analysis) > 50:
                # Extract the most relevant parts of the financial analysis
                analysis_clean = self._clean_text_for_speech(full_analysis)
                
                # Focus on spending categories and amounts
                if any(keyword in analysis_clean.lower() for keyword in ['spending', 'category', 'total', '$', 'dining', 'groceries']):
                    spoken_parts.append(analysis_clean[:300])
                else:
                    spoken_parts.append(summary[:200])
            elif summary:
                spoken_parts.append(summary[:200])
            
            # Add temporal comparisons if we have previous months
            if previous_months:
                prev_month = previous_months[0]  # Most recent previous month
                prev_month_name = prev_month.get("month_year", "last month")
                
                # Create comparison text
                comparison_text = f"Compared to {prev_month_name}, I can see patterns in your spending habits. "
                spoken_parts.append(comparison_text)
            
            # Add most relevant suggestion with financial focus
            if suggestions:
                # Find the most specific financial suggestion
                best_suggestion = suggestions[0]
                for suggestion in suggestions:
                    if any(keyword in suggestion.lower() for keyword in ['budget', 'save', 'reduce', 'track', '$']):
                        best_suggestion = suggestion
                        break
                
                spoken_parts.append(f"My recommendation: {best_suggestion}")
            
            # Combine all parts
            full_response = " ".join(spoken_parts)
            
            # Clean and limit length for natural speech
            full_response = full_response[:600]  # Increased length for comprehensive analysis
            full_response = self._clean_text_for_speech(full_response)
            
            # Ensure it ends with a complete sentence
            sentences = full_response.split('. ')
            if len(sentences) > 1 and not full_response.endswith('.'):
                full_response = '. '.join(sentences[:-1]) + '.'
            
            return full_response
            
        except Exception as e:
            print(f"Error generating temporal spoken response: {e}")
            return "I've analyzed your financial statement and noticed some interesting patterns in your spending."

    def _generate_basic_analysis(self, text_content: str, rag_context: str) -> tuple:
        """Fallback method for basic analysis when temporal analysis fails."""
        try:
            basic_prompt = self._create_financial_analysis_prompt(text_content, rag_context)
            response = self.generate_response(basic_prompt, max_new_tokens=400, temperature=0.7)
            
            summary = self._extract_summary_from_text(response)
            suggestions = self._extract_suggestions_from_text(response)
            spoken_response = f"I've analyzed your financial statement. {summary[:200]}"
            
            return summary, suggestions, spoken_response
            
        except Exception as e:
            print(f"Error in basic analysis: {e}")
            return (
                "Unable to generate financial analysis", 
                ["Please try again or check your input."], 
                "I apologize, but I encountered an error while analyzing your financial statements."
            )

# Global instance
hf_llm_service = HuggingFaceLLMService()

# For backward compatibility, create an alias function
async def analyze_text_with_llm(text_content: str) -> FinancialAnalysis:
    """Backward compatible function that uses the new HF service."""
    return await hf_llm_service.analyze_text_with_llm_and_rag(text_content, use_rag=True)
