# import re
# import spacy
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# import requests  # For Groq API fallback
# import csv

# class ContextualSuggestionGenerator:
#     def __init__(self):
#         try:
#             self.nlp = spacy.load("en_core_web_sm")  # Ensure SpaCy is installed
#             print(f"SpaCy model loaded successfully: {self.nlp}")
#         except Exception as e:
#             print(f"Error loading SpaCy model: {e}")
#             raise e

#         # Priority 1: Load FLAN-T5 for paraphrasing
#         try:
#             self.paraphrase_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small", legacy=False)
#             self.paraphrase_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
#             print("FLAN-T5 paraphrasing model loaded successfully!")
#         except Exception as e:
#             print(f"Error loading FLAN-T5: {e}")
#             self.paraphrase_tokenizer = None
#             self.paraphrase_model = None

#         # Priority 2: Set up Groq API for LLaMA
#         self.groq_api_url = "https://api.groq.ai/v1/models/llama-3.3-70b-versatile"
#         self.api_key = "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"  # Replace with your actual key

#         # Priority 3: Load CSV-based fallback suggestions
#         self.fallback_suggestions = {}
#         try:
#             with open("paraphrase_fallback.csv", "r") as csvfile:
#                 reader = csv.reader(csvfile)
#                 next(reader)  # Skip header
#                 for row in reader:
#                     original, rephrased = row
#                     self.fallback_suggestions[original.lower()] = rephrased
#         except FileNotFoundError:
#             print("Fallback CSV not found. Using default suggestions.")

#     def _generate_rephrased_sentence(self, sentence):
#         # Attempt 1: Use FLAN-T5
#         if self.paraphrase_model and self.paraphrase_tokenizer:
#             try:
#                 input_text = f"paraphrase: {sentence.strip()}"
#                 input_ids = self.paraphrase_tokenizer.encode(input_text, return_tensors="pt", truncation=True)
#                 outputs = self.paraphrase_model.generate(
#                     input_ids,
#                     max_length=100,
#                     num_beams=5,
#                     early_stopping=True
#                 )
#                 rephrased = self.paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
#                 return rephrased
#             except Exception as e:
#                 print(f"FLAN-T5 failed to generate rephrasing: {e}")

#         # Attempt 2: Use LLaMA via Groq API
#         try:
#             headers = {
#                 "Authorization": f"Bearer {self.api_key}",
#                 "Content-Type": "application/json"
#             }
#             payload = {
#                 "prompt": f"Paraphrase: {sentence}",
#                 "max_tokens": 50
#             }
#             response = requests.post(self.groq_api_url, headers=headers, json=payload)
#             if response.status_code == 200:
#                 rephrased = response.json()["choices"][0]["text"].strip()
#                 return rephrased
#             else:
#                 print(f"LLaMA API error: {response.status_code} - {response.text}")
#         except Exception as e:
#             print(f"LLaMA API failed: {e}")

#         # Final Fallback: Predefined suggestions from CSV
#         return self._create_alternative_suggestion(sentence)

#     def _create_alternative_suggestion(self, original_sentence):
#         fallback = {
#             "technology is very important in today life": "Technology plays a crucial role in modern life.",
#             "it help peoples to make work easy": "It helps people simplify their work.",
#             "many person do not have access": "Many people lack access.",
#             "make difficult for them": "Makes it difficult for them.",
#             "model life": "Modern life."
#         }
#         return fallback.get(original_sentence.lower(), "Rephrase for clarity.")

#     def generate_specific_suggestions(self, original_sentence, corrected_sentence):
#         rephrased = self._generate_rephrased_sentence(original_sentence)
#         return f"Suggestion: Consider rephrasing to '{rephrased}'."

# class GrammarAnalyzer:
#     def __init__(self):
#         print("Loading grammar correction model...")
#         self.model_name = 'prithivida/grammar_error_correcter_v1'

#         try:
#             self.tokenizer = T5Tokenizer.from_pretrained(
#                 self.model_name,
#                 legacy=False,
#                 add_prefix_space=True
#             )
#             self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
#             self.suggestion_generator = ContextualSuggestionGenerator()
#             print("All components loaded successfully!")
#         except Exception as e:
#             print(f"Error loading grammar correction model: {e}")
#             raise e

#     def analyze_text(self, text):
#         sentences = re.split(r'(?<=[.!?])\s+', text.strip())
#         results = []

#         for sentence in sentences:
#             if not sentence.strip():
#                 continue

#             corrected = self._correct_sentence(sentence)
#             suggestion = self.suggestion_generator.generate_specific_suggestions(
#                 sentence.strip(),
#                 corrected.strip()
#             )

#             results.append({
#                 "original_text": sentence.strip(),
#                 "corrected_text": corrected.strip(),
#                 "suggestion": suggestion
#             })

#         return results

#     def _correct_sentence(self, sentence):
#         input_text = f"grammar correction: {sentence.strip()}"
#         input_ids = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True)

#         try:
#             outputs = self.model.generate(input_ids, max_length=100)
#             corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

#             # Remove prefixes and extra_id tokens
#             corrected = corrected.replace("grammar correction:", "").strip()
#             corrected = re.sub(r'<extra_id_\d+>', '', corrected).strip()

#             return corrected
#         except Exception as e:
#             print(f"Grammar correction failed: {e}")
#             return sentence  # Fallback to original sentence

# import re
# import spacy
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# import requests
# import csv

# class ContextualSuggestionGenerator:
#     def __init__(self):
#         try:
#             self.nlp = spacy.load("en_core_web_sm")
#             print(f"SpaCy model loaded successfully: {self.nlp}")
#         except Exception as e:
#             print(f"Error loading SpaCy model: {e}")
#             raise e

#         # Groq API configuration with hardcoded key
#         self.groq_api_url = "https://api.groq.ai/v1/generate"
#         self.api_key = "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"

#         # CSV-based fallback suggestions
#         self.fallback_suggestions = {}
#         try:
#             with open("paraphrase_fallback.csv", "r") as csvfile:
#                 reader = csv.reader(csvfile)
#                 next(reader)  # Skip header
#                 for row in reader:
#                     original, rephrased = row
#                     self.fallback_suggestions[original.lower()] = rephrased
#         except FileNotFoundError:
#             print("Fallback CSV not found. Using default suggestions.")

#     def generate_suggestions(self, corrected_sentence):
#         """Generate both formal and casual paraphrases"""
#         enhanced = self._paraphrase(corrected_sentence, style="formal")
#         native = self._paraphrase(corrected_sentence, style="casual")
#         return {
#             "enhanced": enhanced,
#             "native": native
#         }

#     def _paraphrase(self, sentence, style="formal"):
#         """Generate single paraphrase using LLaMA API"""
#         try:
#             headers = {
#                 "Authorization": f"Bearer {self.api_key}",
#                 "Content-Type": "application/json"
#             }
#             prompt = f"Paraphrase in {style} style: {sentence}\nOutput only the paraphrased version."
            
#             response = requests.post(self.groq_api_url, headers=headers, json={
#                 "prompt": prompt,
#                 "max_tokens": 50,
#                 "temperature": 0.7,
#                 "model": "llama-3.3-70b-versatile"
#             })
            
#             if response.status_code == 200:
#                 return response.json()["choices"][0]["text"].strip()
#             print(f"API Error: {response.status_code}")
#         except Exception as e:
#             print(f"Paraphrase failed: {e}")
        
#         return self._fallback_paraphrase(sentence, style)

#     def _fallback_paraphrase(self, sentence, style):
#         """CSV-based fallback with style awareness"""
#         key = sentence.lower()
#         if key in self.fallback_suggestions:
#             return self.fallback_suggestions[key]
        
#         # Default fallback patterns
#         return {
#             "formal": f"Formal version of: {sentence}",
#             "casual": f"Casual version of: {sentence}"
#         }.get(style, "Rephrase suggestion unavailable")

# class GrammarAnalyzer:
#     def __init__(self):
#         print("Loading grammar correction model...")
#         self.model_name = 'prithivida/grammar_error_correcter_v1'

#         try:
#             self.tokenizer = T5Tokenizer.from_pretrained(
#                 self.model_name,
#                 legacy=False,
#                 add_prefix_space=True
#             )
#             self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
#             self.suggestion_generator = ContextualSuggestionGenerator()
#             print("All components loaded successfully!")
#         except Exception as e:
#             print(f"Error loading grammar correction model: {e}")
#             raise e

#     def analyze_text(self, text):
#         sentences = re.split(r'(?<=[.!?])\s+', text.strip())
#         results = []

#         for sentence in sentences:
#             if not sentence.strip():
#                 continue

#             original = sentence.strip()
#             corrected = self._correct_sentence(original)
            
#             # Only generate suggestions for corrected sentences
#             suggestions = None
#             if corrected != original:
#                 suggestions = self.suggestion_generator.generate_suggestions(corrected)

#             results.append({
#                 "original_text": original,
#                 "corrected_text": corrected,
#                 "suggestions": suggestions
#             })

#         return results

#     def _correct_sentence(self, sentence):
#         input_text = f"grammar correction: {sentence.strip()}"
#         input_ids = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True)

#         try:
#             outputs = self.model.generate(input_ids, max_length=100)
#             corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
#             corrected = corrected.replace("grammar correction:", "").strip()
#             corrected = re.sub(r'<extra_id_\d+>', '', corrected).strip()
#             return corrected
#         except Exception as e:
#             print(f"Grammar correction failed: {e}")
#             return sentence


import re
import spacy
from transformers import T5Tokenizer, T5ForConditionalGeneration
from groq import Groq  # Ensure this library is installed

class ContextualSuggestionGenerator:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.client = Groq(api_key="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")
        self.model = "llama-3.3-70b-versatile"

    def generate_suggestions(self, corrected_sentence):
        return {
            "enhanced": self._paraphrase(corrected_sentence, style="enhanced"),
            "native": self._paraphrase(corrected_sentence, style="native")
        }

    def _paraphrase(self, sentence, style):
        max_retries = 3
        prompt = f"""
        You are an expert editor. Paraphrase the following sentence in a {style} style:
        "{sentence}"
        
        Instructions:
        - Maintain the original meaning without adding or removing information.
        - Provide a clear, concise, and natural paraphrase.
        - For 'enhanced' style, use refined vocabulary and a polished tone.
        - For 'native' style, use everyday language that sounds authentic and conversational.
        - Keep the sentence short and to the point.
        - Return only the paraphrased sentence, with no additional comments.
        """
        
        for _ in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error during paraphrasing attempt: {e}")
        return "Error: Unable to generate paraphrase."

class GrammarAnalyzer:
    def __init__(self):
        # Load SpaCy for sentence splitting
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load grammar correction model
        self.tokenizer = T5Tokenizer.from_pretrained(
            "prithivida/grammar_error_correcter_v1",
            legacy=False,
            add_prefix_space=True
        )
        self.model = T5ForConditionalGeneration.from_pretrained("prithivida/grammar_error_correcter_v1")
        
        # Initialize suggestion generator
        self.suggestion_generator = ContextualSuggestionGenerator()

    def analyze_text(self, text):
        doc = self.nlp(text)
        results = []
        
        for sent in doc.sents:
            original = sent.text.strip()
            corrected = self._correct_sentence(original)
            
            if corrected == original:
                results.append({
                    "original_text": original,
                    "corrected_text": corrected,
                    "has_error": False
                })
                continue
            
            suggestions = self.suggestion_generator.generate_suggestions(corrected)
            
            results.append({
                "original_text": original,
                "corrected_text": corrected,
                "has_error": True,
                "suggestions": suggestions
            })
        
        return results

    def _correct_sentence(self, sentence):
        input_text = f"grammar correction: {sentence.strip()}"
        try:
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True)
            outputs = self.model.generate(input_ids, max_length=100)
            corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            corrected = corrected.replace("grammar correction:", "").strip()
            corrected = re.sub(r'<extra_id_\d+>', '', corrected).strip()
            return corrected
        except Exception as e:
            print(f"Grammar correction failed: {e}")
            return sentence