import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

class ContextualIdiomReplacer:
    def __init__(self, model_name='EleutherAI/gpt-neo-1.3B'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPTNeoForCausalLM.from_pretrained(model_name)
    
    def generate_idiom(self, context):
        prompt = f"Provide a relevant idiom for this context: '{context}'"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=50,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            no_repeat_ngram_size=2
        )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the idiom from the generated text
        idiom = generated_text.replace(prompt, "").strip()
        return idiom
    
    def replace_with_idiom(self, sentence):
        # Try to extract a context (basic implementation)
        words_to_replace = ["happy", "sad", "angry", "excited"]  # Extend as needed
        idiom_found = False

        for word in words_to_replace:
            if word in sentence:
                idiom = self.generate_idiom(word)
                modified_sentence = sentence.replace(word, idiom, 1)
                idiom_found = True
                break

        if not idiom_found:  # If no specific emotion is found, generate a general idiom
            idiom = self.generate_idiom(sentence)
            modified_sentence = f"{sentence} ({idiom})"
        
        return modified_sentence

def main():
    replacer = ContextualIdiomReplacer()
    
    while True:
        user_input = input("Enter a sentence or context (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        result = replacer.replace_with_idiom(user_input)
        print(f"Modified Sentence: {result}\n")

if __name__ == "__main__":
    main()
