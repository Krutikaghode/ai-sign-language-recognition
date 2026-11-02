import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class HuggingFaceGlossLLM:
    """
    LLM Engine using Hugging Face transformers (google/flan-t5-base)
    to translate ASL Gloss sequences and RAG context into fluent English sentences
    with intent classification and suggested replies.
    """
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate_translation(self, prompt: str, max_length: int = 128) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def process(self, gloss_sequence: str, rag_context: list = None) -> dict:
        context_str = ""
        if rag_context:
            context_str = " Context: " + "; ".join([item.get('meaning', '') for item in rag_context if isinstance(item, dict)])
        
        prompt = f"Translate the ASL gloss sequence '{gloss_sequence}' into fluent English.{context_str}"
        translated_text = self.generate_translation(prompt)
        
        intent_prompt = f"Classify the intent of this phrase: '{translated_text}'. Options: Greeting, Question, Emergency/Request, Statement."
        intent = self.generate_translation(intent_prompt)
        
        reply_prompt = f"Provide two brief helpful replies to: '{translated_text}'"
        reply_text = self.generate_translation(reply_prompt)
        suggested_replies = [r.strip() for r in reply_text.split(".") if r.strip()]
        if not suggested_replies:
            suggested_replies = ["How can I help you?", "Understood."]

        return {
            "gloss_sequence": gloss_sequence,
            "translated_text": translated_text,
            "intent": intent,
            "suggested_replies": suggested_replies
        }

if __name__ == "__main__":
    llm = HuggingFaceGlossLLM()
    res = llm.process("NAME YOU WHAT", rag_context=[{"meaning": "Asking for someone's name."}])
    print(res)
