from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 🔎 1) Model Selection:
# We're using "microsoft/phi-2", a small but capable open-source LLM.
# It's a decoder-only transformer model designed for text generation.
MODEL_NAME = "microsoft/phi-2"

# 🔎 2) Load Tokenizer: converts text -> tokens (numerical IDs) for the model.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 🔎 3) Load Model: loads the pre-trained weights.
# device_map="auto" lets Hugging Face choose CPU/GPU automatically.
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

# 🔎 4) Create a text generation pipeline:
# The pipeline combines tokenizer + model for easier usage.
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 🔎 5) Summarizer function you will call from FastAPI:
def summarize_feedback(text: str, max_tokens: int = 200) -> str:
    """
    Summarize a single piece of feedback text.
    Returns the generated summary as a string.
    """
    prompt = f"Summarize this customer feedback: {text}"
    outputs = generator(prompt, max_length=max_tokens, temperature=0.7)
    return outputs[0]['generated_text']
