import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from app.core.config import Settings

settings = Settings()

class Llama3Service:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.llama_model,
            token=settings.hf_token,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            settings.llama_model,
            token=settings.hf_token,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=settings.max_tokens,
            temperature=settings.temperature,
            do_sample=True,
            return_full_text=False,
        )

    def decide(self, context: str, query: str) -> str:
        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "You are an expert insurance assistant. Answer only with a valid JSON object containing:\n"
            '{"decision":"<approved|rejected|requires_review>","amount":<number|null>,"justification":"<string>"}\n'
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"Policy excerpts:\n{context}\n\nQuery: {query}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
        return self.pipe(prompt)[0]["generated_text"].strip()