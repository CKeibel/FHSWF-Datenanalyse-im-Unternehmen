import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


class EncoderModel:
    """Encoder class to encode documents and queries into a vector space."""
    def __init__(self, model_id, device=None):
        self.model_id = model_id
        self.device = device if device else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id, device=self.device)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def tokenize(self, document):
        return self.tokenizer(document, padding=True, truncation=True, return_tensors="pt").to(device)

    def model_inference(self, tokenized_document):
        outputs = outputs = self.model(**tokenized_document)
        return outputs
    
    def __call__(self, document):
        tokenized_document = self.tokenize(document)
        outputs = self.model_inference(tokenized_document)
        embeddings = self.mean_pooling(outputs, tokenized_document["attention_mask"])
        return F.normalize(embeddings, p=2, dim=1).cpu().numpy()

class DecoderModel:
    """Causal language model to generate text based on a prompt."""
    def __init__(self, model_id, generation_config=None, device=None, **kwargs):
        self.model_id = model_id
        self.device = device if device else "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # TODO: generation config
        self.generation_config=generation_config if generation_config else {}
        # TODO: jinja template

    def tokenize(self, prompt):
        _device = "cuda" if self.device == "auto" else self.device
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt").to(_device)
        return tokenized_prompt

    def __call__(self, prompt):
        tokenized_prompt = self.tokenize(prompt)
        outputs = self.model.generate(**tokenized_prompt, max_new_tokens=50) # TODO: generation config
        return self.tokenizer.decode(outputs[0][len(tokenized_prompt.input_ids[0]):])

    @staticmethod
    def construct_rag_prompt(question, context):
        pass