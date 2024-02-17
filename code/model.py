import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


class EncoderModel:
    """Encoder class to encode documents and queries into a vector space."""
    def __init__(self, model_id, device):
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id, device=self.device)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def tokenize(self, document):
        return self.tokenizer(document, padding=True, truncation=True, return_tensors="pt").to(device)

    def __call__(self, document):
        tokenized_document = self.tokenize(document)
        outputs = self.model(**tokenized_document)


class DecoderModel:
    """Causal language model to generate text based on a prompt."""
    pass