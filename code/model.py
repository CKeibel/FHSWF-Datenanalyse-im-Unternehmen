import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, GenerationConfig


class EncoderModel:
    """Encoder class to encode documents and queries into a vector space."""
    def __init__(self, model_id, device=None):
        self.model_id = model_id
        self.device = device if device else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def tokenize(self, document):
        return self.tokenizer(document, padding=True, truncation=True, return_tensors="pt").to(self.device)

    def model_inference(self, tokenized_document):
        with torch.no_grad():
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
        
        assert isinstance(generation_config, GenerationConfig) or generation_config is None
        self.model_id = model_id
        self.device = device if device else "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.chat_template = self.get_jinja_tempalte()
        self.generation_config=generation_config if generation_config else self.default_generation_config()
        
    def get_jinja_tempalte(self):
        rag_template = "{% for message in messages %}\n" \
                            "{% if message['role'] == 'context' %}\n"\
                                "{{ '<|context|>\n' + message['content'] + eos_token }}\n"\
                            "{% elif message['role'] == 'system' %}\n"\
                                "{{ '<|system|>\n' + message['content'] + eos_token }}\n"\
                            "{% elif message['role'] == 'question' %}\n"\
                                "{{ '<|question|>\n' + message['content'] + eos_token }}\n"\
                            "{% elif message['role'] == 'assistant' %}\n"\
                                "{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n"\
                            "{% endif %}\n"\
                            "{% if loop.last and add_generation_prompt %}\n"\
                                "{{ '<|assistant|>' }}\n"\
                            "{% endif %}\n" \
                        "{% endfor %}"
        return rag_template
        
    def default_generation_config(self):
        gen_cfg = GenerationConfig.from_pretrained(self.model_id)
        gen_cfg.max_new_tokens = 150
        gen_cfg.pad_token_id = self.tokenizer.pad_token_id
        gen_cfg.begin_suppress_tokensrepetition_penalty = 5
        gen_cfg.no_repeat_ngram_size = 3
        return gen_cfg

    def tokenize(self, prompt):
        _device = "cuda" if self.device == "auto" else self.device
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt").to(_device)
        return tokenized_prompt
    
    def construct_rag_prompt(self, question, context):
        messages = [
            {"role": "system", "content": "Answer the question only based on the given context."},
            {"role": "context", "content": context},
            {"role": "question", "content": question}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def __call__(self, question, context):
        prompt = self.construct_rag_prompt(question, context)
        tokenized_prompt = self.tokenize(prompt)
        outputs = self.model.generate(**tokenized_prompt, generation_config=self.generation_config) # TODO: generation config
        return self.tokenizer.decode(outputs[0][len(tokenized_prompt.input_ids[0]):])
    
class BM25Model:
    # save index
    # save indx2doc mapping
