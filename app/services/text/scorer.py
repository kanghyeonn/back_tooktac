from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

class SimilarityScorer:
    def __init__(self):
        self.model_names = [
            'paraphrase-multilingual-mpnet-base-v2',
            'snunlp/KR-SBERT-V40K-klueNLI-augSTS'
        ]
        self.weights = [0.7, 0.3]
        self.models = self._load_models()

    def _load_models(self):
        models = {}
        for name in self.model_names:
            if "snunlp" in name:
                tokenizer = AutoTokenizer.from_pretrained(name)
                model = AutoModel.from_pretrained(name)
                models[name] = {"type": "huggingface", "model": model, "tokenizer": tokenizer}
            else:
                model = SentenceTransformer(name)
                models[name] = {"type": "sentence-transformers", "model": model}
        return models

    def _embed(self, model_dict, sentence: str):
        if model_dict["type"] == "huggingface":
            inputs = model_dict["tokenizer"](sentence, return_tensors="pt", truncation=True, padding=True,
                                             max_length=128)
            with torch.no_grad():
                outputs = model_dict["model"](**inputs)
                embedding = outputs.pooler_output[0]
        else:
            embedding = model_dict["model"].encode(sentence, convert_to_tensor=True)
        return F.normalize(embedding, p=2, dim=0)

    def calculate_similarity(self, user_answer: str, model_answer: str) -> float:
        total = 0.0
        for name, weight in zip(self.model_names, self.weights):
            model_dict = self.models[name]
            emb1 = self._embed(model_dict, user_answer)
            emb2 = self._embed(model_dict, model_answer)
            sim = torch.dot(emb1, emb2).item()
            total += sim * weight
        return round(max(0.0, min(1.0, total)), 4)