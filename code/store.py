import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from model import BM25Model, EncoderModel


class VectorStore:
    def __init__(self, model_id, hybrid=False, weight=0.5, distance_metric=None):
        self.encoder = EncoderModel(model_id)
        self.bm25 = BM25Model() if hybrid else None
        self.documents = pd.DataFrame(columns=["id", "text", "vector"])
        self.distance_metric = distance_metric
        self.weight = weight
        
    def normalize_scores(self, scores):
        return normalize([scores])[-1]
    
    def merge_results(self, dense_results, sparse_results):
        if sparse_results is None:
            return dense_results
        merged_keys = set(list(dense_results.keys()) + list(sparse_results.keys()))
        hybrid_results = dict()
        for key in merged_keys:
            dense_score =  0.0 if dense_results.get(key) is None else dense_results.get(key)
            sparse_score =  0.0 if sparse_results.get(key) is None else sparse_results.get(key)
            hybrid_score = (1 - self.weight) * sparse_score + self.weight * dense_score
            hybrid_results[key] = hybrid_score
        return hybrid_results
        
    def search(self, query, top_n=10):
        vectorized_query = self.encoder(query)
        dense_results = self.dense_search(list(vectorized_query), top_n)
        
        sparse_results = None
        if self.bm25:
            sparse_results = self.sparse_search(query, top_n)
        hybrid_results = self.merge_results(dense_results, sparse_results)
        result = [
            {
                "score": score, 
                "document": self.documents.text.values[key], 
                "id": self.documents.id.values[key]
            } for key, score in hybrid_results.items()
        ]
        return sorted(result, key=lambda x: x["score"], reverse=True)[:top_n]
    
    def sparse_search(self, query, top_n):
        scores = self.bm25.search(query)
        idxs = np.argpartition(scores, -top_n)[-top_n:]
        scores = scores[idxs]
        scores = self.normalize_scores(scores)
        return {idx: score for idx, score in zip(idxs, scores)}       
    
    def dense_search(self, query, top_n):
        try:
            assert top_n <= self.documents.shape[0]
        except AssertionError:
            top_n = self.documents.shape[0]
            
        vectors = self.documents.vector.values.tolist()
            
        if self.distance_metric == "ip":
            scores = np.inner(query, vectors)[-1]
            idxs = np.argpartition(scores, -top_n)[-top_n:]
        elif self.distance_metric == "l2":
            scores = abs(np.linalg.norm(np.array(query)-np.array(vectors), axis=1))
            if top_n == self.documents.shape[0]:
                idxs = np.argpartition(scores, top_n-1)[:top_n]
            else:
                idxs = np.argpartition(scores, top_n)[:top_n]
        else:
            scores = cosine_similarity(query, vectors)[-1]
            idxs = np.argpartition(scores, -top_n)[-top_n:]
        
        scores = scores[idxs]
        scores = self.normalize_scores(scores)
        if self.distance_metric == "l2":
            scores = np.subtract(1, scores)
        return {idx: score for idx, score in zip(idxs, scores)}
                
    def add_documents(self, documents: list[str], ids=None):
        vectors = self.encoder(documents)
        if ids is None:
            ids = [i + len(self.documents) for i, _ in enumerate(documents)]
        new_data = pd.DataFrame(
                    {
                        "id": ids,
                        "text": documents,
                        "vector": vectors.tolist()
                    }
                )
        self.documents = pd.concat([self.documents, new_data])
        
        if self.bm25:
            self.bm25.add_documents(self.documents.text.values.tolist())