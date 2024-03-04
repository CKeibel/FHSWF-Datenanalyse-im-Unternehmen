import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from model import BM25Model, EncoderModel


class VectorStore:
    """VectorStore class that enables a hybrid search."""
    def __init__(self, model_id, hybrid=False, weight=0.5, distance_metric=None, device=None):
        """Initialize VectorStore
        
        Args:
            model_id: string describing huggingface model to use for retrieval
            hybrid: boolean that determines the search mode (hybrid or dense)
            weight: float between 0 and 1 that determines how much each search score should
                influence the final result
            distance_metric: string that determines how the dense search space should be used
                (cosine, l2, ip)
        
        Note:
            If hybrid is set to True, a BM25 model is instantiated.
        """
        self.encoder = EncoderModel(model_id, device)
        self.bm25 = BM25Model() if hybrid else None
        self.documents = pd.DataFrame(columns=["id", "text", "vector"])
        self.distance_metric = distance_metric
        self.weight = weight
        
    def reset(self):
        self.documents = pd.DataFrame(columns=["id", "text", "vector"])
        
    def normalize_scores(self, scores):
        """Function to normalize search scores.
        
        Args:
            scores: scores from a search result
            
        Returns:
            Normalized scores
        """
        return normalize([scores])[-1]
    
    def merge_results(self, dense_results, sparse_results):
        """Function to merge search scores from hybrid search.
        
        Args:
            dense_results: scores from dense search
            sparse_results: scores from sparse search
            
        Returns:
            Hybrid results if hybrid is set to True, else dense search scores
        """
        
        # return dense dense scores if hybrid is False (therefore sparse_results is None)
        if sparse_results is None:
            return dense_results
        
        # get unique keys from both scores as they may not be identical
        merged_keys = set(list(dense_results.keys()) + list(sparse_results.keys()))
        
        # calculate hybrid scores
        hybrid_results = dict()
        for key in merged_keys:
            # if a key is not found in a search result, set its score to 0
            dense_score =  0.0 if dense_results.get(key) is None else dense_results.get(key)
            sparse_score =  0.0 if sparse_results.get(key) is None else sparse_results.get(key)
            # claculate and save hybrid score
            hybrid_score = (1 - self.weight) * sparse_score + self.weight * dense_score
            hybrid_results[key] = hybrid_score
        return hybrid_results
        
    def search(self, query, top_n=10):
        """Function to search the vector store.
        
        Args:
            query: string query
            top_n: int to determine how much documents should be returned
            
        Returns:
            A list of dictionaries structured as follows:
                {
                    score: float,
                    document: string,
                    id: int
                }
                
        Note:
            This function should be used to perform the search on the vector store.
            It calls function for dense and sparse search.
        """
        # embed query and call dense search
        vectorized_query = self.encoder(query)
        dense_results = self.dense_search(list(vectorized_query), top_n)
        
        # initialize sparse results to None, if hybrid is set to True, sparse search is called
        sparse_results = None
        if self.bm25:
            sparse_results = self.sparse_search(query, top_n)
        
        # merge both search results
        hybrid_results = self.merge_results(dense_results, sparse_results)
        result = [
            {
                "score": score, 
                "document": self.documents.text.values[key], 
                "id": self.documents.id.values[key]
            } for key, score in hybrid_results.items()
        ]
        # return top_n results in descending order 
        return sorted(result, key=lambda x: x["score"], reverse=True)[:top_n]
    
    def sparse_search(self, query, top_n):
        """Sparse search.
        
        Args:
            query: string query
            top_n: int to determine how much documents should be returned
            
        Returns:
            A dictionary strucutred as follows:
                {
                    document id 1: search score 1,
                    ...
                    document id n: search score n,
                }
        """
        
        # call sparse search
        scores = self.bm25.search(query)
        # extract ids of the top_n documents
        idxs = np.argpartition(scores, -top_n)[-top_n:]
        scores = scores[idxs]
        # normalize scores
        scores = self.normalize_scores(scores)
        return {idx: score for idx, score in zip(idxs, scores)}       
    
    def dense_search(self, query, top_n):
        """Dense search.
        
        Args:
            query: string query
            top_n: int to determine how much documents should be returned
            
        Returns:
            A dictionary strucutred as follows:
                {
                    document id 1: search score 1,
                    ...
                    document id n: search score n,
                }
        """
        try:
            assert top_n <= self.documents.shape[0]
        except AssertionError:
            top_n = self.documents.shape[0]
        
        # cast columns vectors to list 
        # (otherwise type mismatch between query(list) and vectors (np.ndarray))
        vectors = self.documents.vector.values.tolist()
        
        # calculate scores with specified distance (self.distance_metric)
        if self.distance_metric == "ip":
            # extract ids of the top_n documents
            scores = np.inner(query, vectors)[-1]
            idxs = np.argpartition(scores, -top_n)[-top_n:]
        elif self.distance_metric == "l2":
            # get absolute distances
            scores = abs(np.linalg.norm(np.array(query)-np.array(vectors), axis=1))
            
            # extract ids of the top_n documents
            if top_n == self.documents.shape[0]:
                idxs = np.argpartition(scores, top_n-1)[:top_n]
            else:
                idxs = np.argpartition(scores, top_n)[:top_n]
        else:
            # extract ids of the top_n documents
            scores = cosine_similarity(query, vectors)[-1]
            idxs = np.argpartition(scores, -top_n)[-top_n:]
        
        scores = scores[idxs]
        # normalize scores
        scores = self.normalize_scores(scores)
        # substract distance scores from one
        if self.distance_metric == "l2":
            scores = np.subtract(1, scores)
        return {idx: score for idx, score in zip(idxs, scores)}
    
    def batch_documents(self, documents, ids, batch_size):
        for i in range(0, len(documents), batch_size):
            yield {"documents": documents[i:i+batch_size], "ids": ids[i:i+batch_size]}
                
    def add_documents(self, documents: list[str], ids=None, batch_size=None):
        """Add documents to vector store.
        
        Args:
            documents: list of documents
            ids: int document ids
            
        Note:
            Creates word corpus for bm25 if hybrid is set to True.
        """
        batch_size = batch_size if batch_size else len(documents)
        
        # generate ids if not given
        if ids is None:
            ids = [i + len(self.documents) for i, _ in enumerate(documents)]
            
        # embed documents into dense vector space
        batched_documents = self.batch_documents(documents, ids, batch_size)
        
        for batch in batched_documents:
            vectors = self.encoder(batch["documents"])
            
            # store new documents to pd.DataFrame
            new_data = pd.DataFrame(
                        {
                            "id": batch["ids"],
                            "text": batch["documents"],
                            "vector": vectors.tolist()
                        }
                    )
            # concat new_data and old documents
            self.documents = pd.concat([self.documents, new_data])
        
        # generate word corpus for sparse search
        if self.bm25:
            self.bm25.add_documents(self.documents.text.values.tolist())