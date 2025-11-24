from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pymongo

class HybridSearchEngine:
    def __init__(self, mongodb_uri="mongodb://localhost:27017/", db_name="nips_papers"):
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[db_name]
        self.collection = self.db["papers_2024"]
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
       
        self.paper_embeddings = self.precompute_embeddings()
    
    def precompute_embeddings(self):
        """Precompute embeddings for all papers for faster search"""
        papers = list(self.collection.find({}, {'title': 1, 'authors': 1, 'keywords': 1}))
        embeddings = {}
        
        for paper in papers:
            text = f"{paper['title']} {' '.join(paper.get('authors', []))} {' '.join(paper.get('keywords', []))}"
            embedding = self.model.encode([text])[0]
            embeddings[paper['_id']] = embedding
        
        return embeddings
    
    def semantic_search(self, query, top_k=10):
        """Semantic search using embeddings"""
        query_embedding = self.model.encode([query])[0]
        
        similarities = {}
        for paper_id, embedding in self.paper_embeddings.items():
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            similarities[paper_id] = similarity
        
        sorted_papers = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for paper_id, similarity in sorted_papers:
            paper = self.collection.find_one({'_id': paper_id})
            if paper:
                paper['semantic_score'] = similarity
                results.append(paper)
        
        return results
    
    def keyword_search(self, query, top_k=10):
        """Traditional keyword search using MongoDB text search"""
        try:
            pipeline = [
                {
                    "$search": {
                        "index": "paper_search",
                        "compound": {
                            "should": [
                                {"text": {"query": query, "path": "title", "score": {"boost": {"value": 3}}}},
                                {"text": {"query": query, "path": "authors", "score": {"boost": {"value": 2}}}},
                                {"text": {"query": query, "path": "keywords"}}
                            ]
                        }
                    }
                },
                {"$limit": top_k},
                {"$project": {"title": 1, "authors": 1, "link": 1, "score": {"$meta": "searchScore"}}}
            ]
            results = list(self.collection.aggregate(pipeline))
            return results
        except:
            results = self.collection.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(top_k)
            return list(results)
    
    def hybrid_search(self, query, top_k=10, alpha=0.7):
        """Hybrid search combining semantic and keyword search"""
        semantic_results = self.semantic_search(query, top_k * 2)
        keyword_results = self.keyword_search(query, top_k * 2)

        combined_scores = {}
        
        for i, paper in enumerate(semantic_results):
            paper_id = paper['_id']
            semantic_score = paper.get('semantic_score', 0)
            combined_scores[paper_id] = alpha * semantic_score

        for i, paper in enumerate(keyword_results):
            paper_id = paper['_id']
            keyword_score = 1.0 / (i + 1) 
            if paper_id in combined_scores:
                combined_scores[paper_id] += (1 - alpha) * keyword_score
            else:
                combined_scores[paper_id] = (1 - alpha) * keyword_score
        
        sorted_paper_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        final_results = []
        for paper_id, score in sorted_paper_ids:
            paper = self.collection.find_one({'_id': paper_id})
            if paper:
                paper['hybrid_score'] = score
                final_results.append(paper)
        
        return final_results
    
    def search(self, query, method='hybrid', top_k=10):
        """Main search interface"""
        if method == 'semantic':
            return self.semantic_search(query, top_k)
        elif method == 'keyword':
            return self.keyword_search(query, top_k)
        else:
            return self.hybrid_search(query, top_k)

search_engine = HybridSearchEngine()

results = search_engine.search("neural networks transformers", method='hybrid', top_k=5)

for i, paper in enumerate(results, 1):
    print(f"{i}. {paper['title']}")
    print(f"   Authors: {', '.join(paper['authors'])}")
    print(f"   Score: {paper.get('hybrid_score', 'N/A')}")
    print(f"   Link: {paper['link']}")
    print()