import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable, Dict, Any, Optional
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio
from datetime import datetime


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None):
        self.vectors = defaultdict(np.array)
        self.metadata = defaultdict(dict)  # NEW: Store metadata for each document
        self.embedding_model = embedding_model or EmbeddingModel()

    def insert(self, key: str, vector: np.array, metadata: Dict[str, Any] = None) -> None:
        """Insert a vector with optional metadata."""
        self.vectors[key] = vector
        if metadata:
            self.metadata[key] = metadata
        else:
            # Always store at least basic metadata
            self.metadata[key] = {
                "inserted_at": datetime.now().isoformat(),
                "text_length": len(key)
            }

    def _matches_filters(self, key: str, filters: Dict[str, Any]) -> bool:
        """Check if a document matches the given filters."""
        if not filters:
            return True
        
        document_metadata = self.metadata.get(key, {})
        
        for filter_key, filter_value in filters.items():
            if filter_key not in document_metadata:
                return False
            
            # Support different filter types
            if isinstance(filter_value, list):
                # Filter value is a list - check if document value is in the list
                if document_metadata[filter_key] not in filter_value:
                    return False
            elif isinstance(filter_value, dict):
                # Support range filters like {"chunk_index": {"min": 0, "max": 10}}
                doc_value = document_metadata[filter_key]
                if "min" in filter_value and doc_value < filter_value["min"]:
                    return False
                if "max" in filter_value and doc_value > filter_value["max"]:
                    return False
            else:
                # Exact match
                if document_metadata[filter_key] != filter_value:
                    return False
        
        return True

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = cosine_similarity,
        filters: Dict[str, Any] = None,
    ) -> List[Tuple[str, float]]:
        """Search vectors with optional metadata filtering."""
        # Filter eligible vectors first
        eligible_vectors = {
            key: vector for key, vector in self.vectors.items()
            if self._matches_filters(key, filters)
        }
        
        if not eligible_vectors:
            return []
        
        scores = [
            (key, distance_measure(query_vector, vector))
            for key, vector in eligible_vectors.items()
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_as_text: bool = False,
        filters: Dict[str, Any] = None,
        include_metadata: bool = False,
    ) -> List[Tuple[str, float]] | List[str] | List[Tuple[str, float, Dict]]:
        """Enhanced search with filtering and metadata options."""
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure, filters)
        
        if return_as_text:
            return [result[0] for result in results]
        elif include_metadata:
            return [(result[0], result[1], self.metadata.get(result[0], {})) for result in results]
        else:
            return results

    def retrieve_from_key(self, key: str) -> np.array:
        """Retrieve vector by key."""
        return self.vectors.get(key, None)
    
    def get_metadata(self, key: str) -> Dict[str, Any]:
        """Retrieve metadata for a specific document."""
        return self.metadata.get(key, {})
    
    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get all metadata."""
        return dict(self.metadata)
    
    def update_metadata(self, key: str, new_metadata: Dict[str, Any]) -> bool:
        """Update metadata for an existing document."""
        if key in self.vectors:
            self.metadata[key].update(new_metadata)
            return True
        return False

    async def abuild_from_list(
        self, 
        list_of_text: List[str], 
        metadata_list: List[Dict[str, Any]] = None
    ) -> "VectorDatabase":
        """Build vector database from text list with optional metadata."""
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        
        for i, (text, embedding) in enumerate(zip(list_of_text, embeddings)):
            # Use provided metadata or generate basic metadata
            metadata = None
            if metadata_list and i < len(metadata_list):
                metadata = metadata_list[i]
            else:
                # Generate basic metadata
                metadata = {
                    "chunk_index": i,
                    "text_length": len(text),
                    "word_count": len(text.split()),
                }
            
            self.insert(text, np.array(embedding), metadata)
        return self

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.vectors:
            return {"total_documents": 0}
        
        return {
            "total_documents": len(self.vectors),
            "total_metadata_entries": len(self.metadata),
            "metadata_keys": list(set().union(*(meta.keys() for meta in self.metadata.values()))),
            "average_text_length": np.mean([len(key) for key in self.vectors.keys()]),
        }


if __name__ == "__main__":
    # Test the enhanced functionality
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]
    
    # Create metadata for each text
    metadata_list = [
        {"topic": "food", "sentiment": "positive", "category": "vegetables"},
        {"topic": "food", "sentiment": "neutral", "category": "breakfast"},
        {"topic": "animals", "sentiment": "positive", "category": "pets"},
        {"topic": "animals", "sentiment": "positive", "category": "family"},
        {"topic": "animals", "sentiment": "positive", "category": "pets"},
    ]

    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text, metadata_list))
    
    # Test regular search
    print("=== Regular Search ===")
    results = vector_db.search_by_text("I think fruit is awesome!", k=2)
    print(f"Results: {results}")
    
    # Test filtered search
    print("\n=== Filtered Search (only animals) ===")
    filtered_results = vector_db.search_by_text(
        "cute pets", k=3, filters={"topic": "animals"}
    )
    print(f"Filtered results: {filtered_results}")
    
    # Test search with metadata
    print("\n=== Search with Metadata ===")
    results_with_metadata = vector_db.search_by_text(
        "food", k=2, include_metadata=True
    )
    for text, score, metadata in results_with_metadata:
        print(f"Text: {text[:50]}...")
        print(f"Score: {score:.3f}")
        print(f"Metadata: {metadata}")
        print("---")
    
    # Show stats
    print("\n=== Database Stats ===")
    print(vector_db.get_stats())
