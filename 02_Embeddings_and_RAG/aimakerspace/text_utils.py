import os
from typing import List, Dict, Any, Tuple
from datetime import datetime
import hashlib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.document_metadata = []  # NEW: Store metadata for each document
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt file."
            )

    def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            content = f.read()
            self.documents.append(content)
            
            # Generate metadata for this document
            file_stats = os.stat(self.path)
            metadata = {
                "source_file": os.path.basename(self.path),
                "source_path": self.path,
                "file_size_bytes": file_stats.st_size,
                "last_modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                "loaded_at": datetime.now().isoformat(),
                "document_index": len(self.documents) - 1,
                "text_length": len(content),
                "word_count": len(content.split()),
                "line_count": len(content.splitlines()),
                "doc_hash": hashlib.md5(content.encode()).hexdigest()[:8]  # Short hash for identification
            }
            self.document_metadata.append(metadata)

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding=self.encoding) as f:
                        content = f.read()
                        self.documents.append(content)
                        
                        # Generate metadata for this document
                        file_stats = os.stat(file_path)
                        metadata = {
                            "source_file": file,
                            "source_path": file_path,
                            "source_directory": root,
                            "file_size_bytes": file_stats.st_size,
                            "last_modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                            "loaded_at": datetime.now().isoformat(),
                            "document_index": len(self.documents) - 1,
                            "text_length": len(content),
                            "word_count": len(content.split()),
                            "line_count": len(content.splitlines()),
                            "doc_hash": hashlib.md5(content.encode()).hexdigest()[:8]
                        }
                        self.document_metadata.append(metadata)

    def load_documents(self):
        self.load()
        return self.documents
    
    def load_documents_with_metadata(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Load documents and return both documents and their metadata."""
        if not self.documents:  # Only load if not already loaded
            self.load()
        return self.documents, self.document_metadata


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        analyze_sentiment: bool = True,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.analyze_sentiment = analyze_sentiment
        
        # Initialize sentiment analyzer if enabled
        if self.analyze_sentiment:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks
    
    def split_with_metadata(
        self, 
        text: str, 
        source_metadata: Dict[str, Any] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Split text and generate metadata for each chunk."""
        chunks = []
        chunk_metadata = []
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i : i + self.chunk_size]
            chunks.append(chunk)
            
            # Generate metadata for this chunk
            metadata = {
                "chunk_index": len(chunks) - 1,
                "chunk_start_pos": i,
                "chunk_end_pos": min(i + self.chunk_size, len(text)),
                "chunk_size": len(chunk),
                "word_count": len(chunk.split()),
                "line_count": len(chunk.splitlines()),
                "overlap_size": self.chunk_overlap if i > 0 else 0,
                "is_first_chunk": i == 0,
                "is_last_chunk": i + self.chunk_size >= len(text),
                "chunk_hash": hashlib.md5(chunk.encode()).hexdigest()[:8],
                "created_at": datetime.now().isoformat(),
            }
            
            # Add sentiment analysis if enabled
            if self.analyze_sentiment:
                sentiment_scores = self.sentiment_analyzer.polarity_scores(chunk)
                metadata.update({
                    "sentiment_compound": sentiment_scores['compound'],  # Overall sentiment (-1 to 1)
                    "sentiment_positive": sentiment_scores['pos'],       # Positive score (0 to 1)
                    "sentiment_negative": sentiment_scores['neg'],       # Negative score (0 to 1)
                    "sentiment_neutral": sentiment_scores['neu'],        # Neutral score (0 to 1)
                    "sentiment_label": self._classify_sentiment(sentiment_scores['compound']),
                })
            
            # Add source metadata if provided
            if source_metadata:
                for key, value in source_metadata.items():
                    # Avoid redundant "source_source_" prefixes
                    if key.startswith('source_'):
                        # If it already starts with "source_", use it as-is
                        metadata[key] = value
                    else:
                        # Add "source_" prefix to other fields
                        metadata[f"source_{key}"] = value
            
            chunk_metadata.append(metadata)
        
        return chunks, chunk_metadata

    def split_texts_with_metadata(
        self, 
        texts: List[str], 
        source_metadata_list: List[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Split multiple texts and generate metadata for all chunks."""
        all_chunks = []
        all_metadata = []
        
        for doc_index, text in enumerate(texts):
            source_metadata = None
            if source_metadata_list and doc_index < len(source_metadata_list):
                source_metadata = source_metadata_list[doc_index]
                source_metadata["document_index"] = doc_index
            
            chunks, chunk_metadata = self.split_with_metadata(text, source_metadata)
            
            # Add global chunk index
            for i, metadata in enumerate(chunk_metadata):
                metadata["global_chunk_index"] = len(all_chunks) + i
                metadata["chunks_from_this_document"] = len(chunks)
                metadata["document_index"] = doc_index
            
            all_chunks.extend(chunks)
            all_metadata.extend(chunk_metadata)
        
        return all_chunks, all_metadata
    
    def _classify_sentiment(self, compound_score: float) -> str:
        """Classify sentiment based on compound score."""
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'


class MetadataEnhancedLoader:
    """A convenience class that combines loading and splitting with full metadata support."""
    
    def __init__(
        self, 
        path: str, 
        encoding: str = "utf-8",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        analyze_sentiment: bool = True
    ):
        self.loader = TextFileLoader(path, encoding)
        self.splitter = CharacterTextSplitter(chunk_size, chunk_overlap, analyze_sentiment)
        self._chunks = None
        self._chunk_metadata = None
        self._documents = None
        self._doc_metadata = None
    
    def load_and_split(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Load documents and split them into chunks with full metadata."""
        if self._chunks is None or self._chunk_metadata is None:
            self._documents, self._doc_metadata = self.loader.load_documents_with_metadata()
            self._chunks, self._chunk_metadata = self.splitter.split_texts_with_metadata(self._documents, self._doc_metadata)
        return self._chunks, self._chunk_metadata
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the loaded and processed documents."""
        # Ensure data is loaded
        if self._chunks is None:
            self.load_and_split()
        
        return {
            "total_documents": len(self._documents),
            "total_chunks": len(self._chunks),
            "total_characters": sum(len(doc) for doc in self._documents),
            "total_words": sum(len(doc.split()) for doc in self._documents),
            "average_chunk_size": sum(len(chunk) for chunk in self._chunks) / len(self._chunks) if self._chunks else 0,
            "source_files": [metadata["source_file"] for metadata in self._doc_metadata],
            "chunk_size_distribution": {
                "min": min(len(chunk) for chunk in self._chunks) if self._chunks else 0,
                "max": max(len(chunk) for chunk in self._chunks) if self._chunks else 0,
                "avg": sum(len(chunk) for chunk in self._chunks) / len(self._chunks) if self._chunks else 0
            }
        }


if __name__ == "__main__":
    # Test the enhanced functionality
    print("=== Testing Enhanced Loader ===")
    
    # Test with your PMarca data
    enhanced_loader = MetadataEnhancedLoader("data/PMarcaBlogs.txt")
    chunks, metadata = enhanced_loader.load_and_split()
    
    print(f"Loaded {len(chunks)} chunks")
    print("\n=== Summary ===")
    summary = enhanced_loader.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\n=== Sample Chunk Metadata ===")
    if metadata:
        sample_metadata = metadata[0]
        for key, value in sample_metadata.items():
            print(f"{key}: {value}")
    
    print("\n=== Sample Chunk Text ===")
    if chunks:
        print(f"First chunk: {chunks[0][:200]}...")
