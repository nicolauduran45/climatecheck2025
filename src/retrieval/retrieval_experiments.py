"""
ClimateCheck Retrieval Experiments
==================================

This module contains the implementation of various retrieval experiments:
- BM25 Retrieval
- Dense Retrieval (using various embedding models)
- Combined/Hybrid retrieval approaches

All experiments inherit from the ClimateCheckExperiment base class.
"""

import os
import json
import pickle
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from tqdm import tqdm

import sys
import os
# Add the parent directory to the path so we can import other modules in src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the base experiment class
from experiment_framework import ClimateCheckExperiment

logger = logging.getLogger(__name__)

class BM25RetrievalExperiment(ClimateCheckExperiment):
    """BM25 Retrieval experiment with optional transformer reranking."""
    
    def __init__(self, config_path: str):
        """Initialize BM25 experiment.
        
        Args:
            config_path: Path to experiment config
        """
        super().__init__(config_path)
        self.retriever_type = "bm25"
        
        # Set specific BM25 paths
        retriever_filename = f"bm25_{self.experiment_id}.pkl"
        self.retriever_path = self.index_dir / retriever_filename
        
        # Load NLTK if needed
        try:
            import nltk
            from nltk.tokenize import word_tokenize
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
        except ImportError:
            logger.error("NLTK is required for BM25 retrieval. Please install it.")
            raise
    
    def create_retriever(self):
        """Create and save a BM25 retriever."""
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.documents import Document
        from nltk.tokenize import word_tokenize
        
        logger.info("Creating BM25 retriever...")
        
        # Prepare content field (combine title and abstract)
        self.corpus_df['content'] = self.corpus_df['title_lowered'] + " " + self.corpus_df['abstract_lowered']
        
        # Create documents for the retriever
        documents = []
        for _, row in tqdm(self.corpus_df.iterrows(), total=len(self.corpus_df), desc="Preparing documents"):
            doc = Document(
                page_content=row['content'],
                metadata={"abstract_id": row['abstract_id']}
            )
            documents.append(doc)
        
        # Create BM25 retriever with word tokenization
        retriever = BM25Retriever.from_documents(
            documents,
            preprocess_func=word_tokenize
        )
        
        # Save the retriever
        logger.info(f"Saving BM25 retriever to {self.retriever_path}")
        self.retriever_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.retriever_path, 'wb') as f:
            pickle.dump(retriever, f)
        
        return retriever
    
    def load_retriever(self):
        """Load a BM25 retriever from disk."""
        if not self.retriever_path.exists():
            logger.info(f"No existing retriever found at {self.retriever_path}")
            return None
        
        logger.info(f"Loading BM25 retriever from {self.retriever_path}")
        with open(self.retriever_path, 'rb') as f:
            retriever = pickle.load(f)
        
        return retriever
    
    def load_reranker(self):
        """Load the transformer reranker if specified in config."""
        if 'reranker' not in self.config:
            return None, None
        
        reranker_config = self.config['reranker']
        model_name = reranker_config.get('model_name', 'intfloat/simlm-msmarco-reranker')
        
        logger.info(f"Loading reranker: {model_name}")
        
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Set device (use GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        return tokenizer, model
    
    def encode_query_doc_pair(self, query, passage, tokenizer, max_length=192):
        """Encode a query-document pair for reranking."""
        return tokenizer(
            query,
            text_pair=passage,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    
    def rerank_with_transformer(self, claim, candidates, tokenizer, model, device='cpu'):
        """Rerank candidates using a transformer model."""
        inputs = [self.encode_query_doc_pair(claim, doc.page_content, tokenizer) for doc in candidates]
        
        # Move tensors to device and batch them
        input_ids = torch.cat([inp["input_ids"] for inp in inputs]).to(device)
        attention_mask = torch.cat([inp["attention_mask"] for inp in inputs]).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            scores = outputs.logits.squeeze(-1)  # Shape: (batch_size,)
        
        # Attach scores to candidates
        for i, doc in enumerate(candidates):
            doc.metadata["rerank_score"] = scores[i].item()
        
        # Sort by reranker score descending
        return sorted(candidates, key=lambda x: x.metadata["rerank_score"], reverse=True)
    
    def retrieve(self, retriever, reranker_info=None, k_bm25=20, k_final=10):
        """Retrieve documents for all claims with optional reranking."""
        retriever.k = k_bm25
        results = {}
        tokenizer, model = reranker_info if reranker_info else (None, None)
        device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.is_available() else "cpu")
        
        for _, row in tqdm(self.claims_df.iterrows(), total=len(self.claims_df), desc="Retrieving documents"):
            claim_id = row['claim_id']
            claim_text = row['claim']
            
            # Get BM25 results
            bm25_docs = retriever.get_relevant_documents(claim_text)
            
            # Apply reranking if available
            if tokenizer and model:
                reranked_docs = self.rerank_with_transformer(claim_text, bm25_docs, tokenizer, model, device=device)
                docs_to_use = reranked_docs[:k_final]
                score_field = "rerank_score"
            else:
                docs_to_use = bm25_docs[:k_final]
                score_field = "score"
            
            # Format results
            results[claim_id] = [
                {
                    "abstract_id": doc.metadata["abstract_id"],
                    "score": doc.metadata.get(score_field, 0),
                    "rank": i + 1
                }
                for i, doc in enumerate(docs_to_use)
            ]
        
        return results
    
    def run(self):
        """Run the complete BM25 retrieval experiment."""
        # Load or create BM25 retriever
        retriever = self.load_retriever()
        if not retriever:
            retriever = self.create_retriever()
        
        # Load reranker if specified
        reranker_info = self.load_reranker() if 'reranker' in self.config else None
        
        # Get retrieval parameters
        retrieval_params = self.config.get('retriever', {}).get('params', {})
        k_bm25 = retrieval_params.get('k_bm25', 20)
        k_final = retrieval_params.get('k_final', 10)
        
        # Retrieve documents
        logger.info(f"Retrieving documents with BM25 (k_bm25={k_bm25}, k_final={k_final})")
        retrieval_results = self.retrieve(retriever, reranker_info, k_bm25, k_final)
        
        # Evaluate results
        logger.info("Generating gold data and evaluating results...")
        gold_data = self.generate_gold_data()
        metrics = self.evaluate(retrieval_results, gold_data)
        
        # Save results
        self.save_results(metrics, retrieval_results)
        
        return metrics, retrieval_results


class DenseRetrievalExperiment(ClimateCheckExperiment):
    """Dense retrieval experiment using embeddings and FAISS."""
    
    def __init__(self, config_path: str):
        """Initialize dense retrieval experiment.
        
        Args:
            config_path: Path to experiment config
        """
        super().__init__(config_path)
        self.retriever_type = "dense"
        
        # Get embedding model info
        self.model_name = self.config.get('retriever', {}).get('model_name', 'sentence-transformers/allenai-specter')
        self.granularity = self.config.get('retriever', {}).get('granularity', 'document')
        
        # Set specific paths for this experiment
        self.index_file = self.index_dir / f"{self.model_name.replace('/', '_')}_{self.granularity}.faiss"
        self.ids_file = self.index_dir / f"{self.model_name.replace('/', '_')}_{self.granularity}_ids.pkl"
        self.embeddings_file = self.index_dir / f"{self.model_name.replace('/', '_')}_{self.granularity}_embeddings.npy"
    
    def create_index(self):
        """Create and save a FAISS index with embeddings."""
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Creating dense index with model {self.model_name} at {self.granularity} level")
            
            # Load embedding model
            model = SentenceTransformer(self.model_name)
            
            # Prepare content based on granularity
            self.corpus_df['content'] = self.corpus_df['title_lowered'] + " " + self.corpus_df['abstract_lowered']
            texts = []
            ids = []
            
            if self.granularity == 'document':
                texts = self.corpus_df['content'].tolist()
                ids = self.corpus_df['abstract_id'].tolist()
                
            elif self.granularity == 'passage':
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
                
                for _, row in tqdm(self.corpus_df.iterrows(), total=len(self.corpus_df), desc="Creating passages"):
                    doc_id = row['abstract_id']
                    chunks = splitter.split_text(row['content'])
                    for i, chunk in enumerate(chunks):
                        texts.append(chunk)
                        ids.append(f"{doc_id}_p{i}")
                        
            elif self.granularity == 'sentence':
                import nltk
                from nltk.tokenize import sent_tokenize
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt')
                    
                for _, row in tqdm(self.corpus_df.iterrows(), total=len(self.corpus_df), desc="Creating sentences"):
                    doc_id = row['abstract_id']
                    sentences = sent_tokenize(row['content'])
                    for i, sentence in enumerate(sentences):
                        texts.append(sentence)
                        ids.append(f"{doc_id}_s{i}")
            
            # Generate embeddings
            logger.info(f"Encoding {len(texts)} texts...")
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
            
            # Create and save FAISS index
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)  # Inner product index (for normalized vectors = cosine)
            index.add(embeddings)
            
            # Create parent directories if they don't exist
            self.index_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save index, ids and embeddings
            logger.info(f"Saving index to {self.index_file}")
            faiss.write_index(index, str(self.index_file))
            
            with open(self.ids_file, "wb") as f:
                pickle.dump(ids, f)
                
            np.save(self.embeddings_file, embeddings)
            
            return index, ids, model
            
        except ImportError as e:
            logger.error(f"Required library not found: {e}")
            logger.error("Please install faiss-cpu or faiss-gpu and sentence-transformers")
            raise
    
    def load_index(self):
        """Load a FAISS index and IDs from disk."""
        try:
            import faiss
            
            if not self.index_file.exists() or not self.ids_file.exists():
                logger.info(f"No existing index found at {self.index_file}")
                return None, None, None
            
            logger.info(f"Loading FAISS index from {self.index_file}")
            index = faiss.read_index(str(self.index_file))
            
            with open(self.ids_file, "rb") as f:
                ids = pickle.load(f)
                
            # Load the embedding model
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.model_name)
            
            return index, ids, model
            
        except ImportError as e:
            logger.error(f"Required library not found: {e}")
            raise
    
    def retrieve(self, index, ids, model, k=10):
        """Retrieve documents using dense embeddings."""
        results = {}
        
        for _, row in tqdm(self.claims_df.iterrows(), total=len(self.claims_df), desc="Retrieving with embeddings"):
            claim_id = row['claim_id']
            claim_text = row['claim']
            
            # Generate query embedding
            query_embedding = model.encode([claim_text], convert_to_numpy=True, normalize_embeddings=True)
            
            # Search in the index
            scores, retrieved_idx = index.search(query_embedding, k)
            
            # Format results
            results[claim_id] = [
                {
                    "abstract_id": self._get_document_id(ids[i]),
                    "score": float(scores[0][j]),
                    "rank": j + 1
                }
                for j, i in enumerate(retrieved_idx[0])
            ]
            
        return results
    
    def _get_document_id(self, index_id):
        """Extract the document ID from an index ID that might include passage/sentence info."""
        # If using document granularity, return as is
        if self.granularity == 'document':
            return index_id
            
        # For passage/sentence, extract the base document ID
        return index_id.split('_')[0] if '_' in index_id else index_id
    
    def run(self):
        """Run the complete dense retrieval experiment."""
        # Load or create the FAISS index
        index, ids, model = self.load_index()
        if not (index and ids and model):
            index, ids, model = self.create_index()
        
        # Get retrieval parameters
        k = self.config.get('retriever', {}).get('params', {}).get('k', 10)
        
        # Perform retrieval
        logger.info(f"Retrieving documents with dense embeddings (k={k})")
        retrieval_results = self.retrieve(index, ids, model, k)
        
        # Evaluate results
        logger.info("Generating gold data and evaluating results...")
        gold_data = self.generate_gold_data()
        metrics = self.evaluate(retrieval_results, gold_data)
        
        # Save results
        self.save_results(metrics, retrieval_results)
        
        return metrics, retrieval_results