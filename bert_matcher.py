"""
BERT Semantic Matching Module for Resume Screening

This module provides BERT-based semantic matching capabilities for the resume
screening application. It includes classes for:
- BERTEncoder: Loading and using BERT models for text encoding
- SimilarityCalculator: Computing cosine similarity between embeddings
- HybridScorer: Combining multiple similarity scores
- EmbeddingCache: Caching candidate embeddings for performance

Author: AI Resume Screening System
Date: 2026-02-26
"""

import os
import logging
import json
from typing import Union, List, Optional, Dict
from datetime import datetime
from dataclasses import dataclass
import hashlib

import numpy as np

# BERT dependencies will be imported with error handling
try:
    from sentence_transformers import SentenceTransformer
    import torch
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    SentenceTransformer = None
    torch = None

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class BERTConfig:
    """Configuration for BERT model"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 8
    max_seq_length: int = 512
    device: str = "auto"  # "auto", "cpu", "cuda"
    normalize_embeddings: bool = True


class BERTEncoder:
    """
    Handles BERT model loading and text encoding
    
    This class manages the BERT model lifecycle, including loading,
    device selection, and text encoding with proper error handling.
    Supports lazy loading to defer model initialization until first use.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 lazy_load: bool = False):
        """
        Initialize BERT encoder with specified model
        
        Args:
            model_name: HuggingFace model identifier
            lazy_load: If True, defer model loading until load_model() is explicitly called.
                       If False (default), caller is expected to call load_model() manually.
        """
        self.model_name = model_name
        self.model = None
        self.device = None
        self.lazy_load = lazy_load
        
    def load_model(self, force_cpu: bool = False) -> bool:
        """
        Load BERT model into memory with GPU fallback.
        
        If the model is already loaded, returns True immediately without reloading.
        
        Args:
            force_cpu: If True, force CPU device regardless of GPU availability.
                       Also honoured when the BERT_FORCE_CPU environment variable
                       is set to 'true'.
        
        Returns:
            True if model loaded successfully (or was already loaded), False otherwise.
        """
        if not BERT_AVAILABLE:
            logger.error("BERT dependencies not available. Install sentence-transformers and torch.")
            return False

        # Return early if model is already loaded
        if self.model is not None:
            logger.debug("BERT model already loaded, skipping reload.")
            return True
            
        try:
            start_time = datetime.now()

            # Determine device: force_cpu flag OR BERT_FORCE_CPU env var
            bert_force_cpu_env = os.environ.get("BERT_FORCE_CPU", "false").lower() == "true"
            if force_cpu or bert_force_cpu_env:
                self.device = "cpu"
                logger.info("Forcing CPU device for BERT model (force_cpu=%s, BERT_FORCE_CPU=%s)",
                            force_cpu, bert_force_cpu_env)
            else:
                self.device = self._detect_device()
                logger.info(f"Attempting to load model on device: {self.device}")
            
            # Try loading model on selected device
            try:
                self.model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    cache_folder=os.environ.get("TRANSFORMERS_CACHE", "./.cache")
                )

                # Set to eval mode to disable gradient tracking and save memory
                self.model.eval()

                load_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"BERT model loaded: {self.model_name} in {load_time:.2f}s on {self.device}")
                return True
                
            except RuntimeError as gpu_error:
                # GPU out of memory - fallback to CPU
                if "out of memory" in str(gpu_error).lower() or "cuda" in str(gpu_error).lower():
                    logger.warning(f"GPU error: {gpu_error}. Falling back to CPU...")
                    self.device = "cpu"
                    self.model = SentenceTransformer(
                        self.model_name,
                        device="cpu",
                        cache_folder=os.environ.get("TRANSFORMERS_CACHE", "./.cache")
                    )
                    # Set to eval mode after CPU fallback as well
                    self.model.eval()
                    load_time = (datetime.now() - start_time).total_seconds()
                    logger.info(f"BERT model loaded on CPU (fallback) in {load_time:.2f}s")
                    return True
                else:
                    raise
            
        except Exception as e:
            logger.error(f"Failed to load BERT model {self.model_name}: {e}", exc_info=True)
            return False
    
    def _detect_device(self) -> str:
        """
        Detect available device (GPU or CPU)
        
        Returns:
            Device string: "cuda" or "cpu"
        """
        if torch is None:
            return "cpu"
            
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def encode(self, 
               texts: Union[str, List[str]], 
               batch_size: int = 8,
               normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for text(s) with error handling
        
        Args:
            texts: Single text or list of texts
            batch_size: Number of texts to process at once
            normalize: Whether to normalize embeddings to unit length
            
        Returns:
            numpy array of shape (n_texts, embedding_dim) or (embedding_dim,)
            
        Raises:
            RuntimeError: If model not loaded
            ValueError: If encoding fails
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Handle single text
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        # Check for empty texts
        if not texts or all(not t.strip() for t in texts):
            logger.warning("Empty text(s) provided for encoding")
            # Return zero embeddings
            dim = self.get_embedding_dimension()
            if is_single:
                return np.zeros(dim)
            return np.zeros((len(texts), dim))
        
        try:
            # Log text truncation warnings
            for i, text in enumerate(texts):
                if len(text) > 512 * 4:  # Rough estimate for token count
                    logger.warning(f"Text {i} may exceed 512 tokens and will be truncated")
            
            # Encode with BERT
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            logger.debug(f"Encoded {len(texts)} text(s) successfully")
            
            # Return single embedding if input was single text
            if is_single:
                return embeddings[0]
            
            return embeddings
            
        except RuntimeError as gpu_error:
            # GPU out of memory during encoding
            if "out of memory" in str(gpu_error).lower():
                logger.error(f"GPU out of memory during encoding. Try reducing batch_size or using CPU.")
                raise ValueError(f"GPU memory error: {gpu_error}")
            raise
            
        except Exception as e:
            logger.error(f"Encoding failed: {e}", exc_info=True)
            raise ValueError(f"Encoding error: {e}")
    
    def get_embedding_dimension(self) -> int:
        """Return the dimensionality of embeddings"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model.get_sentence_embedding_dimension()


class SimilarityCalculator:
    """
    Computes similarity between embeddings
    
    Provides static methods for computing cosine similarity between
    embeddings, with support for both pairwise and batch operations.
    """
    
    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, 
                         embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score in [0, 1]
        """
        # Handle zero vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Ensure result is in [0, 1] (handle floating point errors)
        return float(np.clip(similarity, 0.0, 1.0))
    
    @staticmethod
    def batch_cosine_similarity(embeddings: np.ndarray,
                                query_embedding: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and multiple embeddings
        
        Args:
            embeddings: Matrix of shape (n_samples, embedding_dim)
            query_embedding: Vector of shape (embedding_dim,)
            
        Returns:
            Array of similarities of shape (n_samples,)
        """
        # Handle zero vectors
        if np.linalg.norm(query_embedding) == 0:
            return np.zeros(len(embeddings))
        
        # Vectorized cosine similarity computation
        # If embeddings are normalized, this is just dot product
        similarities = embeddings @ query_embedding
        
        # Ensure results are in [0, 1]
        return np.clip(similarities, 0.0, 1.0)


class HybridScorer:
    """
    Combines multiple similarity scores
    
    Supports three modes:
    - "bert": Use only BERT similarity
    - "tfidf": Use only TF-IDF similarity
    - "hybrid": Average BERT and TF-IDF similarities
    """
    
    def __init__(self, 
                 mode: str = "bert",
                 wt_bert: float = 0.85,
                 wt_tfidf: float = 0.0,
                 wt_exp: float = 0.15):
        """
        Initialize hybrid scorer
        
        Args:
            mode: "bert", "tfidf", or "hybrid"
            wt_bert: Weight for BERT similarity
            wt_tfidf: Weight for TF-IDF similarity
            wt_exp: Weight for experience score
        """
        self.mode = mode.lower()
        
        # Validate mode
        if self.mode not in ["bert", "tfidf", "hybrid"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'bert', 'tfidf', or 'hybrid'")
        
        # Store weights
        self.wt_bert = wt_bert
        self.wt_tfidf = wt_tfidf
        self.wt_exp = wt_exp
        
        # Validate and normalize weights
        self._validate_weights()
    
    def _validate_weights(self):
        """Validate that weights sum to approximately 1.0"""
        total = self.wt_bert + self.wt_tfidf + self.wt_exp
        
        # Check if weights sum to 1.0 (with tolerance)
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total:.4f}, normalizing to 1.0")
            # Normalize weights
            self.wt_bert /= total
            self.wt_tfidf /= total
            self.wt_exp /= total
    
    def compute_final_score(self,
                           bert_sim: Optional[float] = None,
                           tfidf_sim: Optional[float] = None,
                           exp_score: float = 0.0) -> float:
        """
        Compute weighted final match score
        
        Args:
            bert_sim: BERT similarity score [0, 1]
            tfidf_sim: TF-IDF similarity score [0, 1]
            exp_score: Experience score [0, 1]
            
        Returns:
            Final weighted score [0, 1]
        """
        if self.mode == "bert":
            if bert_sim is None:
                raise ValueError("BERT similarity required for 'bert' mode")
            text_sim = bert_sim
            
        elif self.mode == "tfidf":
            if tfidf_sim is None:
                raise ValueError("TF-IDF similarity required for 'tfidf' mode")
            text_sim = tfidf_sim
            
        elif self.mode == "hybrid":
            if bert_sim is None or tfidf_sim is None:
                raise ValueError("Both BERT and TF-IDF similarities required for 'hybrid' mode")
            # Average the two similarities
            text_sim = (bert_sim + tfidf_sim) / 2.0
        
        # Compute final weighted score
        final_score = (self.wt_bert + self.wt_tfidf) * text_sim + self.wt_exp * exp_score
        
        return float(np.clip(final_score, 0.0, 1.0))


class EmbeddingCache:
    """
    Manages cached embeddings for candidates with disk persistence

    Stores embeddings in memory with metadata for validation and
    provides methods for cache operations, batch pre-computation,
    and disk-based persistence for fast loading across restarts.
    """

    def __init__(self, cache_dir: str = ".bert_cache"):
        """
        Initialize cache with optional disk persistence

        Args:
            cache_dir: Directory to store cached embeddings on disk
        """
        self.cache: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict] = {}
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "embeddings.npz")
        self.metadata_file = os.path.join(cache_dir, "metadata.json")

        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            logger.info(f"Created cache directory: {cache_dir}")

    def get(self, candidate_id: str) -> Optional[np.ndarray]:
        """
        Retrieve cached embedding

        Args:
            candidate_id: Unique identifier for candidate

        Returns:
            Cached embedding or None if not found
        """
        return self.cache.get(candidate_id)

    def set(self, candidate_id: str, embedding: np.ndarray, text_hash: str):
        """
        Store embedding in cache

        Args:
            candidate_id: Unique identifier for candidate
            embedding: Embedding vector to cache
            text_hash: Hash of source text for validation
        """
        self.cache[candidate_id] = embedding
        self.metadata[candidate_id] = {
            "text_hash": text_hash,
            "timestamp": datetime.now().isoformat(),
            "shape": list(embedding.shape)
        }

    def invalidate(self, candidate_id: str):
        """
        Remove embedding from cache

        Args:
            candidate_id: Unique identifier for candidate
        """
        self.cache.pop(candidate_id, None)
        self.metadata.pop(candidate_id, None)

    def save_to_disk(self):
        """
        Save embeddings and metadata to disk for persistence

        Saves embeddings as compressed numpy arrays and metadata as JSON.
        This allows fast loading on subsequent startups.
        """
        try:
            logger.info(f"Saving {len(self.cache)} embeddings to disk...")
            start_time = datetime.now()

            # Save embeddings as compressed numpy arrays
            np.savez_compressed(self.cache_file, **self.cache)

            # Save metadata as JSON
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f)

            elapsed = (datetime.now() - start_time).total_seconds()
            file_size = os.path.getsize(self.cache_file) / (1024 * 1024)  # MB
            logger.info(f"Saved embeddings to disk in {elapsed:.2f}s ({file_size:.1f} MB)")

        except Exception as e:
            logger.error(f"Failed to save embeddings to disk: {e}", exc_info=True)

    def load_from_disk(self, cache_dir: Optional[str] = None) -> bool:
        """
        Load embeddings and metadata from disk.

        Args:
            cache_dir: Directory to load cache files from. If None, uses the
                       directory configured at construction time (self.cache_dir).

        Returns:
            True if successfully loaded, False if files are missing or on error.
        """
        # Resolve which directory (and therefore which files) to load from
        if cache_dir is not None:
            cache_file = os.path.join(cache_dir, "embeddings.npz")
            metadata_file = os.path.join(cache_dir, "metadata.json")
            logger.info(f"Loading embeddings from specified directory: {cache_dir}")
        else:
            cache_file = self.cache_file
            metadata_file = self.metadata_file

        try:
            # Handle missing cache files gracefully
            if not os.path.exists(cache_file):
                logger.info(f"No embedding cache file found at: {cache_file}")
                return False
            if not os.path.exists(metadata_file):
                logger.info(f"No metadata file found at: {metadata_file}")
                return False

            logger.info("Loading embeddings from disk...")
            start_time = datetime.now()

            # Load compressed numpy arrays
            data = np.load(cache_file)
            self.cache = {key: data[key] for key in data.files}

            # Load metadata for cache validation
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Loaded {len(self.cache)} embeddings from disk in {elapsed:.2f}s"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load embeddings from disk: {e}", exc_info=True)
            # Reset to empty state so the cache is not partially populated
            self.cache = {}
            self.metadata = {}
            return False

    def clear_disk_cache(self):
        """Remove cached files from disk"""
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
                logger.info("Removed embeddings cache file")
            if os.path.exists(self.metadata_file):
                os.remove(self.metadata_file)
                logger.info("Removed metadata cache file")
        except Exception as e:
            logger.error(f"Failed to clear disk cache: {e}", exc_info=True)

    def precompute_all(self, df, encoder: BERTEncoder, text_column: str = "_combined_text"):
        """
        Pre-compute embeddings for all candidates with error handling and disk persistence

        Args:
            df: DataFrame with candidate data
            encoder: BERTEncoder instance to use
            text_column: Column name containing text to encode
        """
        # Check if embeddings are already loaded
        if len(self.cache) > 0:
            logger.info(f"Embeddings already in cache ({len(self.cache)} candidates)")
            logger.info("Skipping pre-computation - using cached embeddings")
            return
        
        logger.info(f"Pre-computing embeddings for {len(df)} candidates...")
        start_time = datetime.now()

        # Extract texts and IDs
        texts = df[text_column].tolist()
        candidate_ids = df["Candidate_ID"].tolist()

        # Track failures
        failed_candidates = []
        successful_count = 0

        # Encode all texts in batches with error handling
        try:
            embeddings = encoder.encode(texts, batch_size=8, normalize=True)

            # Store in cache
            for cid, text, emb in zip(candidate_ids, texts, embeddings):
                try:
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    self.set(cid, emb, text_hash)
                    successful_count += 1
                except Exception as e:
                    logger.warning(f"Failed to cache embedding for candidate {cid}: {e}")
                    failed_candidates.append(cid)

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Pre-computed {successful_count}/{len(df)} embeddings in {elapsed:.2f}s")

            if failed_candidates:
                logger.warning(f"Failed to cache {len(failed_candidates)} candidates: {failed_candidates[:5]}...")

            # Save to disk for persistence
            self.save_to_disk()

        except Exception as e:
            logger.error(f"Pre-computation failed: {e}", exc_info=True)

            # Try individual encoding as fallback
            logger.info("Attempting individual encoding as fallback...")
            for cid, text in zip(candidate_ids, texts):
                try:
                    emb = encoder.encode(text, normalize=True)
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    self.set(cid, emb, text_hash)
                    successful_count += 1
                except Exception as ind_error:
                    logger.warning(f"Failed to encode candidate {cid}: {ind_error}")
                    failed_candidates.append(cid)

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Fallback encoding completed: {successful_count}/{len(df)} in {elapsed:.2f}s")

            if successful_count == 0:
                raise RuntimeError("All candidate encodings failed")

            # Save to disk even if some failed
            self.save_to_disk()

    def validate_cache(self, candidate_ids: List[str]) -> bool:
        """
        Validate that the cache contains embeddings for all provided candidate IDs.

        Compares the set of IDs currently in the cache against the provided list.
        Logs the number of missing candidates (in dataset but not cached) and
        extra candidates (cached but not in dataset), then returns whether the
        cache is complete.

        Args:
            candidate_ids: List of candidate IDs expected to be in the cache
                           (typically from the loaded dataset).

        Returns:
            True if every candidate_id in the list has a cached embedding,
            False if any are missing.
        """
        expected_ids = set(candidate_ids)
        cached_ids = set(self.cache.keys())

        missing = expected_ids - cached_ids   # in dataset but not cached
        extra = cached_ids - expected_ids     # cached but not in dataset

        if not missing and not extra:
            logger.info(
                "Cache validation passed: all %d candidates are cached.",
                len(expected_ids),
            )
            return True

        # At least one discrepancy — log details and return False if any missing
        if missing:
            logger.warning(
                "Cache validation failed: %d candidate(s) missing from cache "
                "(e.g. %s).",
                len(missing),
                list(missing)[:5],
            )
        if extra:
            logger.warning(
                "Cache validation: %d extra candidate(s) in cache not present "
                "in dataset (e.g. %s).",
                len(extra),
                list(extra)[:5],
            )

        # Validation passes only when there are no missing entries
        passed = len(missing) == 0
        if passed:
            logger.info(
                "Cache validation passed (no missing candidates; %d extra).",
                len(extra),
            )
        else:
            logger.warning(
                "Cache validation failed: missing=%d, extra=%d.",
                len(missing),
                len(extra),
            )
        return passed

    def get_all_embeddings(self, candidate_ids: List[str]) -> np.ndarray:
        """
        Get embeddings for multiple candidates as a matrix with error handling

        Args:
            candidate_ids: List of candidate IDs

        Returns:
            Matrix of shape (n_candidates, embedding_dim)

        Raises:
            ValueError: If no embeddings found
        """
        embeddings = []
        missing_ids = []

        for cid in candidate_ids:
            emb = self.get(cid)
            if emb is not None:
                embeddings.append(emb)
            else:
                logger.warning(f"No cached embedding for candidate {cid}")
                missing_ids.append(cid)

        if not embeddings:
            raise ValueError(f"No embeddings found in cache for {len(candidate_ids)} candidates")

        if missing_ids:
            logger.warning(f"Missing embeddings for {len(missing_ids)}/{len(candidate_ids)} candidates")

        return np.array(embeddings)



# Module-level variables for global access
_bert_encoder: Optional[BERTEncoder] = None
_embedding_cache: Optional[EmbeddingCache] = None


def get_bert_encoder() -> Optional[BERTEncoder]:
    """Get global BERT encoder instance"""
    return _bert_encoder


def get_embedding_cache() -> Optional[EmbeddingCache]:
    """Get global embedding cache instance"""
    return _embedding_cache


def initialize_bert_system(config: Optional[BERTConfig] = None) -> bool:
    """
    Initialize BERT system with model loading and cache setup
    
    Args:
        config: Optional BERTConfig for customization
        
    Returns:
        True if successful, False otherwise
    """
    global _bert_encoder, _embedding_cache
    
    if config is None:
        config = BERTConfig()
    
    logger.info(f"Initializing BERT system with model: {config.model_name}")
    logger.info(f"Configuration: batch_size={config.batch_size}, device={config.device}")
    
    try:
        # Create encoder
        _bert_encoder = BERTEncoder(model_name=config.model_name)
        
        # Load model
        logger.info("Loading BERT model...")
        if not _bert_encoder.load_model():
            logger.error("Failed to load BERT model")
            _bert_encoder = None
            return False
        
        # Create cache and try to load from disk
        logger.info("Creating embedding cache...")
        _embedding_cache = EmbeddingCache()
        
        # Try to load cached embeddings from disk
        if _embedding_cache.load_from_disk():
            logger.info("Successfully loaded embeddings from disk cache")
        else:
            logger.info("No disk cache found - embeddings will be computed on-demand or pre-computed")
        
        logger.info("BERT system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"BERT system initialization failed: {e}", exc_info=True)
        _bert_encoder = None
        _embedding_cache = None
        return False
