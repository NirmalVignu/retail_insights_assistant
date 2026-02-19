"""
Conversation Memory & Context Manager
Maintains conversation history with RAG and semantic search capabilities
"""

from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import logging
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages conversation history with vector embeddings for semantic search
    Supports RAG (Retrieval-Augmented Generation) pattern
    Includes query result caching for performance
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize conversation memory with sentence transformer
        
        Args:
            model_name: Sentence transformer model for embeddings
        """
        self.messages: List[Dict[str, Any]] = []
        self.embeddings: List[np.ndarray] = []
        self.index: Optional[faiss.IndexFlatL2] = None
        self.query_cache: Dict[str, Any] = {}  # Cache for query results
        
        # Load embedding model
        try:
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {model_name} (dim: {self.embedding_dim})")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
            self.embedding_dim = 384
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """
        Add message to conversation memory with embedding
        
        Args:
            role: "user" or "assistant"
            content: Message text
            metadata: Additional metadata (query type, results count, etc.)
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.messages.append(message)
        
        # Create embedding for semantic search
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode([content])[0].astype(np.float32)
                self.embeddings.append(embedding)
                
                # Initialize or update FAISS index
                self._update_index()
            except Exception as e:
                logger.warning(f"Failed to create embedding: {e}")
    
    def _update_index(self):
        """Update FAISS index with current embeddings"""
        if not self.embeddings:
            self.index = None
            return
        
        try:
            embeddings_array = np.array(self.embeddings).astype(np.float32)
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.index.add(embeddings_array)
        except Exception as e:
            logger.warning(f"Failed to update FAISS index: {e}")
    
    def retrieve_similar_messages(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve semantically similar messages from history
        
        Args:
            query: Query text for similarity search
            k: Number of messages to retrieve
            
        Returns:
            List of similar messages with relevance scores
        """
        if not self.index or not self.embedding_model:
            return []
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])[0].astype(np.float32)
            query_embedding = np.array([query_embedding])
            
            # Search similar messages
            distances, indices = self.index.search(query_embedding, min(k, len(self.messages)))
            
            similar_messages = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx >= 0 and idx < len(self.messages):
                    msg = self.messages[idx].copy()
                    msg["similarity_score"] = float(1 / (1 + distance))  # Convert distance to similarity
                    similar_messages.append(msg)
            
            return similar_messages
        except Exception as e:
            logger.warning(f"Similarity search failed: {e}")
            return []
    
    def get_context_window(self, window_size: int = 5) -> str:
        """
        Get recent conversation context for prompt
        
        Args:
            window_size: Number of recent messages to include
            
        Returns:
            Formatted conversation context
        """
        context_messages = self.messages[-window_size:]
        
        if not context_messages:
            return "No previous conversation history."
        
        context = "Recent conversation context:\n"
        for msg in context_messages:
            role = msg["role"].upper()
            content = msg["content"][:200]  # Truncate long messages
            context += f"{role}: {content}\n"
        
        return context
    
    def build_rag_context(self, query: str, k: int = 3) -> str:
        """
        Build RAG (Retrieval-Augmented Generation) context
        Combines recent history + semantically similar messages
        
        Args:
            query: Current query
            k: Number of similar messages to retrieve
            
        Returns:
            Combined context for LLM
        """
        # Get recent context
        recent_context = self.get_context_window(window_size=3)
        
        # Get similar messages
        similar = self.retrieve_similar_messages(query, k=k)
        
        rag_context = recent_context + "\n\nRelevant previous interactions:\n"
        
        for msg in similar:
            role = msg["role"].upper()
            content = msg["content"][:150]  # Truncate
            score = msg.get("similarity_score", 0)
            rag_context += f"[Relevance: {score:.2f}] {role}: {content}\n"
        
        return rag_context
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get conversation summary statistics
        
        Returns:
            Summary with message counts, topics, etc.
        """
        user_msgs = [m for m in self.messages if m["role"] == "user"]
        assistant_msgs = [m for m in self.messages if m["role"] == "assistant"]
        
        return {
            "total_messages": len(self.messages),
            "user_messages": len(user_msgs),
            "assistant_messages": len(assistant_msgs),
            "start_time": self.messages[0]["timestamp"] if self.messages else None,
            "last_message_time": self.messages[-1]["timestamp"] if self.messages else None,
            "average_user_msg_length": np.mean([len(m["content"]) for m in user_msgs]) if user_msgs else 0,
            "average_assistant_msg_length": np.mean([len(m["content"]) for m in assistant_msgs]) if assistant_msgs else 0,
        }
    
    def clear(self):
        """Clear all conversation history"""
        self.messages.clear()
        self.embeddings.clear()
        self.index = None
        logger.info("Conversation memory cleared")
    
    def export_history(self, filepath: str):
        """Export conversation history to JSON"""
        try:
            data = {
                "messages": self.messages,
                "summary": self.get_conversation_summary()
            }
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Conversation exported to {filepath}")
        except Exception as e:
            logger.error(f"Export failed: {e}")
    
    def __len__(self) -> int:
        """Return number of messages in memory"""
        return len(self.messages)
    
    def __repr__(self) -> str:
        return f"ConversationMemory(messages={len(self.messages)}, embeddings={len(self.embeddings)})"
    
    def get_query_cache_key(self, query: str) -> str:
        """Generate cache key from query"""
        return hashlib.md5(query.lower().encode()).hexdigest()
    
    def cache_query_result(self, query: str, result: Any) -> None:
        """Cache query result for repeated queries"""
        cache_key = self.get_query_cache_key(query)
        self.query_cache[cache_key] = {
            "timestamp": datetime.now().isoformat(),
            "result": result,
            "query": query[:100]  # Store query preview for debugging
        }
        logger.info(f"✅ CACHED result for query: {query[:50]}...")
        logger.info(f"   Cache key: {cache_key}")
        logger.info(f"   Cache now has {len(self.query_cache)} entries")
    
    def get_cached_query_result(self, query: str) -> Optional[Any]:
        """Get cached result if available (within 5 minutes)"""
        cache_key = self.get_query_cache_key(query)
        logger.info(f"🔍 Looking up cache key: {cache_key}")
        logger.info(f"   For query: {query[:50]}...")
        logger.info(f"   Available cache keys: {list(self.query_cache.keys())}")
        
        if cache_key in self.query_cache:
            cached = self.query_cache[cache_key]
            cache_time = datetime.fromisoformat(cached["timestamp"])
            age_minutes = (datetime.now() - cache_time).total_seconds() / 60
            
            if age_minutes < 5:  # Cache valid for 5 minutes
                logger.info(f"✅ CACHE HIT! Using cached result (age: {age_minutes:.1f}min)")
                return cached["result"]
            else:
                logger.info(f"❌ Cache expired (age: {age_minutes:.1f}min)")
        else:
            logger.info(f"❌ CACHE MISS - Key not found")
        return None
    
    def clear_cache(self) -> None:
        """Clear query cache"""
        self.query_cache.clear()
        logger.info("Query cache cleared")


if __name__ == "__main__":
    # Example usage
    memory = ConversationMemory()
    
    # Add messages
    memory.add_message("user", "What were the total sales in Q4?")
    memory.add_message("assistant", "Total sales in Q4 were $1.2M, up 15% from Q3.")
    memory.add_message("user", "How about Q3 sales?")
    memory.add_message("assistant", "Q3 sales were approximately $1.04M.")
    memory.add_message("user", "Which region performed best?")
    memory.add_message("assistant", "The North region had the highest sales at $450K in Q4.")
    
    # Retrieve similar messages
    similar = memory.retrieve_similar_messages("region performance", k=2)
    print(f"Similar messages to 'region performance':\n{similar}")
    
    # Get RAG context
    rag = memory.build_rag_context("Which regions grew the most?", k=2)
    print(f"\nRAG Context:\n{rag}")
    # Get summary
    print(f"\nConversation Summary:\n{memory.get_conversation_summary()}")
