import os
import sqlite3
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from app.core.config import settings

class RAGService:
    """Service for Retrieval-Augmented Generation using ChromaDB and Sentence Transformers."""
    
    def __init__(self):
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.db_path = "rag_documents.sqlite"
        self._initialize_components()
        self._init_db()
    
    def _initialize_components(self):
        """Initialize ChromaDB and embedding model."""
        try:
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(settings.HF_EMBEDDING_MODEL)
            
            # Initialize ChromaDB
            if not os.path.exists(settings.CHROMA_PERSIST_DIR):
                os.makedirs(settings.CHROMA_PERSIST_DIR)
            
            self.chroma_client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            
        except Exception as e:
            print(f"Error initializing RAG components: {e}")
            raise
    
    def _init_db(self):
        """Initialize SQLite database for document metadata."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT,
                    content_type TEXT,
                    uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    chunk_count INTEGER,
                    metadata TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT,
                    chunk_index INTEGER,
                    content TEXT,
                    embedding_stored BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            """)
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database initialization error: {e}")
        finally:
            if conn:
                conn.close()
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks."""
        chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
        overlap = overlap or settings.RAG_CHUNK_OVERLAP
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_end = max(end - 100, start + chunk_size // 2)
                for i in range(end, search_end, -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def add_document(self, document_id: str, filename: str, content: str, 
                    content_type: str = "text/plain", metadata: Dict = None) -> bool:
        """Add a document to the RAG system."""
        try:
            # Chunk the document
            chunks = self.chunk_text(content)
            
            # Store document metadata
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO documents (id, filename, content_type, chunk_count, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (document_id, filename, content_type, len(chunks), json.dumps(metadata or {})))
            
            # Store chunks and embeddings
            chunk_ids = []
            chunk_contents = []
            chunk_metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                chunk_ids.append(chunk_id)
                chunk_contents.append(chunk)
                chunk_metadatas.append({
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": i,
                    "content_type": content_type,
                    **(metadata or {})
                })
                
                # Store chunk in SQLite
                cursor.execute("""
                    INSERT OR REPLACE INTO document_chunks (id, document_id, chunk_index, content)
                    VALUES (?, ?, ?, ?)
                """, (chunk_id, document_id, i, chunk))
            
            conn.commit()
            conn.close()
            
            # Generate embeddings and store in ChromaDB
            embeddings = self.embedding_model.encode(chunk_contents).tolist()
            
            # Check if documents already exist in collection
            existing_ids = set()
            try:
                existing = self.collection.get(ids=chunk_ids)
                existing_ids = set(existing['ids'])
            except:
                pass
            
            # Only add new chunks
            new_chunk_ids = [cid for cid in chunk_ids if cid not in existing_ids]
            if new_chunk_ids:
                new_indices = [chunk_ids.index(cid) for cid in new_chunk_ids]
                new_embeddings = [embeddings[i] for i in new_indices]
                new_contents = [chunk_contents[i] for i in new_indices]
                new_metadatas = [chunk_metadatas[i] for i in new_indices]
                
                self.collection.add(
                    ids=new_chunk_ids,
                    embeddings=new_embeddings,
                    documents=new_contents,
                    metadatas=new_metadatas
                )
            
            return True
            
        except Exception as e:
            print(f"Error adding document to RAG: {e}")
            return False
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve the most relevant chunks for a query."""
        top_k = top_k or settings.RAG_TOP_K
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            relevant_chunks = []
            for i in range(len(results['ids'][0])):
                relevant_chunks.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i]  # Convert distance to similarity
                })
            
            return relevant_chunks
            
        except Exception as e:
            print(f"Error retrieving relevant chunks: {e}")
            return []
    
    def get_context_for_query(self, query: str, max_context_length: int = 2000) -> str:
        """Get concatenated context from relevant chunks for a query."""
        relevant_chunks = self.retrieve_relevant_chunks(query)
        
        context_parts = []
        current_length = 0
        
        for chunk in relevant_chunks:
            content = chunk['content']
            if current_length + len(content) <= max_context_length:
                context_parts.append(f"[From {chunk['metadata'].get('filename', 'unknown')}]: {content}")
                current_length += len(content)
            else:
                # Add partial content if there's space
                remaining_space = max_context_length - current_length
                if remaining_space > 100:  # Only add if meaningful space remains
                    partial_content = content[:remaining_space - 10] + "..."
                    context_parts.append(f"[From {chunk['metadata'].get('filename', 'unknown')}]: {partial_content}")
                break
        
        return "\n\n".join(context_parts)
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks from the RAG system."""
        try:
            # Get chunk IDs to delete from ChromaDB
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM document_chunks WHERE document_id = ?", (document_id,))
            chunk_ids = [row[0] for row in cursor.fetchall()]
            
            # Delete from ChromaDB
            if chunk_ids:
                self.collection.delete(ids=chunk_ids)
            
            # Delete from SQLite
            cursor.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
            cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"Error deleting document from RAG: {e}")
            return False
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the RAG system."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, filename, content_type, uploaded_at, chunk_count, metadata
                FROM documents ORDER BY uploaded_at DESC
            """)
            
            documents = []
            for row in cursor.fetchall():
                documents.append({
                    'id': row[0],
                    'filename': row[1],
                    'content_type': row[2],
                    'uploaded_at': row[3],
                    'chunk_count': row[4],
                    'metadata': json.loads(row[5]) if row[5] else {}
                })
            
            conn.close()
            return documents
            
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents in the RAG system (alias for list_documents)."""
        return self.list_documents()

    def get_documents_by_date_range(self, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Get documents within a specific date range for temporal analysis."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM documents"
            params = []
            
            if start_date or end_date:
                query += " WHERE"
                conditions = []
                
                if start_date:
                    conditions.append(" uploaded_at >= ?")
                    params.append(start_date)
                
                if end_date:
                    conditions.append(" uploaded_at <= ?")
                    params.append(end_date)
                
                query += " AND".join(conditions)
            
            query += " ORDER BY uploaded_at DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            documents = []
            for row in rows:
                doc = {
                    "id": row[0],
                    "filename": row[1],
                    "content_type": row[2],
                    "uploaded_at": row[3],
                    "chunk_count": row[4],
                    "metadata": json.loads(row[5]) if row[5] else {}
                }
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Error getting documents by date range: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def get_documents_by_month(self, month: int, year: int) -> List[Dict]:
        """Get all documents for a specific month and year."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create date range for the month
            start_date = f"{year}-{month:02d}-01"
            if month == 12:
                end_date = f"{year + 1}-01-01"
            else:
                end_date = f"{year}-{month + 1:02d}-01"
            
            cursor.execute("""
                SELECT * FROM documents 
                WHERE uploaded_at >= ? AND uploaded_at < ?
                ORDER BY uploaded_at DESC
            """, (start_date, end_date))
            
            rows = cursor.fetchall()
            
            documents = []
            for row in rows:
                doc = {
                    "id": row[0],
                    "filename": row[1],
                    "content_type": row[2],
                    "uploaded_at": row[3],
                    "chunk_count": row[4],
                    "metadata": json.loads(row[5]) if row[5] else {}
                }
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Error getting documents by month: {e}")
            return []
        finally:
            if conn:
                conn.close()

# Global RAG service instance
rag_service = RAGService()
