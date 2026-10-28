"""Document loading utilities"""
from pathlib import Path
from typing import List, Optional
import logging
from langchain_core.documents import Document  # ← FIXED
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from config import get_config

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Loads documents from various file formats"""
    
    LOADERS = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.txt': TextLoader,
        '.md': UnstructuredMarkdownLoader
    }
    
    @classmethod
    def load_document(cls, file_path: str) -> Optional[List[Document]]:
        """
        Load a single document
        
        Args:
            file_path: Path to document file
            
        Returns:
            List of Document objects or None if loading fails
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            # Get loader class for file extension
            loader_class = cls.LOADERS.get(path.suffix.lower())
            
            if loader_class is None:
                logger.error(f"Unsupported file type: {path.suffix}")
                return None
            
            # Load document
            loader = loader_class(str(path))
            docs = loader.load()
            
            # Add source metadata
            for doc in docs:
                doc.metadata['source'] = path.name
                doc.metadata['file_path'] = str(path)
            
            logger.info(f"Loaded {len(docs)} pages from {path.name}")
            return docs
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return None
    
    @classmethod
    def load_directory(cls, directory: str) -> List[Document]:
        """
        Load all documents from a directory
        
        Args:
            directory: Path to directory
            
        Returns:
            List of all loaded documents
        """
        all_docs = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.warning(f"Directory not found: {directory}")
            return all_docs
        
        # Get supported file extensions
        config = get_config()
        supported_formats = config.documents.supported_formats
        
        # Load all supported files
        for ext in supported_formats:
            for file_path in dir_path.glob(f"*{ext}"):
                docs = cls.load_document(str(file_path))
                if docs:
                    all_docs.extend(docs)
        
        logger.info(f"Loaded {len(all_docs)} documents from {directory}")
        return all_docs
    
    @classmethod
    def validate_file(cls, file_path: str, max_size: Optional[int] = None) -> tuple[bool, str]:
        """
        Validate a file before loading
        
        Args:
            file_path: Path to file
            max_size: Maximum file size in bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                return False, "File does not exist"
            
            # Check file extension
            config = get_config()
            if path.suffix.lower() not in config.documents.supported_formats:
                return False, f"Unsupported file type: {path.suffix}"
            
            # Check file size
            max_size = max_size or config.documents.max_file_size
            if path.stat().st_size > max_size:
                return False, f"File too large (max {max_size // 1024 // 1024}MB)"
            
            return True, "Valid"
            
        except Exception as e:
            return False, str(e)