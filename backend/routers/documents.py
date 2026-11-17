# routers/documents.py
"""Document management endpoints"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from typing import List
import logging
from pathlib import Path
from datetime import datetime
from pydantic import ValidationError
from sqlalchemy.orm import Session

from config import get_config
from utils.document_loader import DocumentLoader
from utils.validators import FileUploadValidator
from vector_db.retriever import get_retriever, reset_retriever
from database.connection import get_db
from database.models import AdminUser
from auth.dependencies import require_admin, client_ip, log_admin_action

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"],
    dependencies=[Depends(require_admin)],
)


@router.get("/")
async def list_documents() -> List[dict]:
    """
    List all indexed documents
    
    Returns:
        List of documents with metadata
    """
    try:
        retriever = get_retriever()
        count = retriever.get_vector_store().get_document_count()
        
        # Get documents from data directory
        config = get_config()
        data_dir = Path(config.documents.data_dir)
        
        documents = []
        if data_dir.exists():
            for file_path in data_dir.iterdir():
                if file_path.is_file() and file_path.suffix in config.documents.supported_formats:
                    documents.append({
                        "name": file_path.name,
                        "size": file_path.stat().st_size,
                        "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                        "type": file_path.suffix
                    })
        
        return documents
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list documents")


@router.post("/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    admin: AdminUser = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict:
    """
    Upload and index a new document
    
    Args:
        file: Document file to upload
        
    Returns:
        Upload status and document info
    """
    ip = client_ip(request)
    file_path = None
    try:
        config = get_config()
        
        # Validate file
        try:
            validator = FileUploadValidator(
                filename=file.filename,
                size=0  # Size validation will happen after reading
            )
        except ValidationError:
            log_admin_action(
                db,
                username=admin.username,
                action="upload",
                target=file.filename,
                succeeded=False,
                detail="Invalid filename or unsupported file type",
                ip=ip,
            )
            raise HTTPException(
                status_code=400,
                detail="Invalid filename or unsupported file type"
            )
        
        # Create upload directory
        upload_dir = Path(config.documents.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Validate file size
        file_size = file_path.stat().st_size
        if file_size > config.documents.max_file_size:
            file_path.unlink()
            detail_msg = f"File too large (max {config.documents.max_file_size // 1024 // 1024}MB)"
            log_admin_action(
                db,
                username=admin.username,
                action="upload",
                target=file.filename,
                succeeded=False,
                detail=detail_msg,
                ip=ip,
            )
            raise HTTPException(
                status_code=400,
                detail=detail_msg
            )
        
        # Load and index document
        docs = DocumentLoader.load_document(str(file_path))
        
        if not docs:
            file_path.unlink()
            log_admin_action(
                db,
                username=admin.username,
                action="upload",
                target=file.filename,
                succeeded=False,
                detail="Failed to load document",
                ip=ip,
            )
            raise HTTPException(status_code=400, detail="Failed to load document")
        
        # Add to vector store
        retriever = get_retriever()
        chunks = retriever.get_vector_store().add_documents(docs)
        
        logger.info(f"Uploaded and indexed {file.filename} ({chunks} chunks)")
        log_admin_action(
            db,
            username=admin.username,
            action="upload",
            target=file.filename,
            succeeded=True,
            detail=f"{chunks} chunks",
            ip=ip,
        )
        
        return {
            "success": True,
            "filename": file.filename,
            "size": file_size,
            "chunks": chunks,
            "message": "Document uploaded and indexed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_admin_action(
            db,
            username=admin.username,
            action="upload",
            target=getattr(file, "filename", None),
            succeeded=False,
            detail="Internal server error",
            ip=ip,
        )
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to upload document")


@router.delete("/{filename}")
def delete_document(
    filename: str,
    request: Request,
    admin: AdminUser = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict:
    """
    Delete a document
    
    Args:
        filename: Name of file to delete
        
    Returns:
        Deletion status
    """
    ip = client_ip(request)
    try:
        config = get_config()
        
        # Check in data directory
        data_path = Path(config.documents.data_dir) / filename
        upload_path = Path(config.documents.upload_dir) / filename
        
        deleted = False
        if data_path.exists():
            data_path.unlink()
            deleted = True
        
        if upload_path.exists():
            upload_path.unlink()
            deleted = True
        
        if not deleted:
            log_admin_action(
                db,
                username=admin.username,
                action="delete",
                target=filename,
                succeeded=False,
                detail="Document not found",
                ip=ip,
            )
            raise HTTPException(status_code=404, detail="Document not found")
        
        logger.info(f"Deleted document: {filename}")
        log_admin_action(
            db,
            username=admin.username,
            action="delete",
            target=filename,
            succeeded=True,
            detail="Document deleted",
            ip=ip,
        )
        
        return {
            "success": True,
            "message": f"Document {filename} deleted"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_admin_action(
            db,
            username=admin.username,
            action="delete",
            target=filename,
            succeeded=False,
            detail="Internal server error",
            ip=ip,
        )
        logger.error(f"Delete error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete document")


@router.post("/refresh")
def refresh_knowledge_base(
    request: Request,
    admin: AdminUser = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict:
    """
    Refresh the knowledge base by reindexing all documents
    
    Returns:
        Refresh status
    """
    ip = client_ip(request)
    try:
        config = get_config()
        
        # Clear existing vector store
        retriever = get_retriever()
        retriever.get_vector_store().delete_collection()

        # Drop the cached instance; it now points at a deleted collection
        reset_retriever()
        retriever = get_retriever()
        
        # Load all documents from data directory
        data_dir = config.documents.data_dir
        docs = DocumentLoader.load_directory(data_dir)
        
        if not docs:
            log_admin_action(
                db,
                username=admin.username,
                action="refresh",
                succeeded=True,
                detail="No documents found to index",
                ip=ip,
            )
            return {
                "success": True,
                "message": "No documents found to index",
                "documents": 0,
                "chunks": 0
            }
        
        # Add to vector store
        chunks = retriever.get_vector_store().add_documents(docs)
        
        logger.info(f"Refreshed knowledge base: {len(docs)} documents, {chunks} chunks")
        log_admin_action(
            db,
            username=admin.username,
            action="refresh",
            succeeded=True,
            detail=f"{len(docs)} documents, {chunks} chunks",
            ip=ip,
        )
        
        return {
            "success": True,
            "message": "Knowledge base refreshed successfully",
            "documents": len(docs),
            "chunks": chunks
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_admin_action(
            db,
            username=admin.username,
            action="refresh",
            succeeded=False,
            detail="Internal server error",
            ip=ip,
        )
        logger.error(f"Refresh error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to refresh knowledge base")
