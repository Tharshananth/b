"""Input validation utilities"""
from typing import Optional
from pydantic import BaseModel, Field, validator
import re

class ChatMessageValidator(BaseModel):
    """Validate chat message input"""
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    
    @validator('message')
    def validate_message(cls, v):
        # Remove excessive whitespace
        v = re.sub(r'\s+', ' ', v.strip())
        if not v:
            raise ValueError("Message cannot be empty")
        return v
    
    @validator('session_id')
    def validate_session_id(cls, v):
        if v and not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Invalid session ID format")
        return v

class FileUploadValidator(BaseModel):
    """Validate file upload"""
    filename: str
    size: int
    
    @validator('filename')
    def validate_filename(cls, v):
        # Check for path traversal
        if '..' in v or '/' in v or '\\' in v:
            raise ValueError("Invalid filename")
        
        # Check extension
        if not any(v.lower().endswith(ext) for ext in ['.pdf', '.docx', '.txt', '.md']):
            raise ValueError("Unsupported file type")
        
        return v
    
    @validator('size')
    def validate_size(cls, v):
        max_size = 10 * 1024 * 1024  # 10MB
        if v > max_size:
            raise ValueError(f"File too large (max 10MB)")
        return v
