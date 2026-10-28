"""Feedback API endpoints"""
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy.orm import Session
from datetime import datetime
import logging

from database import get_db, FeedbackInteraction

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/feedback", tags=["Feedback"])

class FeedbackSubmit(BaseModel):
    message_id: str
    feedback_type: str  # 'thumbs_up' or 'thumbs_down'
    feedback_comment: Optional[str] = None

class InteractionHistory(BaseModel):
    """Response model for interaction history"""
    id: str
    message_id: str
    timestamp: str
    question: str
    response: str
    provider_used: Optional[str]
    feedback_type: Optional[str]
    feedback_comment: Optional[str]

@router.post("/submit")
async def submit_feedback(
    feedback: FeedbackSubmit,
    db: Session = Depends(get_db)
):
    """
    Submit feedback for a message
    
    Args:
        feedback: Feedback data (message_id, type, optional comment)
        
    Returns:
        Success message
    """
    try:
        # Validate feedback type
        if feedback.feedback_type not in ['thumbs_up', 'thumbs_down']:
            raise HTTPException(
                status_code=400, 
                detail="feedback_type must be 'thumbs_up' or 'thumbs_down'"
            )
        
        # Find the interaction
        interaction = db.query(FeedbackInteraction).filter(
            FeedbackInteraction.message_id == feedback.message_id
        ).first()
        
        if not interaction:
            raise HTTPException(status_code=404, detail="Message not found")
        
        # Update feedback
        interaction.feedback_type = feedback.feedback_type
        interaction.feedback_comment = feedback.feedback_comment
        interaction.feedback_timestamp = datetime.utcnow()
        
        db.commit()
        
        logger.info(f"Feedback submitted for message {feedback.message_id}: {feedback.feedback_type}")
        
        return {
            "success": True,
            "message": "Feedback submitted successfully",
            "message_id": feedback.message_id,
            "feedback_type": feedback.feedback_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))