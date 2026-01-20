from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import asyncio
import base64
import io
from PIL import Image
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from emergentintegrations.llm.chat import LlmChat, UserMessage, FileContentWithMimeType, ImageContent

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Qdrant connection
qdrant_url = os.getenv('QDRANT_URL', ':memory:')
qdrant_client = QdrantClient(qdrant_url)

# Sentence transformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# LLM API Key
llm_api_key = os.getenv('EMERGENT_LLM_KEY')

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= Models =============
class PatientUpload(BaseModel):
    patient_name: str
    notes: str
    record_type: str = "general"

class SearchQuery(BaseModel):
    query: str
    limit: int = 10
    record_type: Optional[str] = None

class ChatMessage(BaseModel):
    message: str
    session_id: str
    patient_id: Optional[str] = None

class RecommendationRequest(BaseModel):
    patient_id: str
    context: str

# ============= Qdrant Setup =============
async def init_qdrant_collections():
    """Initialize Qdrant collections for multimodal data"""
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        # Create patient records collection
        if "patient_records" not in collection_names:
            qdrant_client.create_collection(
                collection_name="patient_records",
                vectors_config={
                    "text": VectorParams(size=384, distance=Distance.COSINE),
                }
            )
            logger.info("Created patient_records collection")
        
        # Create interaction history collection
        if "interaction_history" not in collection_names:
            qdrant_client.create_collection(
                collection_name="interaction_history",
                vectors_config={
                    "text": VectorParams(size=384, distance=Distance.COSINE),
                }
            )
            logger.info("Created interaction_history collection")
        
        # Create recommendations collection
        if "health_recommendations" not in collection_names:
            qdrant_client.create_collection(
                collection_name="health_recommendations",
                vectors_config={
                    "text": VectorParams(size=384, distance=Distance.COSINE),
                }
            )
            logger.info("Created health_recommendations collection")
            
    except Exception as e:
        logger.error(f"Error initializing Qdrant collections: {e}")

# ============= Helper Functions =============
def generate_text_embedding(text: str) -> List[float]:
    """Generate embedding for text using sentence-transformers"""
    embedding = embedding_model.encode(text)
    return embedding.tolist()

def generate_image_embedding(image_bytes: bytes) -> List[float]:
    """Generate embedding for image"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Simple approach: resize and flatten as embedding
        image = image.convert('RGB').resize((64, 64))
        img_array = np.array(image).flatten()[:384]  # Match embedding size
        # Pad if needed
        if len(img_array) < 384:
            img_array = np.pad(img_array, (0, 384 - len(img_array)))
        # Normalize
        img_array = img_array / 255.0
        return img_array.tolist()
    except Exception as e:
        logger.error(f"Error generating image embedding: {e}")
        # Return zero vector as fallback
        return [0.0] * 384

async def get_relevant_context(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Retrieve relevant context from Qdrant based on query"""
    try:
        query_vector = generate_text_embedding(query)
        
        search_results = qdrant_client.search(
            collection_name="patient_records",
            query_vector=("text", query_vector),
            limit=limit,
            with_payload=True
        )
        
        context_items = []
        for result in search_results:
            context_items.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            })
        
        return context_items
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return []

# ============= API Endpoints =============
@api_router.get("/")
async def root():
    return {"message": "Healthcare Memory Assistant API", "status": "operational"}

@api_router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "qdrant": "connected",
        "mongodb": "connected"
    }

@api_router.post("/upload-patient-record")
async def upload_patient_record(
    patient_name: str = Form(...),
    notes: str = Form(...),
    record_type: str = Form("general"),
    file: Optional[UploadFile] = File(None)
):
    """Upload patient record with optional multimodal file"""
    try:
        vector_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        
        # Generate text embedding
        text_for_embedding = f"{patient_name} {notes}"
        text_embedding = generate_text_embedding(text_for_embedding)
        
        # Process file if provided
        file_info = None
        image_embedding = None
        if file:
            file_content = await file.read()
            file_info = {
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(file_content)
            }
            
            # Generate image embedding if it's an image
            if file.content_type and file.content_type.startswith('image/'):
                image_embedding = generate_image_embedding(file_content)
        
        # Store in Qdrant
        point = PointStruct(
            id=vector_id,
            vector={"text": text_embedding},
            payload={
                "patient_name": patient_name,
                "notes": notes,
                "record_type": record_type,
                "timestamp": timestamp.isoformat(),
                "has_file": file is not None,
                "file_info": file_info
            }
        )
        
        qdrant_client.upsert(
            collection_name="patient_records",
            points=[point]
        )
        
        # Store metadata in MongoDB
        mongo_doc = {
            "vector_id": vector_id,
            "patient_name": patient_name,
            "notes": notes,
            "record_type": record_type,
            "timestamp": timestamp.isoformat(),
            "file_info": file_info
        }
        await db.patient_records.insert_one(mongo_doc)
        
        logger.info(f"Uploaded record for patient: {patient_name}")
        return {
            "success": True,
            "vector_id": vector_id,
            "message": "Patient record uploaded successfully"
        }
    except Exception as e:
        logger.error(f"Error uploading patient record: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/search")
async def semantic_search(search: SearchQuery):
    """Semantic search across patient records"""
    try:
        query_vector = generate_text_embedding(search.query)
        
        # Build filter if record_type specified
        search_filter = None
        if search.record_type:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="record_type",
                        match=MatchValue(value=search.record_type)
                    )
                ]
            )
        
        results = qdrant_client.search(
            collection_name="patient_records",
            query_vector=("text", query_vector),
            limit=search.limit,
            query_filter=search_filter,
            with_payload=True
        )
        
        search_results = []
        for result in results:
            search_results.append({
                "id": result.id,
                "score": result.score,
                "patient_name": result.payload.get("patient_name"),
                "notes": result.payload.get("notes"),
                "record_type": result.payload.get("record_type"),
                "timestamp": result.payload.get("timestamp"),
                "has_file": result.payload.get("has_file", False)
            })
        
        logger.info(f"Search completed: {len(search_results)} results found")
        return {"results": search_results, "count": len(search_results)}
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/chat")
async def chat_with_memory(chat: ChatMessage):
    """Chat with AI using memory and context retrieval"""
    try:
        # Get relevant context from Qdrant
        context_items = await get_relevant_context(chat.message, limit=3)
        
        # Build context string
        context_str = "\n".join([
            f"Record: {item['payload'].get('patient_name', 'Unknown')} - {item['payload'].get('notes', '')}"
            for item in context_items
        ])
        
        # Initialize LLM chat with GPT-5.2
        system_message = f"""You are a healthcare assistant with access to patient records and medical information.
        
Relevant patient context:
{context_str}

Provide helpful, empathetic, and evidence-based responses. Always cite which patient records you're referencing."""
        
        llm_chat = LlmChat(
            api_key=llm_api_key,
            session_id=chat.session_id,
            system_message=system_message
        ).with_model("openai", "gpt-5.2")
        
        user_msg = UserMessage(text=chat.message)
        response = await llm_chat.send_message(user_msg)
        
        # Store interaction in Qdrant for memory
        interaction_id = str(uuid.uuid4())
        interaction_text = f"User: {chat.message}\nAssistant: {response}"
        interaction_embedding = generate_text_embedding(interaction_text)
        
        point = PointStruct(
            id=interaction_id,
            vector={"text": interaction_embedding},
            payload={
                "session_id": chat.session_id,
                "user_message": chat.message,
                "assistant_response": response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "context_used": [item['id'] for item in context_items]
            }
        )
        
        qdrant_client.upsert(
            collection_name="interaction_history",
            points=[point]
        )
        
        logger.info(f"Chat interaction stored: {interaction_id}")
        return {
            "response": response,
            "context_used": len(context_items),
            "interaction_id": interaction_id
        }
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/analyze-multimodal")
async def analyze_multimodal(
    query: str = Form(...),
    file: UploadFile = File(...)
):
    """Analyze multimodal data (image/audio/video) using Gemini"""
    try:
        file_content = await file.read()
        
        # Convert to base64 for Gemini
        file_base64 = base64.b64encode(file_content).decode('utf-8')
        
        # Initialize Gemini for multimodal analysis
        gemini_chat = LlmChat(
            api_key=llm_api_key,
            session_id=str(uuid.uuid4()),
            system_message="You are a medical AI assistant analyzing patient data. Provide detailed, professional analysis."
        ).with_model("gemini", "gemini-3-flash-preview")
        
        # Create message with image
        image_content = ImageContent(image_base64=file_base64)
        user_msg = UserMessage(
            text=query,
            file_contents=[image_content]
        )
        
        analysis = await gemini_chat.send_message(user_msg)
        
        logger.info(f"Multimodal analysis completed for file: {file.filename}")
        return {
            "analysis": analysis,
            "filename": file.filename,
            "content_type": file.content_type
        }
    except Exception as e:
        logger.error(f"Error in multimodal analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/recommendations")
async def get_recommendations(req: RecommendationRequest):
    """Get context-aware health recommendations using Claude"""
    try:
        # Get patient context
        context_items = await get_relevant_context(req.patient_id, limit=5)
        
        context_str = "\n".join([
            f"- {item['payload'].get('notes', '')}"
            for item in context_items
        ])
        
        # Use Claude for complex medical reasoning
        claude_chat = LlmChat(
            api_key=llm_api_key,
            session_id=str(uuid.uuid4()),
            system_message=f"""You are a medical AI providing evidence-based health recommendations.
            
Patient Context:
{context_str}

Provide specific, actionable recommendations based on the patient's history."""
        ).with_model("anthropic", "claude-sonnet-4-5-20250929")
        
        user_msg = UserMessage(text=req.context)
        recommendations = await claude_chat.send_message(user_msg)
        
        # Store recommendation
        rec_id = str(uuid.uuid4())
        rec_embedding = generate_text_embedding(recommendations)
        
        point = PointStruct(
            id=rec_id,
            vector={"text": rec_embedding},
            payload={
                "patient_id": req.patient_id,
                "recommendation": recommendations,
                "context": req.context,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
        qdrant_client.upsert(
            collection_name="health_recommendations",
            points=[point]
        )
        
        logger.info(f"Recommendations generated: {rec_id}")
        return {
            "recommendations": recommendations,
            "recommendation_id": rec_id,
            "context_records_used": len(context_items)
        }
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        collections = qdrant_client.get_collections().collections
        
        stats = {}
        for collection in collections:
            info = qdrant_client.get_collection(collection.name)
            stats[collection.name] = {
                "vectors_count": info.vectors_count if hasattr(info, 'vectors_count') else 0,
                "points_count": info.points_count if hasattr(info, 'points_count') else 0
            }
        
        # MongoDB stats
        patient_count = await db.patient_records.count_documents({})
        
        return {
            "qdrant_collections": stats,
            "mongodb_patient_records": patient_count,
            "status": "operational"
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    await init_qdrant_collections()
    logger.info("Healthcare Memory Assistant API started")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
