"""
TTV Service Event Emission System
Handles job lifecycle notifications and integration with BHIV event system
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import asyncio
import uuid

import httpx
from redis import Redis

from .config import settings


logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Event types for TTV service"""
    JOB_CREATED = "ttv.job.created"
    JOB_STARTED = "ttv.job.started"
    JOB_PROGRESS = "ttv.job.progress"
    JOB_COMPLETED = "ttv.job.completed"
    JOB_FAILED = "ttv.job.failed"
    JOB_CANCELLED = "ttv.job.cancelled"
    SYSTEM_HEALTH = "ttv.system.health"
    STORAGE_ACTION = "ttv.storage.action"
    SECURITY_ALERT = "ttv.security.alert"


class Event:
    """Event data structure"""
    
    def __init__(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        user_id: Optional[str] = None,
        job_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        self.id = str(uuid.uuid4())
        self.event_type = event_type
        self.data = data
        self.user_id = user_id
        self.job_id = job_id
        self.timestamp = timestamp or datetime.utcnow()
        self.service = "ttv"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "data": self.data,
            "user_id": self.user_id,
            "job_id": self.job_id,
            "timestamp": self.timestamp.isoformat(),
            "service": self.service
        }
    
    def to_json(self) -> str:
        """Convert event to JSON string"""
        return json.dumps(self.to_dict())


class EventHandler:
    """Base class for event handlers"""
    
    async def handle(self, event: Event) -> bool:
        """Handle an event. Return True if handled successfully."""
        raise NotImplementedError


class RedisEventHandler(EventHandler):
    """Redis-based event handler for pub/sub"""
    
    def __init__(self):
        self.redis_client = Redis.from_url(settings.redis_url)
        self.channel_prefix = "ttv_events"
    
    async def handle(self, event: Event) -> bool:
        """Publish event to Redis"""
        try:
            channel = f"{self.channel_prefix}:{event.event_type}"
            
            # Publish to specific event type channel
            self.redis_client.publish(channel, event.to_json())
            
            # Also publish to general TTV events channel
            self.redis_client.publish(f"{self.channel_prefix}:all", event.to_json())
            
            # Store event for potential replay
            key = f"ttv_event_log:{event.timestamp.strftime('%Y-%m-%d')}"
            self.redis_client.lpush(key, event.to_json())
            self.redis_client.expire(key, 86400 * 7)  # Keep for 7 days
            
            logger.debug(f"Published event {event.id} to Redis")
            return True
            
        except Exception as e:
            logger.error(f"Error publishing event to Redis: {str(e)}")
            return False


class WebhookEventHandler(EventHandler):
    """Webhook-based event handler for BHIV backend integration"""
    
    def __init__(self):
        self.webhook_url = f"{settings.bhiv_backend_url}/webhooks/ttv"
        self.webhook_secret = settings.bhiv_webhook_secret
        self.client = httpx.AsyncClient(timeout=10.0)
    
    async def handle(self, event: Event) -> bool:
        """Send event to BHIV backend via webhook"""
        try:
            if not settings.bhiv_backend_url:
                return False
            
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "TTV-Service/1.0"
            }
            
            # Add webhook signature if secret is configured
            if self.webhook_secret:
                import hmac
                import hashlib
                
                payload = event.to_json()
                signature = hmac.new(
                    self.webhook_secret.encode(),
                    payload.encode(),
                    hashlib.sha256
                ).hexdigest()
                headers["X-TTV-Signature"] = f"sha256={signature}"
            
            response = await self.client.post(
                self.webhook_url,
                json=event.to_dict(),
                headers=headers
            )
            
            if response.status_code == 200:
                logger.debug(f"Sent event {event.id} to BHIV webhook")
                return True
            else:
                logger.warning(f"Webhook returned status {response.status_code} for event {event.id}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending event to webhook: {str(e)}")
            return False


class DatabaseEventHandler(EventHandler):
    """Database event handler for event persistence"""
    
    def __init__(self):
        from sqlmodel import SQLModel, Field, Session, create_engine
        from .job_manager import TTVJob
        
        self.db_engine = create_engine(settings.database_config['url'])
        
        # Event storage model
        class TTVEvent(SQLModel, table=True):
            __tablename__ = "ttv_events"
            
            id: Optional[str] = Field(default=None, primary_key=True)
            event_type: str = Field(index=True)
            user_id: Optional[str] = Field(default=None, index=True)
            job_id: Optional[str] = Field(default=None, index=True)
            data: str = Field()  # JSON string
            timestamp: datetime = Field(index=True)
            service: str = Field(default="ttv", index=True)
        
        self.TTVEvent = TTVEvent
        
        # Create tables
        SQLModel.metadata.create_all(self.db_engine)
    
    async def handle(self, event: Event) -> bool:
        """Store event in database"""
        try:
            from sqlmodel import Session
            
            with Session(self.db_engine) as session:
                db_event = self.TTVEvent(
                    id=event.id,
                    event_type=event.event_type,
                    user_id=event.user_id,
                    job_id=event.job_id,
                    data=json.dumps(event.data),
                    timestamp=event.timestamp,
                    service=event.service
                )
                session.add(db_event)
                session.commit()
                
                logger.debug(f"Stored event {event.id} in database")
                return True
                
        except Exception as e:
            logger.error(f"Error storing event in database: {str(e)}")
            return False


class EventEmitter:
    """Main event emission system"""
    
    def __init__(self):
        self.handlers: List[EventHandler] = []
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup event handlers based on configuration"""
        # Always use Redis for internal pub/sub
        self.handlers.append(RedisEventHandler())
        
        # Add webhook handler for BHIV integration
        if settings.bhiv_backend_url:
            self.handlers.append(WebhookEventHandler())
        
        # Add database handler for event persistence
        if settings.database_url:
            self.handlers.append(DatabaseEventHandler())
        
        logger.info(f"Initialized event emitter with {len(self.handlers)} handlers")
    
    async def emit(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        user_id: Optional[str] = None,
        job_id: Optional[str] = None
    ):
        """Emit an event to all configured handlers"""
        event = Event(
            event_type=event_type,
            data=data,
            user_id=user_id,
            job_id=job_id
        )
        
        # Send to all handlers concurrently
        tasks = [handler.handle(event) for handler in self.handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log results
        success_count = sum(1 for result in results if result is True)
        logger.debug(f"Emitted event {event.id}: {success_count}/{len(self.handlers)} handlers succeeded")
        
        # Log any failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Handler {i} failed for event {event.id}: {str(result)}")
    
    async def emit_job_created(self, job_id: str, user_id: str, request_data: Dict[str, Any]):
        """Emit job created event"""
        await self.emit(
            EventType.JOB_CREATED,
            {
                "job_id": job_id,
                "request_data": request_data,
                "created_at": datetime.utcnow().isoformat()
            },
            user_id=user_id,
            job_id=job_id
        )
    
    async def emit_job_started(self, job_id: str, user_id: str):
        """Emit job started event"""
        await self.emit(
            EventType.JOB_STARTED,
            {
                "job_id": job_id,
                "started_at": datetime.utcnow().isoformat()
            },
            user_id=user_id,
            job_id=job_id
        )
    
    async def emit_job_progress(
        self, 
        job_id: str, 
        user_id: str, 
        progress: int, 
        message: str = "",
        current_step: str = ""
    ):
        """Emit job progress event"""
        await self.emit(
            EventType.JOB_PROGRESS,
            {
                "job_id": job_id,
                "progress": progress,
                "message": message,
                "current_step": current_step,
                "timestamp": datetime.utcnow().isoformat()
            },
            user_id=user_id,
            job_id=job_id
        )
    
    async def emit_job_completed(
        self, 
        job_id: str, 
        user_id: str, 
        result: Dict[str, Any]
    ):
        """Emit job completed event"""
        await self.emit(
            EventType.JOB_COMPLETED,
            {
                "job_id": job_id,
                "result": result,
                "completed_at": datetime.utcnow().isoformat()
            },
            user_id=user_id,
            job_id=job_id
        )
    
    async def emit_job_failed(
        self, 
        job_id: str, 
        user_id: str, 
        error: str,
        error_details: Dict[str, Any] = None
    ):
        """Emit job failed event"""
        await self.emit(
            EventType.JOB_FAILED,
            {
                "job_id": job_id,
                "error": error,
                "error_details": error_details or {},
                "failed_at": datetime.utcnow().isoformat()
            },
            user_id=user_id,
            job_id=job_id
        )
    
    async def emit_job_cancelled(self, job_id: str, user_id: str):
        """Emit job cancelled event"""
        await self.emit(
            EventType.JOB_CANCELLED,
            {
                "job_id": job_id,
                "cancelled_at": datetime.utcnow().isoformat()
            },
            user_id=user_id,
            job_id=job_id
        )
    
    async def emit_system_health(self, health_data: Dict[str, Any]):
        """Emit system health event"""
        await self.emit(
            EventType.SYSTEM_HEALTH,
            {
                "health_data": health_data,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def emit_storage_action(
        self, 
        action: str, 
        key: str, 
        metadata: Dict[str, Any],
        user_id: Optional[str] = None
    ):
        """Emit storage action event"""
        await self.emit(
            EventType.STORAGE_ACTION,
            {
                "action": action,
                "key": key,
                "metadata": metadata,
                "timestamp": datetime.utcnow().isoformat()
            },
            user_id=user_id
        )
    
    async def emit_security_alert(
        self, 
        alert_type: str, 
        details: Dict[str, Any],
        user_id: Optional[str] = None
    ):
        """Emit security alert event"""
        await self.emit(
            EventType.SECURITY_ALERT,
            {
                "alert_type": alert_type,
                "details": details,
                "timestamp": datetime.utcnow().isoformat()
            },
            user_id=user_id
        )


class EventListener:
    """Event listener for consuming events"""
    
    def __init__(self):
        self.redis_client = Redis.from_url(settings.redis_url)
        self.channel_prefix = "ttv_events"
        self.callbacks = {}
    
    def subscribe(self, event_type: EventType, callback):
        """Subscribe to events of a specific type"""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)
    
    async def listen(self):
        """Start listening for events"""
        pubsub = self.redis_client.pubsub()
        
        # Subscribe to all TTV events
        await pubsub.subscribe(f"{self.channel_prefix}:all")
        
        logger.info("Started event listener")
        
        try:
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        event_data = json.loads(message['data'])
                        event_type = EventType(event_data['event_type'])
                        
                        # Call registered callbacks
                        if event_type in self.callbacks:
                            for callback in self.callbacks[event_type]:
                                try:
                                    await callback(event_data)
                                except Exception as e:
                                    logger.error(f"Error in event callback: {str(e)}")
                    
                    except Exception as e:
                        logger.error(f"Error processing event message: {str(e)}")
        
        except Exception as e:
            logger.error(f"Event listener error: {str(e)}")
        finally:
            await pubsub.unsubscribe(f"{self.channel_prefix}:all")


# Global event emitter instance
event_emitter = EventEmitter()


# Convenience functions
async def emit_event(
    event_type: EventType,
    data: Dict[str, Any],
    user_id: Optional[str] = None,
    job_id: Optional[str] = None
):
    """Emit an event using the global emitter"""
    await event_emitter.emit(event_type, data, user_id, job_id)


async def emit_job_event(
    event_type: EventType,
    job_id: str,
    user_id: str,
    data: Dict[str, Any]
):
    """Emit a job-related event"""
    await event_emitter.emit(event_type, data, user_id, job_id)