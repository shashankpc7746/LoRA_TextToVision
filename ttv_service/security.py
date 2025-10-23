"""
TTV Service Security and Authentication
Implements Supabase JWT validation, content moderation, and audit logging
"""

import json
import logging
import hashlib
import hmac
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from functools import wraps
import re

import jwt
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlmodel import SQLModel, Field, Session, create_engine, select

from .config import settings
from .events import emit_event, EventType


logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()


class SecurityException(Exception):
    """Custom security exception"""
    pass


class ContentModerationException(Exception):
    """Content moderation exception"""
    pass


# Audit Log Model
class AuditLog(SQLModel, table=True):
    """Audit log for GDPR compliance and security tracking"""
    __tablename__ = "ttv_audit_logs"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[str] = Field(index=True)
    action: str = Field(index=True)
    resource_type: str = Field(index=True)
    resource_id: Optional[str] = Field(index=True)
    details: str = Field()  # JSON string
    ip_address: Optional[str] = Field()
    user_agent: Optional[str] = Field()
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    success: bool = Field(default=True, index=True)
    error_message: Optional[str] = Field()


# User Session Model
class UserSession(SQLModel, table=True):
    """User session tracking"""
    __tablename__ = "ttv_user_sessions"
    
    id: Optional[str] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True)
    jwt_token_hash: str = Field()
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(index=True)
    ip_address: Optional[str] = Field()
    user_agent: Optional[str] = Field()
    is_active: bool = Field(default=True, index=True)


class JWTValidator:
    """JWT token validation for Supabase integration"""
    
    def __init__(self):
        self.secret = settings.supabase_jwt_secret or settings.jwt_secret_key
        self.algorithm = settings.jwt_algorithm
        self.db_engine = create_engine(settings.database_config['url'])
        
        # Create tables
        SQLModel.metadata.create_all(self.db_engine)
    
    async def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token and return user info"""
        try:
            # Decode JWT
            payload = jwt.decode(
                token, 
                self.secret, 
                algorithms=[self.algorithm],
                options={"verify_exp": True}
            )
            
            # Check token blacklist/session
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            
            with Session(self.db_engine) as session:
                session_query = select(UserSession).where(
                    UserSession.jwt_token_hash == token_hash,
                    UserSession.is_active == True,
                    UserSession.expires_at > datetime.utcnow()
                )
                session_record = session.exec(session_query).first()
                
                if not session_record:
                    raise SecurityException("Invalid or expired session")
                
                # Update last accessed
                session_record.last_accessed = datetime.utcnow()
                session.add(session_record)
                session.commit()
            
            return {
                "user_id": payload.get("sub"),
                "email": payload.get("email"),
                "role": payload.get("role", "user"),
                "session_id": session_record.id
            }
            
        except jwt.ExpiredSignatureError:
            raise SecurityException("Token has expired")
        except jwt.InvalidTokenError:
            raise SecurityException("Invalid token")
        except Exception as e:
            logger.error(f"Token validation error: {str(e)}")
            raise SecurityException("Token validation failed")
    
    async def create_session(
        self, 
        token: str, 
        user_id: str, 
        ip_address: str = None,
        user_agent: str = None
    ) -> str:
        """Create a new user session"""
        try:
            # Decode token to get expiry
            payload = jwt.decode(
                token,
                self.secret,
                algorithms=[self.algorithm],
                options={"verify_exp": False}  # We'll check manually
            )
            
            expires_at = datetime.fromtimestamp(payload.get("exp", 0))
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            
            session_id = f"ttv_{user_id}_{int(datetime.utcnow().timestamp())}"
            
            with Session(self.db_engine) as session:
                user_session = UserSession(
                    id=session_id,
                    user_id=user_id,
                    jwt_token_hash=token_hash,
                    expires_at=expires_at,
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                session.add(user_session)
                session.commit()
            
            return session_id
            
        except Exception as e:
            logger.error(f"Session creation error: {str(e)}")
            raise SecurityException("Failed to create session")
    
    async def invalidate_session(self, session_id: str):
        """Invalidate a user session"""
        with Session(self.db_engine) as session:
            session_query = select(UserSession).where(UserSession.id == session_id)
            session_record = session.exec(session_query).first()
            
            if session_record:
                session_record.is_active = False
                session.add(session_record)
                session.commit()


class ContentModerator:
    """Content moderation for TTV requests"""
    
    def __init__(self):
        self.forbidden_keywords = [word.lower() for word in settings.forbidden_keywords]
        self.max_script_length = settings.max_script_length
    
    async def moderate_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Moderate content and return moderation result"""
        script = content.get("script", "")
        violations = []
        
        # Check script length
        if len(script) > self.max_script_length:
            violations.append({
                "type": "length_violation",
                "message": f"Script exceeds maximum length of {self.max_script_length} characters",
                "severity": "error"
            })
        
        # Check for forbidden keywords
        script_lower = script.lower()
        found_keywords = []
        
        for keyword in self.forbidden_keywords:
            if keyword in script_lower:
                found_keywords.append(keyword)
        
        if found_keywords:
            violations.append({
                "type": "content_violation",
                "message": f"Content contains forbidden keywords: {', '.join(found_keywords)}",
                "severity": "error",
                "keywords": found_keywords
            })
        
        # Check for potential harmful patterns
        harmful_patterns = [
            r'\b(kill|murder|violence|harm)\b',
            r'\b(hate|racist|sexist)\b',
            r'\b(illegal|drugs|weapons)\b'
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, script_lower):
                violations.append({
                    "type": "pattern_violation",
                    "message": "Content contains potentially harmful language",
                    "severity": "warning",
                    "pattern": pattern
                })
        
        # Additional AI-based moderation could be added here
        # For now, using rule-based approach
        
        is_approved = len([v for v in violations if v["severity"] == "error"]) == 0
        
        result = {
            "approved": is_approved,
            "violations": violations,
            "score": self._calculate_safety_score(violations),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return result
    
    def _calculate_safety_score(self, violations: List[Dict[str, Any]]) -> float:
        """Calculate safety score (0-1, higher is safer)"""
        if not violations:
            return 1.0
        
        error_count = len([v for v in violations if v["severity"] == "error"])
        warning_count = len([v for v in violations if v["severity"] == "warning"])
        
        # Simple scoring algorithm
        score = 1.0 - (error_count * 0.5 + warning_count * 0.2)
        return max(0.0, score)


class AuditLogger:
    """Audit logging for GDPR compliance"""
    
    def __init__(self):
        self.db_engine = create_engine(settings.database_config['url'])
        SQLModel.metadata.create_all(self.db_engine)
    
    async def log_action(
        self,
        user_id: Optional[str],
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        details: Dict[str, Any] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Log an action for audit purposes"""
        try:
            with Session(self.db_engine) as session:
                audit_log = AuditLog(
                    user_id=user_id,
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    details=json.dumps(details or {}),
                    ip_address=ip_address,
                    user_agent=user_agent,
                    success=success,
                    error_message=error_message
                )
                session.add(audit_log)
                session.commit()
                
                # Emit security event for monitoring
                await emit_event(
                    EventType.SECURITY_ALERT if not success else EventType.STORAGE_ACTION,
                    {
                        "action": action,
                        "resource_type": resource_type,
                        "resource_id": resource_id,
                        "success": success,
                        "user_id": user_id
                    },
                    user_id=user_id
                )
                
        except Exception as e:
            logger.error(f"Audit logging error: {str(e)}")
    
    async def get_user_audit_logs(
        self, 
        user_id: str, 
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get audit logs for a user (GDPR compliance)"""
        with Session(self.db_engine) as session:
            query = (
                select(AuditLog)
                .where(AuditLog.user_id == user_id)
                .order_by(AuditLog.timestamp.desc())
                .offset(offset)
                .limit(limit)
            )
            logs = session.exec(query).all()
            
            return [
                {
                    "id": log.id,
                    "action": log.action,
                    "resource_type": log.resource_type,
                    "resource_id": log.resource_id,
                    "timestamp": log.timestamp.isoformat(),
                    "success": log.success,
                    "details": json.loads(log.details) if log.details else {}
                }
                for log in logs
            ]
    
    async def delete_user_data(self, user_id: str) -> bool:
        """Delete user audit data (GDPR right to be forgotten)"""
        try:
            with Session(self.db_engine) as session:
                # Anonymize instead of deleting to maintain audit integrity
                query = select(AuditLog).where(AuditLog.user_id == user_id)
                logs = session.exec(query).all()
                
                for log in logs:
                    log.user_id = f"deleted_{hashlib.sha256(user_id.encode()).hexdigest()[:8]}"
                    session.add(log)
                
                session.commit()
                
                logger.info(f"Anonymized audit logs for user {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting user data: {str(e)}")
            return False


# Global instances
jwt_validator = JWTValidator()
content_moderator = ContentModerator()
audit_logger = AuditLogger()


# Dependency functions for FastAPI
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict[str, Any]:
    """FastAPI dependency to get current authenticated user"""
    try:
        token = credentials.credentials
        user_info = await jwt_validator.validate_token(token)
        return user_info
    except SecurityException as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(status_code=401, detail="Authentication failed")


async def get_admin_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """FastAPI dependency to ensure user has admin role"""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return current_user


async def moderate_request_content(content: Dict[str, Any]) -> Dict[str, Any]:
    """FastAPI dependency for content moderation"""
    if not settings.content_moderation_enabled:
        return content
    
    try:
        moderation_result = await content_moderator.moderate_content(content)
        
        if not moderation_result["approved"]:
            error_violations = [v for v in moderation_result["violations"] if v["severity"] == "error"]
            error_messages = [v["message"] for v in error_violations]
            
            raise ContentModerationException(
                f"Content moderation failed: {'; '.join(error_messages)}"
            )
        
        return content
        
    except ContentModerationException:
        raise
    except Exception as e:
        logger.error(f"Content moderation error: {str(e)}")
        # Fail closed - reject if moderation fails
        raise HTTPException(status_code=400, detail="Content moderation failed")


# Decorator for audit logging
def audit_action(action: str, resource_type: str):
    """Decorator to automatically log actions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user info and request details
            user_id = None
            ip_address = None
            user_agent = None
            
            # Try to get user info from kwargs
            for arg in kwargs.values():
                if isinstance(arg, dict) and "user_id" in arg:
                    user_id = arg["user_id"]
                    break
            
            # Try to get request info
            for arg in args:
                if hasattr(arg, "client") and hasattr(arg.client, "host"):
                    ip_address = arg.client.host
                if hasattr(arg, "headers"):
                    user_agent = arg.headers.get("user-agent")
                    break
            
            try:
                result = await func(*args, **kwargs)
                
                # Log successful action
                await audit_logger.log_action(
                    user_id=user_id,
                    action=action,
                    resource_type=resource_type,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    success=True
                )
                
                return result
                
            except Exception as e:
                # Log failed action
                await audit_logger.log_action(
                    user_id=user_id,
                    action=action,
                    resource_type=resource_type,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    success=False,
                    error_message=str(e)
                )
                raise
        
        return wrapper
    return decorator


# Rate limiting
class RateLimiter:
    """Simple rate limiter using Redis"""
    
    def __init__(self):
        from redis import Redis
        self.redis_client = Redis.from_url(settings.redis_url)
        self.requests_per_minute = settings.rate_limit_requests_per_minute
        self.burst_limit = settings.rate_limit_burst
    
    async def check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limits"""
        try:
            now = datetime.utcnow()
            minute_key = f"rate_limit:{user_id}:{now.strftime('%Y-%m-%d-%H-%M')}"
            burst_key = f"rate_limit_burst:{user_id}"
            
            # Check minute limit
            minute_count = self.redis_client.incr(minute_key)
            if minute_count == 1:
                self.redis_client.expire(minute_key, 60)
            
            if minute_count > self.requests_per_minute:
                return False
            
            # Check burst limit
            burst_count = self.redis_client.incr(burst_key)
            if burst_count == 1:
                self.redis_client.expire(burst_key, 10)  # 10 second window
            
            if burst_count > self.burst_limit:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limiting error: {str(e)}")
            # Fail open - allow if rate limiting fails
            return True


rate_limiter = RateLimiter()


async def check_rate_limit(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """FastAPI dependency for rate limiting"""
    user_id = current_user["user_id"]
    
    if not await rate_limiter.check_rate_limit(user_id):
        await emit_event(
            EventType.SECURITY_ALERT,
            {
                "alert_type": "rate_limit_exceeded",
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            user_id=user_id
        )
        
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    
    return current_user