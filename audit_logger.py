"""
Audit Logger with KSML Compliance
==================================

Purpose:
    Enhanced audit logging with KSML token metadata:
    - Intent tracking
    - Karma state management
    - Lineage preservation
    - Operation audit trail
    
Architecture:
    - KSMLToken: Token data structure
    - AuditLogger: Main logging interface
    - AuditStorage: Persistent storage
    
Compliance:
    - Full KSML lineage tracking
    - Secure audit trail
    - Tamper-evident logs
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class KSMLToken:
    """KSML Token structure for lineage tracking."""
    ksml_token: str
    intent: str
    karma_state: str
    operation: str
    timestamp: str
    lineage: Dict
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def compute_hash(self) -> str:
        """Compute hash for tamper detection."""
        token_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(token_str.encode()).hexdigest()


class AuditLogger:
    """
    Enhanced audit logger with KSML compliance.
    
    Features:
        - KSML token tracking
        - Intent and karma state logging
        - Lineage preservation
        - Secure audit trail
        - Tamper-evident storage
    """
    
    def __init__(
        self,
        log_dir: str = "logs/audit",
        enable_console: bool = True
    ):
        """
        Initialize audit logger.
        
        Args:
            log_dir: Directory for audit logs
            enable_console: Enable console logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.enable_console = enable_console
        
        # Current log file
        self.current_log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        logger.info(f"📝 Audit logger initialized: {self.log_dir}")
    
    def log_operation(
        self,
        operation: str,
        ksml_token: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        status: str = "success"
    ) -> str:
        """
        Log an operation with KSML compliance.
        
        Args:
            operation: Operation name
            ksml_token: KSML token metadata
            metadata: Additional metadata
            status: Operation status
            
        Returns:
            Log entry ID
        """
        timestamp = datetime.now().isoformat()
        
        # Create log entry
        entry = {
            "entry_id": hashlib.md5(f"{timestamp}{operation}".encode()).hexdigest(),
            "timestamp": timestamp,
            "operation": operation,
            "status": status,
            "ksml_compliance": {
                "token": ksml_token.get("ksml_token") if ksml_token else None,
                "intent": ksml_token.get("intent") if ksml_token else None,
                "karma_state": ksml_token.get("karma_state") if ksml_token else None,
                "lineage": ksml_token.get("lineage") if ksml_token else None
            },
            "metadata": metadata or {},
            "hash": ""  # Computed after entry creation
        }
        
        # Compute entry hash for tamper detection
        entry_copy = entry.copy()
        entry_copy.pop("hash")
        entry["hash"] = hashlib.sha256(
            json.dumps(entry_copy, sort_keys=True).encode()
        ).hexdigest()
        
        # Write to log file
        with open(self.current_log_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")
        
        # Console output
        if self.enable_console:
            logger.info(f"📝 Audit: {operation} [{status}] - Token: {entry['ksml_compliance']['token']}")
        
        return entry["entry_id"]
    
    def log_video_generation(
        self,
        prompt: str,
        output_path: str,
        ksml_token: Optional[Dict] = None,
        quality_metrics: Optional[Dict] = None,
        processing_time: Optional[float] = None,
        security_metadata: Optional[Dict] = None
    ) -> str:
        """
        Log video generation with full metadata.
        
        Args:
            prompt: Input prompt
            output_path: Output video path
            ksml_token: KSML token
            quality_metrics: Quality metrics (VMAF, etc.)
            processing_time: Processing time in seconds
            security_metadata: Security fields (build_id, artifact_hash, watermark_id, signed)
            
        Returns:
            Log entry ID
        """
        metadata = {
            "prompt": prompt,
            "output_path": output_path,
            "quality_metrics": quality_metrics or {},
            "processing_time_seconds": processing_time,
            "security": security_metadata or {}
        }
        
        return self.log_operation(
            operation="video_generation",
            ksml_token=ksml_token,
            metadata=metadata,
            status="success"
        )
    
    def log_pipeline_stage(
        self,
        stage: str,
        input_data: Dict,
        output_data: Dict,
        ksml_token: Optional[Dict] = None
    ) -> str:
        """
        Log a pipeline stage execution.
        
        Args:
            stage: Pipeline stage name
            input_data: Input data summary
            output_data: Output data summary
            ksml_token: KSML token
            
        Returns:
            Log entry ID
        """
        metadata = {
            "stage": stage,
            "input": input_data,
            "output": output_data
        }
        
        return self.log_operation(
            operation=f"pipeline_{stage}",
            ksml_token=ksml_token,
            metadata=metadata
        )
    
    def log_error(
        self,
        operation: str,
        error: Exception,
        ksml_token: Optional[Dict] = None,
        context: Optional[Dict] = None
    ) -> str:
        """
        Log an error with context.
        
        Args:
            operation: Operation that failed
            error: Exception object
            ksml_token: KSML token
            context: Additional context
            
        Returns:
            Log entry ID
        """
        metadata = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        
        return self.log_operation(
            operation=operation,
            ksml_token=ksml_token,
            metadata=metadata,
            status="error"
        )
    
    # ===== Day 6: TTV Studio Intelligence Metrics =====
    
    def log_ttv_intelligence(
        self,
        lesson_name: str,
        story_analysis: Optional[Dict] = None,
        scene_graph: Optional[Dict] = None,
        narrative: Optional[Dict] = None,
        emotion: Optional[Dict] = None,
        extension: Optional[Dict] = None,
        quality: Optional[Dict] = None,
        ksml_token: Optional[Dict] = None
    ) -> str:
        """
        Log TTV Studio intelligence metrics (Day 6).
        
        Args:
            lesson_name: Name of lesson being processed
            story_analysis: Story analysis metrics (characters, gender, condensation)
            scene_graph: Scene graph metrics (scenes, entities, transitions)
            narrative: Narrative metrics (beats, arcs, tension)
            emotion: Emotion metrics (changes, motion intensity)
            extension: Extension metrics (clips extended, methods used)
            quality: Quality metrics (sync, frame quality, bitrate)
            ksml_token: KSML token
            
        Returns:
            Log entry ID
        """
        metadata = {
            "lesson_name": lesson_name,
            "ttv_metrics": {
                "story_analysis": story_analysis or {},
                "scene_graph": scene_graph or {},
                "narrative": narrative or {},
                "emotion": emotion or {},
                "extension": extension or {},
                "quality": quality or {}
            }
        }
        
        return self.log_operation(
            operation="ttv_intelligence_analysis",
            ksml_token=ksml_token,
            metadata=metadata
        )
    
    def log_performance_metric(
        self,
        module_name: str,
        execution_time: float,
        cache_hits: int = 0,
        cache_misses: int = 0,
        ksml_token: Optional[Dict] = None
    ) -> str:
        """
        Log performance metrics for TTV modules.
        
        Args:
            module_name: Name of module being measured
            execution_time: Execution time in seconds
            cache_hits: Number of cache hits
            cache_misses: Number of cache misses
            ksml_token: KSML token
            
        Returns:
            Log entry ID
        """
        cache_hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0.0
        
        metadata = {
            "module": module_name,
            "execution_time_seconds": execution_time,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": cache_hit_rate
        }
        
        return self.log_operation(
            operation="module_performance",
            ksml_token=ksml_token,
            metadata=metadata
        )
    
    def query_logs(
        self,
        operation: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        ksml_token: Optional[str] = None
    ) -> List[Dict]:
        """
        Query audit logs with filters.
        
        Args:
            operation: Filter by operation
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            ksml_token: Filter by KSML token
            
        Returns:
            List of matching log entries
        """
        results = []
        
        # Read all log files
        for log_file in sorted(self.log_dir.glob("audit_*.jsonl")):
            with open(log_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    
                    # Apply filters
                    if operation and entry["operation"] != operation:
                        continue
                    
                    if start_date and entry["timestamp"] < start_date:
                        continue
                    
                    if end_date and entry["timestamp"] > end_date:
                        continue
                    
                    if ksml_token and entry["ksml_compliance"]["token"] != ksml_token:
                        continue
                    
                    results.append(entry)
        
        return results
    
    def verify_integrity(self) -> Dict:
        """
        Verify integrity of audit logs.
        
        Returns:
            Verification results
        """
        total_entries = 0
        tampered_entries = 0
        corrupted_entries = 0
        
        for log_file in self.log_dir.glob("audit_*.jsonl"):
            with open(log_file, 'r') as f:
                for line in f:
                    total_entries += 1
                    
                    try:
                        entry = json.loads(line)
                        
                        # Verify hash
                        stored_hash = entry.pop("hash")
                        computed_hash = hashlib.sha256(
                            json.dumps(entry, sort_keys=True).encode()
                        ).hexdigest()
                        
                        if stored_hash != computed_hash:
                            tampered_entries += 1
                            logger.warning(f"⚠️ Tampered entry detected: {entry.get('entry_id')}")
                        
                    except json.JSONDecodeError:
                        corrupted_entries += 1
        
        return {
            "total_entries": total_entries,
            "valid_entries": total_entries - tampered_entries - corrupted_entries,
            "tampered_entries": tampered_entries,
            "corrupted_entries": corrupted_entries,
            "integrity_status": "PASS" if tampered_entries == 0 and corrupted_entries == 0 else "FAIL"
        }


# Global audit logger instance
_global_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(log_dir: str = "logs/audit") -> AuditLogger:
    """Get or create global audit logger instance."""
    global _global_audit_logger
    
    if _global_audit_logger is None:
        _global_audit_logger = AuditLogger(log_dir=log_dir)
    
    return _global_audit_logger


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Audit Logger CLI")
    parser.add_argument("--verify", action="store_true", help="Verify log integrity")
    parser.add_argument("--query", type=str, help="Query operation")
    parser.add_argument("--log_dir", type=str, default="logs/audit", help="Log directory")
    
    args = parser.parse_args()
    
    logger_instance = AuditLogger(log_dir=args.log_dir)
    
    if args.verify:
        print("\n🔍 Verifying audit log integrity...")
        results = logger_instance.verify_integrity()
        
        print(f"\n{'='*60}")
        print("Integrity Verification Results")
        print(f"{'='*60}")
        print(f"Total entries: {results['total_entries']}")
        print(f"Valid entries: {results['valid_entries']}")
        print(f"Tampered entries: {results['tampered_entries']}")
        print(f"Corrupted entries: {results['corrupted_entries']}")
        print(f"Status: {results['integrity_status']}")
    
    elif args.query:
        print(f"\n🔍 Querying logs for operation: {args.query}")
        entries = logger_instance.query_logs(operation=args.query)
        
        print(f"\nFound {len(entries)} matching entries:")
        for entry in entries[:10]:  # Show first 10
            print(f"  [{entry['timestamp']}] {entry['operation']} - {entry['status']}")
