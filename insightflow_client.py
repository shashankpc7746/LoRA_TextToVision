"""
InsightFlow Telemetry Integration
==================================

Purpose:
    Telemetry and observability for TTV pipeline:
    - Real-time metrics emission
    - Performance monitoring
    - Error tracking
    - Usage analytics
    
Architecture:
    - InsightFlowClient: Telemetry client
    - MetricsCollector: Metrics aggregation
    - EventEmitter: Event streaming
    
Compliance:
    - KSML lineage integration
    - Privacy-preserving metrics
    - Secure transmission
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TelemetryEvent:
    """Telemetry event structure."""
    event_type: str
    timestamp: str
    component: str
    metrics: Dict[str, Any]
    ksml_token: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class MetricsCollector:
    """
    Metrics collector for aggregation.
    
    Features:
        - Real-time metric tracking
        - Sliding window aggregation
        - Percentile calculation
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics collector.
        
        Args:
            window_size: Number of samples to keep for aggregation
        """
        self.window_size = window_size
        self.metrics = defaultdict(list)
        
    def record(self, metric_name: str, value: float):
        """
        Record a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
        """
        self.metrics[metric_name].append(value)
        
        # Keep only recent values
        if len(self.metrics[metric_name]) > self.window_size:
            self.metrics[metric_name] = self.metrics[metric_name][-self.window_size:]
    
    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """
        Get statistics for a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Dictionary with mean, min, max, p50, p95, p99
        """
        values = self.metrics.get(metric_name, [])
        
        if not values:
            return {}
        
        import numpy as np
        
        return {
            "count": len(values),
            "mean": np.mean(values),
            "min": np.min(values),
            "max": np.max(values),
            "p50": np.percentile(values, 50),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99)
        }
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all metrics."""
        return {
            metric_name: self.get_stats(metric_name)
            for metric_name in self.metrics.keys()
        }


class InsightFlowClient:
    """
    InsightFlow telemetry client.
    
    Features:
        - Event emission
        - Metrics tracking
        - Performance monitoring
        - Error tracking
        - KSML integration
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        enable_local_logging: bool = True,
        log_dir: str = "logs/telemetry"
    ):
        """
        Initialize InsightFlow client.
        
        Args:
            endpoint: InsightFlow API endpoint (optional)
            api_key: API key for authentication (optional)
            enable_local_logging: Enable local log files
            log_dir: Directory for local logs
        """
        self.endpoint = endpoint or os.getenv("INSIGHTFLOW_ENDPOINT")
        self.api_key = api_key or os.getenv("INSIGHTFLOW_API_KEY")
        self.enable_local_logging = enable_local_logging
        self.log_dir = Path(log_dir)
        
        if self.enable_local_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.current_log_file = self.log_dir / f"telemetry_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        self.metrics_collector = MetricsCollector()
        self.session_id = f"session_{int(time.time())}"
        
        # Connection status
        self.is_connected = self.endpoint is not None and self.api_key is not None
        
        if self.is_connected:
            logger.info(f"📊 InsightFlow client connected: {self.endpoint}")
        else:
            logger.info("📊 InsightFlow client in local mode (no remote endpoint)")
    
    def emit(
        self,
        event_type: str,
        component: str,
        metrics: Dict[str, Any],
        ksml_token: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Emit a telemetry event.
        
        Args:
            event_type: Type of event (e.g., "video_generation", "frame_upscale")
            component: Component name (e.g., "indigenous_adapter", "upscaler")
            metrics: Metrics dictionary
            ksml_token: KSML token for lineage
            user_id: User identifier
        """
        timestamp = datetime.now().isoformat()
        
        # Create event
        event = TelemetryEvent(
            event_type=event_type,
            timestamp=timestamp,
            component=component,
            metrics=metrics,
            ksml_token=ksml_token,
            user_id=user_id,
            session_id=self.session_id
        )
        
        # Record numeric metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.metrics_collector.record(f"{component}.{key}", value)
        
        # Local logging
        if self.enable_local_logging:
            with open(self.current_log_file, 'a') as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        
        # Remote emission (stub for now)
        if self.is_connected:
            self._emit_remote(event)
        
        logger.debug(f"📊 Telemetry: {event_type} from {component}")
    
    def _emit_remote(self, event: TelemetryEvent):
        """
        Emit event to remote InsightFlow endpoint.
        
        This is a stub implementation. In production:
        - Use HTTP POST to InsightFlow API
        - Implement retry logic
        - Buffer events for batch sending
        - Handle authentication
        """
        # Stub: In production, would send to remote endpoint
        # Example:
        # import requests
        # requests.post(
        #     f"{self.endpoint}/events",
        #     headers={"Authorization": f"Bearer {self.api_key}"},
        #     json=event.to_dict()
        # )
        pass
    
    def emit_pipeline_stage(
        self,
        stage: str,
        duration: float,
        input_size: Optional[int] = None,
        output_size: Optional[int] = None,
        ksml_token: Optional[str] = None
    ):
        """
        Emit pipeline stage metrics.
        
        Args:
            stage: Pipeline stage name
            duration: Duration in seconds
            input_size: Input data size (bytes)
            output_size: Output data size (bytes)
            ksml_token: KSML token
        """
        metrics = {
            "duration_seconds": duration,
            "input_size_bytes": input_size,
            "output_size_bytes": output_size
        }
        
        self.emit(
            event_type="pipeline_stage",
            component=stage,
            metrics=metrics,
            ksml_token=ksml_token
        )
    
    def emit_video_generation(
        self,
        prompt: str,
        duration: float,
        num_frames: int,
        resolution: str,
        quality_score: Optional[float] = None,
        cost_usd: Optional[float] = None,
        ksml_token: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Emit video generation event.
        
        Args:
            prompt: Input prompt
            duration: Processing duration (seconds)
            num_frames: Number of frames generated
            resolution: Output resolution (e.g., "1920x1080")
            quality_score: Quality metric (e.g., VMAF)
            cost_usd: Estimated cost in USD
            ksml_token: KSML token
            user_id: User identifier
        """
        metrics = {
            "prompt_length": len(prompt),
            "duration_seconds": duration,
            "num_frames": num_frames,
            "resolution": resolution,
            "quality_score": quality_score,
            "cost_usd": cost_usd,
            "fps": num_frames / duration if duration > 0 else 0
        }
        
        self.emit(
            event_type="video_generation",
            component="ttv_pipeline",
            metrics=metrics,
            ksml_token=ksml_token,
            user_id=user_id
        )
    
    def emit_error(
        self,
        component: str,
        error_type: str,
        error_message: str,
        context: Optional[Dict] = None,
        ksml_token: Optional[str] = None
    ):
        """
        Emit error event.
        
        Args:
            component: Component where error occurred
            error_type: Type of error
            error_message: Error message
            context: Additional context
            ksml_token: KSML token
        """
        metrics = {
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {}
        }
        
        self.emit(
            event_type="error",
            component=component,
            metrics=metrics,
            ksml_token=ksml_token
        )
    
    def get_metrics_summary(self) -> Dict:
        """
        Get summary of collected metrics.
        
        Returns:
            Dictionary with metric statistics
        """
        return {
            "session_id": self.session_id,
            "metrics": self.metrics_collector.get_all_stats()
        }
    
    def flush(self):
        """
        Flush any buffered events.
        
        In production implementation:
        - Send any buffered events to remote endpoint
        - Close connections
        - Finalize local log files
        """
        if self.enable_local_logging:
            logger.info(f"📊 Telemetry logs written to {self.log_dir}")


# Global InsightFlow client instance
_global_insightflow_client: Optional[InsightFlowClient] = None


def get_insightflow_client(
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None
) -> InsightFlowClient:
    """Get or create global InsightFlow client instance."""
    global _global_insightflow_client
    
    if _global_insightflow_client is None:
        _global_insightflow_client = InsightFlowClient(
            endpoint=endpoint,
            api_key=api_key
        )
    
    return _global_insightflow_client


# Convenience functions
def emit_telemetry(
    event_type: str,
    component: str,
    metrics: Dict[str, Any],
    **kwargs
):
    """Convenience function to emit telemetry."""
    client = get_insightflow_client()
    client.emit(event_type, component, metrics, **kwargs)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="InsightFlow Telemetry CLI")
    parser.add_argument("--test", action="store_true", help="Run test emissions")
    parser.add_argument("--summary", action="store_true", help="Show metrics summary")
    parser.add_argument("--log_dir", type=str, default="logs/telemetry", help="Log directory")
    
    args = parser.parse_args()
    
    client = InsightFlowClient(log_dir=args.log_dir)
    
    if args.test:
        print("\n📊 Running test telemetry emissions...")
        
        # Test video generation event
        client.emit_video_generation(
            prompt="Test Gurukul classroom scene",
            duration=45.5,
            num_frames=240,
            resolution="1920x1080",
            quality_score=87.5,
            cost_usd=0.05,
            ksml_token="test_token_123"
        )
        
        # Test pipeline stage event
        client.emit_pipeline_stage(
            stage="upscaler",
            duration=12.3,
            input_size=1024*1024*50,
            output_size=1024*1024*200,
            ksml_token="test_token_123"
        )
        
        # Test error event
        client.emit_error(
            component="indigenous_adapter",
            error_type="CUDAOutOfMemoryError",
            error_message="CUDA out of memory",
            context={"batch_size": 4, "gpu": "cuda:0"}
        )
        
        print("✅ Test emissions complete")
    
    if args.summary:
        print("\n📊 Metrics Summary")
        print(f"{'='*60}")
        
        summary = client.get_metrics_summary()
        print(f"Session ID: {summary['session_id']}")
        print(f"\nMetrics:")
        
        for metric_name, stats in summary['metrics'].items():
            print(f"\n  {metric_name}:")
            for stat_name, value in stats.items():
                print(f"    {stat_name}: {value:.2f}")
