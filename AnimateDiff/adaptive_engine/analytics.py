"""
Analytics and Reporting System for Task 4 Day 4
Cost/latency reporting and comprehensive system analytics
"""

import time
import json
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import threading


@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    request_id: str
    timestamp: float
    user_id: str
    tier_used: str
    response_time_seconds: float
    cost_usd: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System-wide metrics snapshot"""
    timestamp: float
    active_users: int
    queued_requests: int
    total_requests_today: int
    success_rate_percent: float
    average_response_time: float
    total_cost_today: float
    tier_usage: Dict[str, int] = field(default_factory=dict)
    error_breakdown: Dict[str, int] = field(default_factory=dict)


@dataclass
class CostReport:
    """Cost analysis report"""
    period_days: int
    total_cost_usd: float
    cost_per_request: float
    cost_by_tier: Dict[str, float]
    cost_trend: List[Tuple[str, float]]
    cost_efficiency_score: float
    recommendations: List[str]


@dataclass
class LatencyReport:
    """Latency analysis report"""
    period_hours: int
    average_latency_seconds: float
    p95_latency_seconds: float
    p99_latency_seconds: float
    latency_by_tier: Dict[str, float]
    latency_trend: List[Tuple[str, float]]
    performance_score: float
    bottlenecks: List[str]


class AnalyticsManager:
    """Comprehensive analytics and reporting system"""

    def __init__(self, data_dir: str = "analytics"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Data storage
        self.request_metrics: List[RequestMetrics] = []
        self.system_snapshots: List[SystemMetrics] = []

        # File paths
        self.requests_file = self.data_dir / "requests.jsonl"
        self.snapshots_file = self.data_dir / "snapshots.jsonl"

        # Load existing data
        self._load_data()

        # Start background monitoring
        self.monitoring_thread = threading.Thread(target=self._background_monitor, daemon=True)
        self.monitoring_thread.start()

    def _load_data(self):
        """Load existing analytics data"""
        # Load request metrics
        if self.requests_file.exists():
            try:
                with open(self.requests_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            self.request_metrics.append(RequestMetrics(**data))
            except Exception as e:
                print(f"Warning: Failed to load request metrics: {e}")

        # Load system snapshots
        if self.snapshots_file.exists():
            try:
                with open(self.snapshots_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            self.system_snapshots.append(SystemMetrics(**data))
            except Exception as e:
                print(f"Warning: Failed to load system snapshots: {e}")

    def record_request(self, metrics: RequestMetrics):
        """Record a request's metrics"""
        self.request_metrics.append(metrics)

        # Save to file
        try:
            with open(self.requests_file, 'a') as f:
                json.dump(metrics.__dict__, f)
                f.write('\n')
        except Exception as e:
            print(f"Warning: Failed to save request metrics: {e}")

        # Clean up old data (keep last 30 days)
        cutoff_time = time.time() - (30 * 24 * 60 * 60)
        self.request_metrics = [m for m in self.request_metrics if m.timestamp > cutoff_time]

    def record_system_snapshot(self, metrics: SystemMetrics):
        """Record system metrics snapshot"""
        self.system_snapshots.append(metrics)

        # Save to file
        try:
            with open(self.snapshots_file, 'a') as f:
                json.dump(metrics.__dict__, f)
                f.write('\n')
        except Exception as e:
            print(f"Warning: Failed to save system snapshot: {e}")

        # Clean up old snapshots (keep last 7 days)
        cutoff_time = time.time() - (7 * 24 * 60 * 60)
        self.system_snapshots = [s for s in self.system_snapshots if s.timestamp > cutoff_time]

    def _background_monitor(self):
        """Background monitoring thread"""
        while True:
            try:
                # Take system snapshot every 60 seconds
                snapshot = self._create_system_snapshot()
                self.record_system_snapshot(snapshot)

                # Clean up old data every hour
                if int(time.time()) % 3600 == 0:
                    self._cleanup_old_data()

            except Exception as e:
                print(f"Warning: Background monitoring error: {e}")

            time.sleep(60)  # Check every minute

    def _create_system_snapshot(self) -> SystemMetrics:
        """Create a current system metrics snapshot"""
        # Get recent requests (last hour)
        one_hour_ago = time.time() - 3600
        recent_requests = [r for r in self.request_metrics if r.timestamp > one_hour_ago]

        if recent_requests:
            success_count = sum(1 for r in recent_requests if r.success)
            success_rate = (success_count / len(recent_requests)) * 100
            avg_response_time = statistics.mean(r.response_time_seconds for r in recent_requests)
            total_cost = sum(r.cost_usd for r in recent_requests)
        else:
            success_rate = 100.0
            avg_response_time = 0.0
            total_cost = 0.0

        # Tier usage
        tier_usage = {}
        for request in recent_requests:
            tier = request.tier_used
            tier_usage[tier] = tier_usage.get(tier, 0) + 1

        # Error breakdown
        error_breakdown = {}
        for request in recent_requests:
            if not request.success and request.error_message:
                error_type = request.error_message.split(':')[0] if ':' in request.error_message else 'unknown'
                error_breakdown[error_type] = error_breakdown.get(error_type, 0) + 1

        return SystemMetrics(
            timestamp=time.time(),
            active_users=0,  # Would be populated by load tester
            queued_requests=0,  # Would be populated by queue manager
            total_requests_today=len(recent_requests),
            success_rate_percent=success_rate,
            average_response_time=avg_response_time,
            total_cost_today=total_cost,
            tier_usage=tier_usage,
            error_breakdown=error_breakdown
        )

    def _cleanup_old_data(self):
        """Clean up old analytics data"""
        # Keep only last 30 days of requests
        cutoff_30_days = time.time() - (30 * 24 * 60 * 60)
        self.request_metrics = [r for r in self.request_metrics if r.timestamp > cutoff_30_days]

        # Keep only last 7 days of snapshots
        cutoff_7_days = time.time() - (7 * 24 * 60 * 60)
        self.system_snapshots = [s for s in self.system_snapshots if s.timestamp > cutoff_7_days]

        # Rewrite files with cleaned data
        try:
            with open(self.requests_file, 'w') as f:
                for request in self.request_metrics:
                    json.dump(request.__dict__, f)
                    f.write('\n')

            with open(self.snapshots_file, 'w') as f:
                for snapshot in self.system_snapshots:
                    json.dump(snapshot.__dict__, f)
                    f.write('\n')
        except Exception as e:
            print(f"Warning: Failed to cleanup analytics data: {e}")

    def generate_cost_report(self, days: int = 7) -> CostReport:
        """Generate cost analysis report"""
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        relevant_requests = [r for r in self.request_metrics if r.timestamp > cutoff_time]

        if not relevant_requests:
            return CostReport(
                period_days=days,
                total_cost_usd=0.0,
                cost_per_request=0.0,
                cost_by_tier={},
                cost_trend=[],
                cost_efficiency_score=100.0,
                recommendations=["No data available for the specified period"]
            )

        # Calculate metrics
        total_cost = sum(r.cost_usd for r in relevant_requests)
        cost_per_request = total_cost / len(relevant_requests)

        # Cost by tier
        cost_by_tier = {}
        for request in relevant_requests:
            tier = request.tier_used
            cost_by_tier[tier] = cost_by_tier.get(tier, 0.0) + request.cost_usd

        # Cost trend (daily)
        cost_trend = []
        for day_offset in range(days):
            day_start = time.time() - ((days - day_offset) * 24 * 60 * 60)
            day_end = time.time() - ((days - day_offset - 1) * 24 * 60 * 60)

            day_requests = [r for r in relevant_requests if day_start <= r.timestamp < day_end]
            day_cost = sum(r.cost_usd for r in day_requests)

            day_str = datetime.fromtimestamp(day_start).strftime("%Y-%m-%d")
            cost_trend.append((day_str, day_cost))

        # Cost efficiency score (lower cost per request = higher score)
        base_cost_per_request = 0.05  # Baseline expectation
        efficiency_ratio = base_cost_per_request / cost_per_request if cost_per_request > 0 else 1.0
        cost_efficiency_score = min(100.0, efficiency_ratio * 50)  # Scale to 0-100

        # Recommendations
        recommendations = []
        if cost_per_request > 0.10:
            recommendations.append("High cost per request - consider optimizing local processing")
        if "yotta" in cost_by_tier and cost_by_tier["yotta"] > total_cost * 0.5:
            recommendations.append("Heavy Yotta usage - optimize local/office GPU utilization")
        if cost_efficiency_score < 60:
            recommendations.append("Cost efficiency below optimal - review tier routing logic")

        return CostReport(
            period_days=days,
            total_cost_usd=total_cost,
            cost_per_request=cost_per_request,
            cost_by_tier=cost_by_tier,
            cost_trend=cost_trend,
            cost_efficiency_score=cost_efficiency_score,
            recommendations=recommendations
        )

    def generate_latency_report(self, hours: int = 24) -> LatencyReport:
        """Generate latency analysis report"""
        cutoff_time = time.time() - (hours * 60 * 60)
        relevant_requests = [r for r in self.request_metrics if r.timestamp > cutoff_time]

        if not relevant_requests:
            return LatencyReport(
                period_hours=hours,
                average_latency_seconds=0.0,
                p95_latency_seconds=0.0,
                p99_latency_seconds=0.0,
                latency_by_tier={},
                latency_trend=[],
                performance_score=100.0,
                bottlenecks=[]
            )

        # Calculate latency metrics
        latencies = [r.response_time_seconds for r in relevant_requests]
        average_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
        p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies)

        # Latency by tier
        latency_by_tier = {}
        tier_counts = {}
        for request in relevant_requests:
            tier = request.tier_used
            if tier not in latency_by_tier:
                latency_by_tier[tier] = 0.0
                tier_counts[tier] = 0
            latency_by_tier[tier] += request.response_time_seconds
            tier_counts[tier] += 1

        for tier in latency_by_tier:
            latency_by_tier[tier] /= tier_counts[tier]

        # Latency trend (hourly)
        latency_trend = []
        for hour_offset in range(hours):
            hour_start = time.time() - ((hours - hour_offset) * 60 * 60)
            hour_end = time.time() - ((hours - hour_offset - 1) * 60 * 60)

            hour_requests = [r for r in relevant_requests if hour_start <= r.timestamp < hour_end]
            if hour_requests:
                hour_avg_latency = statistics.mean(r.response_time_seconds for r in hour_requests)
            else:
                hour_avg_latency = 0.0

            hour_str = datetime.fromtimestamp(hour_start).strftime("%H:00")
            latency_trend.append((hour_str, hour_avg_latency))

        # Performance score (lower latency = higher score)
        target_latency = 10.0  # 10 seconds target
        performance_ratio = target_latency / average_latency if average_latency > 0 else 1.0
        performance_score = min(100.0, performance_ratio * 50)  # Scale to 0-100

        # Identify bottlenecks
        bottlenecks = []
        if average_latency > 15.0:
            bottlenecks.append("High average latency - consider faster tiers")
        if p95_latency > 30.0:
            bottlenecks.append("High P95 latency - optimize for consistency")
        if "yotta" in latency_by_tier and latency_by_tier["yotta"] > 20.0:
            bottlenecks.append("Yotta cloud latency high - optimize local processing")

        return LatencyReport(
            period_hours=hours,
            average_latency_seconds=average_latency,
            p95_latency_seconds=p95_latency,
            p99_latency_seconds=p99_latency,
            latency_by_tier=latency_by_tier,
            latency_trend=latency_trend,
            performance_score=performance_score,
            bottlenecks=bottlenecks
        )

    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics"""
        # Get recent data (last hour)
        one_hour_ago = time.time() - 3600
        recent_requests = [r for r in self.request_metrics if r.timestamp > one_hour_ago]
        recent_snapshots = [s for s in self.system_snapshots if s.timestamp > one_hour_ago]

        if not recent_requests:
            return {
                "status": "unknown",
                "message": "No recent request data available"
            }

        # Calculate health metrics
        success_rate = sum(1 for r in recent_requests if r.success) / len(recent_requests) * 100
        avg_latency = statistics.mean(r.response_time_seconds for r in recent_requests)
        error_rate = 100 - success_rate

        # Determine status
        if error_rate > 20:
            status = "critical"
            message = f"High error rate: {error_rate:.1f}%"
        elif error_rate > 10:
            status = "warning"
            message = f"Elevated error rate: {error_rate:.1f}%"
        elif avg_latency > 20:
            status = "warning"
            message = f"High latency: {avg_latency:.1f}s"
        else:
            status = "healthy"
            message = "System operating normally"

        return {
            "status": status,
            "message": message,
            "success_rate_percent": success_rate,
            "average_latency_seconds": avg_latency,
            "requests_per_minute": len(recent_requests),
            "active_users": recent_snapshots[-1].active_users if recent_snapshots else 0
        }


# Global analytics instance
_analytics = None

def get_analytics() -> AnalyticsManager:
    """Get global analytics instance"""
    global _analytics
    if _analytics is None:
        _analytics = AnalyticsManager()
    return _analytics