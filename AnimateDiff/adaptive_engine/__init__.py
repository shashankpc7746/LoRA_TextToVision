#!/usr/bin/env python3
"""
Adaptive Engine Package - Task-4 Day-1 + Day-2
Intelligent video generation system with device-aware processing, caching, and quality optimization
"""

from .device_probe import (  # type: ignore
    get_device_capabilities,  # type: ignore
    can_handle_heavy_workload,  # type: ignore
    DeviceProbe,  # type: ignore
    device_probe  # type: ignore
)

from .budget_planner import (  # type: ignore
    plan_video_quality,  # type: ignore
    BudgetPlanner,  # type: ignore
    BudgetConstraints,  # type: ignore
    budget_planner  # type: ignore
)

from .tier_router import (  # type: ignore
    route_generation_task,  # type: ignore
    TierRouter,  # type: ignore
    RoutingDecision,  # type: ignore
    tier_router  # type: ignore
)

from .workload_analyzer import (  # type: ignore
    analyze_generation_task,  # type: ignore
    WorkloadAnalyzer,  # type: ignore
    TaskAnalysis,  # type: ignore
    workload_analyzer  # type: ignore
)

# Day 2 Components
from .cache_manager import (  # type: ignore
    get_cache_manager,  # type: ignore
    CacheManager,  # type: ignore
    CacheEntry  # type: ignore
)

from .rl_policy import (  # type: ignore
    get_rl_policy,  # type: ignore
    RLPolicy,  # type: ignore
    Action,  # type: ignore
    State,  # type: ignore
    Experience  # type: ignore
)

from .compression_engine import (  # type: ignore
    get_compression_engine,  # type: ignore
    CompressionEngine,  # type: ignore
    CompressionPreset  # type: ignore
)

from .quality_assessor import (  # type: ignore
    get_quality_assessor,  # type: ignore
    QualityAssessor,  # type: ignore
    QualityMetrics  # type: ignore
)

from .adaptive_pipeline import (  # type: ignore
    get_adaptive_pipeline,  # type: ignore
    process_adaptive_request,  # type: ignore
    AdaptivePipeline,  # type: ignore
    PipelineResult  # type: ignore
)

# Day 3 Components
from .nas_storage import (  # type: ignore
    get_nas_storage,  # type: ignore
    NASStorageManager  # type: ignore
)

from .gpu_queue import (  # type: ignore
    get_gpu_queue,  # type: ignore
    GPUQueueManager,  # type: ignore
    GPUJob,  # type: ignore
    JobStatus,  # type: ignore
    JobPriority  # type: ignore
)

from .mixed_precision import (  # type: ignore
    get_mixed_precision,  # type: ignore
    MixedPrecisionManager,  # type: ignore
    PrecisionConfig,  # type: ignore
    PrecisionMode,  # type: ignore
    DeviceType  # type: ignore
)

from .lip_sync import (  # type: ignore
    get_lip_sync,  # type: ignore
    LipSyncManager,  # type: ignore
    LipSyncConfig,  # type: ignore
    LipSyncResult  # type: ignore
)

# Day 4 Components
from .load_tester import (  # type: ignore
    get_load_tester,  # type: ignore
    get_degradation_manager,  # type: ignore
    LoadTester,  # type: ignore
    GracefulDegradationManager,  # type: ignore
    LoadTestResult,  # type: ignore
    SimulatedUser  # type: ignore
)

from .analytics import (  # type: ignore
    get_analytics,  # type: ignore
    AnalyticsManager,  # type: ignore
    RequestMetrics,  # type: ignore
    SystemMetrics,  # type: ignore
    CostReport,  # type: ignore
    LatencyReport  # type: ignore
)

__version__ = "2.0.0"
__all__ = [
    # Device Probe
    "get_device_capabilities",
    "can_handle_heavy_workload",
    "DeviceProbe",
    "device_probe",

    # Budget Planner
    "plan_video_quality",
    "BudgetPlanner",
    "BudgetConstraints",
    "budget_planner",

    # Tier Router
    "route_generation_task",
    "TierRouter",
    "RoutingDecision",
    "tier_router",

    # Workload Analyzer
    "analyze_generation_task",
    "WorkloadAnalyzer",
    "TaskAnalysis",
    "workload_analyzer",

    # Day 2: Caching System
    "get_cache_manager",
    "CacheManager",
    "CacheEntry",

    # Day 2: RL Policy
    "get_rl_policy",
    "RLPolicy",
    "Action",
    "State",
    "Experience",

    # Day 2: Compression Engine
    "get_compression_engine",
    "CompressionEngine",
    "CompressionPreset",

    # Day 2: Quality Assessor
    "get_quality_assessor",
    "QualityAssessor",
    "QualityMetrics",

    # Day 2: Adaptive Pipeline
    "get_adaptive_pipeline",
    "process_adaptive_request",
    "AdaptivePipeline",
    "PipelineResult",

    # Day 3: NAS Storage
    "get_nas_storage",
    "NASStorageManager",

    # Day 3: GPU Queue
    "get_gpu_queue",
    "GPUQueueManager",
    "GPUJob",
    "JobStatus",
    "JobPriority",

    # Day 3: Mixed Precision
    "get_mixed_precision",
    "MixedPrecisionManager",
    "PrecisionConfig",
    "PrecisionMode",
    "DeviceType",

    # Day 3: Lip Sync
    "get_lip_sync",
    "LipSyncManager",
    "LipSyncConfig",
    "LipSyncResult",

    # Day 4: Load Testing & Scaling
    "get_load_tester",
    "get_degradation_manager",
    "LoadTester",
    "GracefulDegradationManager",
    "LoadTestResult",
    "SimulatedUser",

    # Day 4: Analytics & Reporting
    "get_analytics",
    "AnalyticsManager",
    "RequestMetrics",
    "SystemMetrics",
    "CostReport",
    "LatencyReport"
]