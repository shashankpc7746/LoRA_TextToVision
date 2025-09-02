#!/usr/bin/env python3
"""
Adaptive Engine Package - Task-4 Day-1
Intelligent video generation system with device-aware processing
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

__version__ = "1.0.0"
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
    "workload_analyzer"
]