#!/usr/bin/env python3
"""
Metrics Visualizer for Task-6 Production Hardening
Creates graphs and charts from stress test telemetry data
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from datetime import datetime


class MetricsVisualizer:
    """Visualize stress test metrics and telemetry data"""

    def __init__(self, results_file: str = "gradual_stress_test_results.json"):
        self.results_file = results_file
        self.data = None
        self.output_dir = Path("metrics_charts")
        self.output_dir.mkdir(exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def load_data(self):
        """Load stress test results"""
        try:
            with open(self.results_file, 'r') as f:
                self.data = json.load(f)
            print(f"✅ Loaded data from {self.results_file}")
            return True
        except FileNotFoundError:
            print(f"❌ Results file not found: {self.results_file}")
            return False
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in results file: {e}")
            return False

    def create_success_rate_chart(self):
        """Create success rate comparison chart"""
        if not self.data:
            return

        levels = []
        success_rates = []

        for result in self.data.get("results", []):
            level_name = result["level"]
            success_rate = result["summary"].get("success_rate", 0)

            levels.append(level_name)
            success_rates.append(success_rate)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(levels, success_rates, color=['#2ecc71', '#f39c12', '#e74c3c'])

        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    '.1f', ha='center', va='bottom', fontweight='bold')

        plt.title('Stress Test Success Rate by Load Level', fontsize=14, fontweight='bold')
        plt.xlabel('Test Level')
        plt.ylabel('Success Rate (%)')
        plt.ylim(0, 105)
        plt.grid(axis='y', alpha=0.3)

        # Add threshold line
        plt.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='95% Threshold')
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'success_rate_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Created success rate comparison chart")

    def create_response_time_chart(self):
        """Create response time analysis chart"""
        if not self.data:
            return

        levels = []
        avg_times = []
        p95_times = []

        for result in self.data.get("results", []):
            level_name = result["level"]
            summary = result["summary"]

            levels.append(level_name)
            avg_times.append(summary.get("average_response_time", 0))
            p95_times.append(summary.get("p95_response_time", 0))

        x = np.arange(len(levels))
        width = 0.35

        plt.figure(figsize=(12, 7))
        plt.bar(x - width/2, avg_times, width, label='Average Response Time', color='#3498db', alpha=0.8)
        plt.bar(x + width/2, p95_times, width, label='P95 Response Time', color='#e74c3c', alpha=0.8)

        plt.xlabel('Test Level')
        plt.ylabel('Response Time (seconds)')
        plt.title('Response Time Analysis by Load Level', fontsize=14, fontweight='bold')
        plt.xticks(x, levels)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        # Add threshold line
        plt.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='10s Threshold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'response_time_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Created response time analysis chart")

    def create_throughput_chart(self):
        """Create throughput analysis chart"""
        if not self.data:
            return

        levels = []
        throughputs = []

        for result in self.data.get("results", []):
            level_name = result["level"]
            throughput = result["summary"].get("throughput_rps", 0)

            levels.append(level_name)
            throughputs.append(throughput)

        plt.figure(figsize=(10, 6))
        plt.plot(levels, throughputs, marker='o', linewidth=3, markersize=8, color='#9b59b6')

        # Add value labels
        for i, (level, throughput) in enumerate(zip(levels, throughputs)):
            plt.text(i, throughput + 0.1, '.2f', ha='center', va='bottom', fontweight='bold')

        plt.title('System Throughput by Load Level', fontsize=14, fontweight='bold')
        plt.xlabel('Test Level')
        plt.ylabel('Requests per Second (RPS)')
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'throughput_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Created throughput analysis chart")

    def create_status_distribution_chart(self):
        """Create status code distribution chart"""
        if not self.data:
            return

        # Collect all status codes across levels
        all_statuses = {}
        level_names = []

        for result in self.data.get("results", []):
            level_name = result["level"]
            level_names.append(level_name)

            status_dist = result["summary"].get("status_distribution", {})
            for status, count in status_dist.items():
                if status not in all_statuses:
                    all_statuses[status] = []
                all_statuses[status].append(count)

        # Fill missing values with 0
        for status in all_statuses:
            while len(all_statuses[status]) < len(level_names):
                all_statuses[status].append(0)

        # Create stacked bar chart
        plt.figure(figsize=(12, 7))

        bottom = np.zeros(len(level_names))
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#3498db']

        for i, (status, counts) in enumerate(all_statuses.items()):
            color = colors[i % len(colors)]
            plt.bar(level_names, counts, bottom=bottom, label=f'Status {status}', color=color, alpha=0.8)
            bottom += np.array(counts)

        plt.title('HTTP Status Code Distribution by Load Level', fontsize=14, fontweight='bold')
        plt.xlabel('Test Level')
        plt.ylabel('Number of Requests')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'status_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Created status distribution chart")

    def create_performance_summary_chart(self):
        """Create overall performance summary"""
        if not self.data:
            return

        # Extract key metrics
        levels = []
        success_rates = []
        response_times = []
        throughputs = []

        for result in self.data.get("results", []):
            levels.append(result["level"])
            summary = result["summary"]
            success_rates.append(summary.get("success_rate", 0))
            response_times.append(summary.get("average_response_time", 0))
            throughputs.append(summary.get("throughput_rps", 0))

        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Success Rate
        ax1.bar(levels, success_rates, color='#2ecc71', alpha=0.8)
        ax1.set_title('Success Rate (%)')
        ax1.set_ylim(0, 105)
        ax1.axhline(y=95, color='red', linestyle='--', alpha=0.7)

        # Response Time
        ax2.bar(levels, response_times, color='#3498db', alpha=0.8)
        ax2.set_title('Avg Response Time (s)')
        ax2.axhline(y=10, color='red', linestyle='--', alpha=0.7)

        # Throughput
        ax3.plot(levels, throughputs, marker='o', linewidth=3, color='#9b59b6')
        ax3.set_title('Throughput (RPS)')
        ax3.grid(alpha=0.3)

        # Performance Score (custom metric)
        performance_scores = []
        for i in range(len(levels)):
            success_score = min(100, success_rates[i]) / 100
            time_score = max(0, 1 - (response_times[i] / 10))  # Better if faster than 10s
            score = (success_score * 0.7 + time_score * 0.3) * 100
            performance_scores.append(score)

        ax4.bar(levels, performance_scores, color='#e67e22', alpha=0.8)
        ax4.set_title('Performance Score')
        ax4.set_ylim(0, 105)

        plt.suptitle('Task-6 Production Hardening - Performance Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Created performance summary dashboard")

    def generate_report(self):
        """Generate comprehensive metrics report"""
        if not self.data:
            return

        report = f"""
# Task-6 Production Hardening - Metrics Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Test Overview
- **Test Type:** Gradual Stress Test (10 → 25 → 50 users)
- **Overall Success:** {'✅ PASSED' if self.data.get('overall_success', False) else '❌ FAILED'}
- **Levels Tested:** {len(self.data.get('results', []))}

## Key Findings

"""

        for result in self.data.get("results", []):
            level = result["level"]
            summary = result["summary"]
            success_rate = summary.get("success_rate", 0)
            avg_time = summary.get("average_response_time", 0)
            throughput = summary.get("throughput_rps", 0)
            passed = summary.get("test_passed", False)

            report += f"""
### {level}
- **Success Rate:** {success_rate:.1f}%
- **Avg Response Time:** {avg_time:.2f}s
- **Throughput:** {throughput:.2f} RPS
- **Status:** {'✅ PASSED' if passed else '❌ FAILED'}
"""

        report += """
## Charts Generated
- `success_rate_comparison.png` - Success rate by load level
- `response_time_analysis.png` - Response time analysis
- `throughput_analysis.png` - System throughput trends
- `status_distribution.png` - HTTP status code distribution
- `performance_summary.png` - Overall performance dashboard

## Recommendations

"""

        # Add recommendations based on results
        if self.data.get('overall_success', False):
            report += "- ✅ System is production-ready with excellent performance\n"
            report += "- ✅ Gradual scaling prevents GPU memory issues\n"
            report += "- ✅ All load levels maintained quality thresholds\n"
        else:
            report += "- ⚠️ Some load levels failed - investigate performance bottlenecks\n"
            report += "- ⚠️ Consider GPU memory optimization or scaling limits\n"

        report += "\n---\n*Task-6 Production Hardening - Metrics Analysis*"

        # Save report
        with open(self.output_dir / 'metrics_report.md', 'w') as f:
            f.write(report)

        print("✅ Generated comprehensive metrics report")

    def run_all_visualizations(self):
        """Run all visualization functions"""
        print("🎨 Generating Task-6 Production Metrics Visualizations...")
        print("=" * 60)

        if not self.load_data():
            return False

        try:
            self.create_success_rate_chart()
            self.create_response_time_chart()
            self.create_throughput_chart()
            self.create_status_distribution_chart()
            self.create_performance_summary_chart()
            self.generate_report()

            print("\n" + "=" * 60)
            print("✅ All visualizations completed successfully!")
            print(f"📊 Charts saved to: {self.output_dir}/")
            print("📋 Report saved to: metrics_report.md")
            return True

        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            return False


def main():
    """Main visualization execution"""
    visualizer = MetricsVisualizer()

    if visualizer.run_all_visualizations():
        print("\n🎯 Task-6 Metrics Analysis Complete!")
        print("📈 Use these charts for production monitoring and optimization")
        return 0
    else:
        print("\n❌ Metrics visualization failed")
        return 1


if __name__ == "__main__":
    exit(main())