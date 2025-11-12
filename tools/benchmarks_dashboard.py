"""
Benchmarks Dashboard Generator
Creates HTML dashboard for visualizing performance metrics over time
"""

import os
import json
import glob
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict


class BenchmarksDashboard:
    """Generate performance benchmarks dashboard"""

    def __init__(self, metrics_dir: str = "metrics"):
        self.metrics_dir = metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)

    def collect_metrics(self, days: int = 30) -> List[Dict]:
        """Collect metrics from the last N days"""
        metrics = []
        cutoff_date = datetime.now() - timedelta(days=days)

        # Find all metric files
        for metrics_file in glob.glob(os.path.join(self.metrics_dir, "*.json")):
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    
                    # Extract timestamp from filename or data
                    if 'timestamp' in data:
                        timestamp = datetime.fromisoformat(data['timestamp'])
                    else:
                        # Try to parse from filename (e.g., "2025-11-12.json")
                        filename = os.path.basename(metrics_file).replace('.json', '')
                        try:
                            timestamp = datetime.strptime(filename, "%Y-%m-%d")
                        except ValueError:
                            timestamp = datetime.now()
                    
                    if timestamp >= cutoff_date:
                        metrics.append({
                            'timestamp': timestamp,
                            'data': data
                        })
            except Exception as e:
                print(f"Warning: Could not load {metrics_file}: {e}")

        return sorted(metrics, key=lambda x: x['timestamp'])

    def compute_statistics(self, metrics: List[Dict]) -> Dict:
        """Compute aggregate statistics"""
        if not metrics:
            return {
                'total_generations': 0,
                'success_rate': 0.0,
                'avg_quality': 0.0,
                'avg_latency': 0.0,
                'avg_cost': 0.0,
                'p50_latency': 0.0,
                'p95_latency': 0.0,
                'p99_latency': 0.0
            }

        # Extract all operations
        all_operations = []
        for m in metrics:
            if 'operations' in m['data']:
                all_operations.extend(m['data']['operations'])

        if not all_operations:
            return {
                'total_generations': 0,
                'success_rate': 0.0,
                'avg_quality': 0.0,
                'avg_latency': 0.0,
                'avg_cost': 0.0,
                'p50_latency': 0.0,
                'p95_latency': 0.0,
                'p99_latency': 0.0
            }

        # Compute statistics
        successful = [op for op in all_operations if op.get('success', True)]
        latencies = sorted([op.get('duration', 0) for op in all_operations])
        qualities = [op.get('quality', 0) for op in successful if 'quality' in op]
        costs = [op.get('cost', 0) for op in successful if 'cost' in op]

        def percentile(data, p):
            if not data:
                return 0.0
            k = (len(data) - 1) * p / 100.0
            f = int(k)
            c = int(k) + 1
            if c >= len(data):
                return data[-1]
            return data[f] + (k - f) * (data[c] - data[f])

        return {
            'total_generations': len(all_operations),
            'success_rate': len(successful) / len(all_operations) * 100 if all_operations else 0,
            'avg_quality': sum(qualities) / len(qualities) if qualities else 0,
            'avg_latency': sum(latencies) / len(latencies) if latencies else 0,
            'avg_cost': sum(costs) / len(costs) if costs else 0,
            'p50_latency': percentile(latencies, 50),
            'p95_latency': percentile(latencies, 95),
            'p99_latency': percentile(latencies, 99)
        }

    def generate_html_dashboard(self, output_file: str = "benchmarks_dashboard.html"):
        """Generate HTML dashboard"""
        print(f"📊 Generating benchmarks dashboard...")

        # Collect metrics
        metrics = self.collect_metrics(days=30)
        stats = self.compute_statistics(metrics)

        # Generate HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LoRA_TextToVision - Performance Benchmarks</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #667eea;
            margin-bottom: 10px;
        }}
        .header p {{
            color: #666;
            font-size: 14px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }}
        .metric-label {{
            font-size: 14px;
            color: #666;
            margin-bottom: 8px;
            font-weight: 500;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-trend {{
            font-size: 12px;
            color: #28a745;
            margin-top: 5px;
        }}
        .metric-trend.down {{
            color: #dc3545;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .chart-card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .chart-title {{
            font-size: 18px;
            font-weight: 600;
            color: #333;
            margin-bottom: 20px;
        }}
        .status-indicator {{
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        .status-good {{
            background: #28a745;
        }}
        .status-warning {{
            background: #ffc107;
        }}
        .status-bad {{
            background: #dc3545;
        }}
        .footer {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 LoRA_TextToVision Performance Dashboard</h1>
            <p>Real-time monitoring of video generation performance metrics</p>
            <p style="margin-top: 10px;"><strong>Last Updated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">
                    <span class="status-indicator status-good"></span>
                    Total Generations
                </div>
                <div class="metric-value">{stats['total_generations']:,}</div>
                <div class="metric-trend">Last 30 days</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">
                    <span class="status-indicator {'status-good' if stats['success_rate'] >= 95 else 'status-warning' if stats['success_rate'] >= 80 else 'status-bad'}"></span>
                    Success Rate
                </div>
                <div class="metric-value">{stats['success_rate']:.1f}%</div>
                <div class="metric-trend">Target: ≥95%</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">
                    <span class="status-indicator {'status-good' if stats['avg_quality'] >= 0.8 else 'status-warning' if stats['avg_quality'] >= 0.7 else 'status-bad'}"></span>
                    Average Quality (VMAF)
                </div>
                <div class="metric-value">{stats['avg_quality']:.2f}</div>
                <div class="metric-trend">Target: ≥0.80</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">
                    <span class="status-indicator {'status-good' if stats['avg_latency'] <= 180 else 'status-warning' if stats['avg_latency'] <= 300 else 'status-bad'}"></span>
                    Avg Latency
                </div>
                <div class="metric-value">{stats['avg_latency']:.0f}s</div>
                <div class="metric-trend">Target: ≤180s</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">
                    <span class="status-indicator status-good"></span>
                    P95 Latency
                </div>
                <div class="metric-value">{stats['p95_latency']:.0f}s</div>
                <div class="metric-trend">95th percentile</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">
                    <span class="status-indicator {'status-good' if stats['avg_cost'] <= 0.10 else 'status-warning' if stats['avg_cost'] <= 0.20 else 'status-bad'}"></span>
                    Avg Cost per Video
                </div>
                <div class="metric-value">${stats['avg_cost']:.2f}</div>
                <div class="metric-trend">Target: ≤$0.10</div>
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-card">
                <div class="chart-title">📊 Quality Over Time</div>
                <canvas id="qualityChart"></canvas>
            </div>

            <div class="chart-card">
                <div class="chart-title">⚡ Latency Distribution</div>
                <canvas id="latencyChart"></canvas>
            </div>

            <div class="chart-card">
                <div class="chart-title">💰 Cost Trend</div>
                <canvas id="costChart"></canvas>
            </div>

            <div class="chart-card">
                <div class="chart-title">✅ Success Rate Trend</div>
                <canvas id="successChart"></canvas>
            </div>
        </div>

        <div class="footer">
            <p>🎬 LoRA_TextToVision | Enterprise Video Generation Platform</p>
            <p style="margin-top: 10px;">Metrics collected from {len(metrics)} data points over last 30 days</p>
        </div>
    </div>

    <script>
        // Prepare data from metrics
        const metricsData = {json.dumps([{
            'date': m['timestamp'].strftime('%Y-%m-%d'),
            'operations': m['data'].get('operations', [])
        } for m in metrics])};

        // Process data for charts
        const dates = metricsData.map(m => m.date);
        const qualityData = metricsData.map(m => {{
            const ops = m.operations.filter(op => op.quality !== undefined);
            return ops.length > 0 ? ops.reduce((sum, op) => sum + op.quality, 0) / ops.length : 0;
        }});
        const latencyData = metricsData.map(m => {{
            const ops = m.operations;
            return ops.length > 0 ? ops.reduce((sum, op) => sum + (op.duration || 0), 0) / ops.length : 0;
        }});
        const costData = metricsData.map(m => {{
            const ops = m.operations.filter(op => op.cost !== undefined);
            return ops.length > 0 ? ops.reduce((sum, op) => sum + op.cost, 0) / ops.length : 0;
        }});
        const successData = metricsData.map(m => {{
            const ops = m.operations;
            const successful = ops.filter(op => op.success !== false);
            return ops.length > 0 ? (successful.length / ops.length) * 100 : 0;
        }});

        // Chart configuration
        const chartConfig = {{
            type: 'line',
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }};

        // Quality Chart
        new Chart(document.getElementById('qualityChart'), {{
            ...chartConfig,
            data: {{
                labels: dates,
                datasets: [{{
                    label: 'Quality Score',
                    data: qualityData,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: true
                }}]
            }},
            options: {{
                ...chartConfig.options,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1.0,
                        ticks: {{
                            callback: function(value) {{ return value.toFixed(2); }}
                        }}
                    }}
                }}
            }}
        }});

        // Latency Chart
        new Chart(document.getElementById('latencyChart'), {{
            ...chartConfig,
            data: {{
                labels: dates,
                datasets: [{{
                    label: 'Avg Latency (s)',
                    data: latencyData,
                    borderColor: '#f093fb',
                    backgroundColor: 'rgba(240, 147, 251, 0.1)',
                    tension: 0.4,
                    fill: true
                }}]
            }}
        }});

        // Cost Chart
        new Chart(document.getElementById('costChart'), {{
            ...chartConfig,
            data: {{
                labels: dates,
                datasets: [{{
                    label: 'Avg Cost ($)',
                    data: costData,
                    borderColor: '#4facfe',
                    backgroundColor: 'rgba(79, 172, 254, 0.1)',
                    tension: 0.4,
                    fill: true
                }}]
            }},
            options: {{
                ...chartConfig.options,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        ticks: {{
                            callback: function(value) {{ return '$' + value.toFixed(2); }}
                        }}
                    }}
                }}
            }}
        }});

        // Success Rate Chart
        new Chart(document.getElementById('successChart'), {{
            ...chartConfig,
            data: {{
                labels: dates,
                datasets: [{{
                    label: 'Success Rate (%)',
                    data: successData,
                    borderColor: '#43e97b',
                    backgroundColor: 'rgba(67, 233, 123, 0.1)',
                    tension: 0.4,
                    fill: true
                }}]
            }},
            options: {{
                ...chartConfig.options,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        ticks: {{
                            callback: function(value) {{ return value + '%'; }}
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"   ✅ Dashboard generated: {output_file}")
        print(f"   📊 Total generations: {stats['total_generations']}")
        print(f"   ✅ Success rate: {stats['success_rate']:.1f}%")
        print(f"   📈 Avg quality: {stats['avg_quality']:.2f}")
        print(f"   ⚡ Avg latency: {stats['avg_latency']:.1f}s")
        print(f"   💰 Avg cost: ${stats['avg_cost']:.2f}")

        return output_file


def main():
    """Generate dashboard from command line"""
    import sys

    dashboard = BenchmarksDashboard()

    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    else:
        output_file = "benchmarks_dashboard.html"

    dashboard.generate_html_dashboard(output_file)
    print(f"\n🌐 Open in browser: file://{os.path.abspath(output_file)}")


if __name__ == "__main__":
    main()
