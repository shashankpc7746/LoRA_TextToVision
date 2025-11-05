"""
Comprehensive Automation Testing Suite for Task 9
Runs all tests and generates detailed error report
"""
import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import traceback

class TestRunner:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'test_runs': [],
            'summary': {
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'errors': 0,
                'skipped': 0
            },
            'errors': [],
            'warnings': []
        }
    
    def run_test(self, test_path, test_name, category):
        """Run a single test file and capture results"""
        print(f"\n{'='*70}")
        print(f"🧪 Running: {test_name}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        try:
            # Run pytest without JSON report (not installed)
            cmd = [
                sys.executable, '-m', 'pytest',
                test_path,
                '-v',
                '--tb=short',
                '-x'  # Stop on first failure
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            duration = time.time() - start_time
            
            # Parse output
            stdout = result.stdout
            stderr = result.stderr
            return_code = result.returncode
            
            # Determine status
            if return_code == 0:
                status = "PASSED"
            elif return_code == 1:
                status = "FAILED"
            elif return_code == 2:
                status = "ERROR"
            else:
                status = "UNKNOWN"
            
            # Extract test counts from output
            passed = stdout.count(" PASSED")
            failed = stdout.count(" FAILED")
            errors = stdout.count(" ERROR")
            skipped = stdout.count(" SKIPPED")
            
            test_result = {
                'name': test_name,
                'category': category,
                'path': test_path,
                'status': status,
                'duration': round(duration, 2),
                'passed': passed,
                'failed': failed,
                'errors': errors,
                'skipped': skipped,
                'stdout': stdout,
                'stderr': stderr,
                'return_code': return_code
            }
            
            self.results['test_runs'].append(test_result)
            
            # Update summary
            self.results['summary']['total_tests'] += (passed + failed + errors + skipped)
            self.results['summary']['passed'] += passed
            self.results['summary']['failed'] += failed
            self.results['summary']['errors'] += errors
            self.results['summary']['skipped'] += skipped
            
            # Extract errors
            if status in ["FAILED", "ERROR"]:
                self.extract_errors(test_result)
            
            # Print result
            status_emoji = "✅" if status == "PASSED" else "❌"
            print(f"\n{status_emoji} {status}: {test_name}")
            print(f"   Duration: {duration:.2f}s")
            print(f"   Tests: {passed} passed, {failed} failed, {errors} errors")
            
            return test_result
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            test_result = {
                'name': test_name,
                'category': category,
                'path': test_path,
                'status': 'TIMEOUT',
                'duration': round(duration, 2),
                'error': 'Test timed out after 1 hour'
            }
            self.results['test_runs'].append(test_result)
            self.results['errors'].append({
                'test': test_name,
                'error': 'TIMEOUT',
                'message': 'Test execution exceeded 1 hour limit'
            })
            print(f"\n⏱️ TIMEOUT: {test_name} (after {duration:.2f}s)")
            return test_result
            
        except Exception as e:
            duration = time.time() - start_time
            test_result = {
                'name': test_name,
                'category': category,
                'path': test_path,
                'status': 'EXCEPTION',
                'duration': round(duration, 2),
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            self.results['test_runs'].append(test_result)
            self.results['errors'].append({
                'test': test_name,
                'error': type(e).__name__,
                'message': str(e)
            })
            print(f"\n💥 EXCEPTION: {test_name}")
            print(f"   Error: {e}")
            return test_result
    
    def extract_errors(self, test_result):
        """Extract specific error messages from test output"""
        output = test_result.get('stdout', '') + test_result.get('stderr', '')
        
        # Look for common error patterns
        error_patterns = [
            'AssertionError',
            'FileNotFoundError',
            'ModuleNotFoundError',
            'ImportError',
            'RuntimeError',
            'ValueError',
            'TypeError',
            'AttributeError'
        ]
        
        for pattern in error_patterns:
            if pattern in output:
                # Extract context around error
                lines = output.split('\n')
                for i, line in enumerate(lines):
                    if pattern in line:
                        context = '\n'.join(lines[max(0, i-2):min(len(lines), i+3)])
                        self.results['errors'].append({
                            'test': test_result['name'],
                            'error': pattern,
                            'context': context
                        })
    
    def generate_report(self, output_file):
        """Generate detailed markdown report"""
        report = []
        
        # Header
        report.append("# Automation Testing Report - Task 9")
        report.append(f"\n**Date:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
        report.append(f"**Project:** LoRA_TextToVision")
        report.append(f"**Branch:** task_quality_leap")
        report.append(f"**Test Suite:** Task 9 - Indigenous Image Adapter")
        report.append("\n---\n")
        
        # Executive Summary
        report.append("## 📊 Executive Summary\n")
        summary = self.results['summary']
        total_runs = len(self.results['test_runs'])
        passed_runs = sum(1 for r in self.results['test_runs'] if r.get('status') == 'PASSED')
        failed_runs = sum(1 for r in self.results['test_runs'] if r.get('status') in ['FAILED', 'ERROR'])
        
        report.append(f"**Test Files Executed:** {total_runs}")
        report.append(f"**Test Files Passed:** {passed_runs}")
        report.append(f"**Test Files Failed:** {failed_runs}")
        report.append(f"")
        report.append(f"**Total Test Cases:** {summary['total_tests']}")
        report.append(f"**Passed:** {summary['passed']} ✅")
        report.append(f"**Failed:** {summary['failed']} ❌")
        report.append(f"**Errors:** {summary['errors']} 💥")
        report.append(f"**Skipped:** {summary['skipped']} ⏭️")
        report.append(f"")
        
        if total_runs > 0:
            success_rate = (passed_runs / total_runs) * 100
            report.append(f"**Success Rate:** {success_rate:.1f}%")
        
        report.append("\n---\n")
        
        # Test Results by Category
        categories = {}
        for test in self.results['test_runs']:
            cat = test.get('category', 'Other')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(test)
        
        report.append("## 🧪 Test Results by Category\n")
        
        for category, tests in sorted(categories.items()):
            report.append(f"### {category}\n")
            report.append("| Test Name | Status | Duration | Tests | Result |")
            report.append("|-----------|--------|----------|-------|--------|")
            
            for test in tests:
                status = test.get('status', 'UNKNOWN')
                emoji = {
                    'PASSED': '✅',
                    'FAILED': '❌',
                    'ERROR': '💥',
                    'TIMEOUT': '⏱️',
                    'EXCEPTION': '🔥',
                    'SKIPPED': '⏭️'
                }.get(status, '❓')
                
                name = test.get('name', 'Unknown')
                duration = test.get('duration', 0)
                passed = test.get('passed', 0)
                failed = test.get('failed', 0)
                errors = test.get('errors', 0)
                
                test_summary = f"{passed}✅ {failed}❌ {errors}💥"
                
                report.append(f"| {name} | {emoji} {status} | {duration:.2f}s | {test_summary} | - |")
            
            report.append("")
        
        report.append("---\n")
        
        # Detailed Error Analysis
        if self.results['errors']:
            report.append("## ❌ Detailed Error Analysis\n")
            report.append(f"**Total Errors Found:** {len(self.results['errors'])}\n")
            
            for idx, error in enumerate(self.results['errors'], 1):
                report.append(f"### Error #{idx}: {error.get('test', 'Unknown Test')}\n")
                report.append(f"**Error Type:** `{error.get('error', 'Unknown')}`\n")
                
                if 'message' in error:
                    report.append(f"**Message:** {error['message']}\n")
                
                if 'context' in error:
                    report.append("**Context:**")
                    report.append("```")
                    report.append(error['context'])
                    report.append("```\n")
        else:
            report.append("## ✅ No Errors Found\n")
            report.append("All tests passed successfully!\n")
        
        report.append("---\n")
        
        # Detailed Test Outputs
        report.append("## 📝 Detailed Test Outputs\n")
        
        for test in self.results['test_runs']:
            report.append(f"### {test.get('name', 'Unknown')}\n")
            report.append(f"**Category:** {test.get('category', 'Unknown')}")
            report.append(f"**Status:** {test.get('status', 'UNKNOWN')}")
            report.append(f"**Duration:** {test.get('duration', 0):.2f}s")
            report.append(f"**Path:** `{test.get('path', 'Unknown')}`\n")
            
            if test.get('stdout'):
                report.append("**Standard Output:**")
                report.append("```")
                # Truncate very long outputs
                stdout = test['stdout']
                if len(stdout) > 5000:
                    report.append(stdout[:2500])
                    report.append("\n... (output truncated) ...\n")
                    report.append(stdout[-2500:])
                else:
                    report.append(stdout)
                report.append("```\n")
            
            if test.get('stderr'):
                report.append("**Standard Error:**")
                report.append("```")
                stderr = test['stderr']
                if len(stderr) > 2000:
                    report.append(stderr[:1000])
                    report.append("\n... (output truncated) ...\n")
                    report.append(stderr[-1000:])
                else:
                    report.append(stderr)
                report.append("```\n")
            
            report.append("---\n")
        
        # Recommendations
        report.append("## 💡 Recommendations\n")
        
        if failed_runs == 0:
            report.append("✅ All tests passed! System is working as expected.\n")
        else:
            report.append("### Action Items:\n")
            
            for test in self.results['test_runs']:
                if test.get('status') in ['FAILED', 'ERROR']:
                    report.append(f"- [ ] Fix errors in `{test.get('name')}`")
            
            report.append("\n### Common Issues:\n")
            report.append("1. **Import Errors**: Check if all dependencies are installed")
            report.append("2. **File Not Found**: Verify file paths and working directory")
            report.append("3. **GPU Memory**: Ensure sufficient VRAM for tests")
            report.append("4. **Model Files**: Confirm SDXL models are downloaded")
        
        report.append("\n---\n")
        
        # System Information
        report.append("## 🖥️ System Information\n")
        report.append(f"**Python Version:** {sys.version.split()[0]}")
        
        try:
            import torch
            report.append(f"**PyTorch Version:** {torch.__version__}")
            if torch.cuda.is_available():
                report.append(f"**CUDA Available:** Yes")
                report.append(f"**GPU:** {torch.cuda.get_device_name(0)}")
                report.append(f"**GPU Memory:** {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                report.append(f"**CUDA Available:** No")
        except:
            pass
        
        report.append("\n---\n")
        report.append(f"\n*Report generated by automation_testing.py on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*")
        
        # Write report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"\n✅ Report saved to: {output_file}")

def main():
    print("="*70)
    print("🤖 TASK 9 - AUTOMATION TESTING SUITE")
    print("="*70)
    print(f"\nStarted at: {datetime.now().strftime('%I:%M %p')}")
    print(f"Working directory: {Path.cwd()}")
    
    runner = TestRunner()
    
    # Define test suite
    tests = [
        # Component Tests
        {
            'path': 'tests/task9/components/upscaler/test_upscaler_component.py',
            'name': 'Upscaler Component Test',
            'category': '🔧 Component Tests'
        },
        {
            'path': 'tests/task9/components/temporal/test_temporal_simple.py',
            'name': 'Temporal Consistency Test',
            'category': '🔧 Component Tests'
        },
        {
            'path': 'tests/task9/components/motion/test_motion_controller.py',
            'name': 'Motion Controller Test',
            'category': '🔧 Component Tests'
        },
        
        # Integration Tests
        {
            'path': 'tests/task9/integration/test_task9_simple.py',
            'name': 'Simple Integration Test',
            'category': '🔗 Integration Tests'
        },
        {
            'path': 'tests/task9/integration/test_task9_integration.py',
            'name': 'Full Integration Test',
            'category': '🔗 Integration Tests'
        },
        
        # Quality Tests
        {
            'path': 'tests/task9/quality/test_comprehensive.py',
            'name': 'Comprehensive Quality Test',
            'category': '⭐ Quality Tests'
        },
        {
            'path': 'tests/task9/quality/test_quality_card.py',
            'name': 'Quality Card (VMAF + Lip-sync)',
            'category': '⭐ Quality Tests'
        },
        
        # Adapter Tests
        {
            'path': 'adapters/gurukul_lora/test_imports.py',
            'name': 'Import Validation Test',
            'category': '🎨 Adapter Tests'
        },
        {
            'path': 'adapters/gurukul_lora/test_adapter.py',
            'name': 'Adapter Functionality Test',
            'category': '🎨 Adapter Tests'
        }
    ]
    
    print(f"\n📋 Total test files to run: {len(tests)}\n")
    
    # Run all tests
    for test in tests:
        test_path = Path(test['path'])
        if test_path.exists():
            runner.run_test(str(test_path), test['name'], test['category'])
        else:
            print(f"\n⚠️  SKIPPED: {test['name']} (file not found)")
            runner.results['test_runs'].append({
                'name': test['name'],
                'category': test['category'],
                'path': test['path'],
                'status': 'SKIPPED',
                'reason': 'File not found'
            })
    
    # Generate report
    print(f"\n{'='*70}")
    print("📊 GENERATING REPORT")
    print("="*70)
    
    output_file = Path('AUTOMATION_TESTING_ERRORS_AND_BUGS.md')
    runner.generate_report(output_file)
    
    # Print summary
    print(f"\n{'='*70}")
    print("📈 FINAL SUMMARY")
    print("="*70)
    summary = runner.results['summary']
    print(f"Total Tests: {summary['total_tests']}")
    print(f"✅ Passed: {summary['passed']}")
    print(f"❌ Failed: {summary['failed']}")
    print(f"💥 Errors: {summary['errors']}")
    print(f"⏭️  Skipped: {summary['skipped']}")
    
    passed_runs = sum(1 for r in runner.results['test_runs'] if r.get('status') == 'PASSED')
    total_runs = len([r for r in runner.results['test_runs'] if r.get('status') != 'SKIPPED'])
    
    if total_runs > 0:
        success_rate = (passed_runs / total_runs) * 100
        print(f"\n🎯 Success Rate: {success_rate:.1f}%")
    
    print(f"\n✅ Report saved to: {output_file.absolute()}")
    print(f"\nCompleted at: {datetime.now().strftime('%I:%M %p')}")
    print("="*70)

if __name__ == "__main__":
    main()
