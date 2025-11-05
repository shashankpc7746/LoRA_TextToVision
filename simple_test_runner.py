"""
Simple Test Runner for Task 9 - Generates Error Report
Runs tests individually and captures all output
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import time

def run_single_test(test_path, test_name):
    """Run a single test file"""
    print(f"\n{'='*70}")
    print(f"🧪 {test_name}")
    print(f"{'='*70}")
    
    if not Path(test_path).exists():
        return {
            'name': test_name,
            'status': 'SKIPPED',
            'reason': 'File not found',
            'path': test_path
        }
    
    start = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', test_path, '-v', '--tb=short'],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per test
        )
        
        duration = time.time() - start
        
        # Parse output
        output = result.stdout + '\n' + result.stderr
        
        # Count results
        passed = output.count(' PASSED')
        failed = output.count(' FAILED')
        errors = output.count(' ERROR')
        skipped = output.count(' SKIPPED')
        
        # Determine status
        if result.returncode == 0:
            status = 'PASSED' if passed > 0 else 'NO_TESTS'
        elif 'FAILED' in output or 'ERROR' in output:
            status = 'FAILED'
        elif 'ModuleNotFoundError' in output or 'ImportError' in output:
            status = 'IMPORT_ERROR'
        else:
            status = 'ERROR'
        
        result_data = {
            'name': test_name,
            'status': status,
            'duration': round(duration, 2),
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'skipped': skipped,
            'output': output,
            'return_code': result.returncode,
            'path': test_path
        }
        
        # Print status
        emoji = '✅' if status == 'PASSED' else '❌'
        print(f"{emoji} {status} - {passed}✅ {failed}❌ {errors}💥 in {duration:.1f}s")
        
        return result_data
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start
        print(f"⏱️  TIMEOUT after {duration:.1f}s")
        return {
            'name': test_name,
            'status': 'TIMEOUT',
            'duration': round(duration, 2),
            'error': f'Test exceeded 10 minute timeout',
            'path': test_path
        }
    except Exception as e:
        print(f"💥 EXCEPTION: {e}")
        return {
            'name': test_name,
            'status': 'EXCEPTION',
            'error': str(e),
            'path': test_path
        }

def generate_report(results, output_file):
    """Generate markdown report"""
    lines = []
    
    lines.append("# Automation Testing Report - Task 9\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    lines.append(f"**Project:** LoRA_TextToVision")
    lines.append(f"**Branch:** task_quality_leap\n")
    lines.append("---\n")
    
    # Summary
    lines.append("## 📊 Executive Summary\n")
    total = len(results)
    passed = sum(1 for r in results if r['status'] == 'PASSED')
    failed = sum(1 for r in results if r['status'] in ['FAILED', 'ERROR', 'IMPORT_ERROR'])
    skipped = sum(1 for r in results if r['status'] == 'SKIPPED')
    timeout = sum(1 for r in results if r['status'] == 'TIMEOUT')
    
    lines.append(f"**Total Test Files:** {total}")
    lines.append(f"**Passed:** {passed} ✅")
    lines.append(f"**Failed:** {failed} ❌")
    lines.append(f"**Skipped:** {skipped} ⏭️")
    lines.append(f"**Timeout:** {timeout} ⏱️")
    
    if total > 0:
        success_rate = (passed / total) * 100
        lines.append(f"**Success Rate:** {success_rate:.1f}%\n")
    
    # Test case summary
    total_tests = sum(r.get('passed', 0) + r.get('failed', 0) + r.get('errors', 0) for r in results)
    total_passed = sum(r.get('passed', 0) for r in results)
    total_failed = sum(r.get('failed', 0) for r in results)
    total_errors = sum(r.get('errors', 0) for r in results)
    
    lines.append(f"**Total Test Cases:** {total_tests}")
    lines.append(f"**Test Cases Passed:** {total_passed} ✅")
    lines.append(f"**Test Cases Failed:** {total_failed} ❌")
    lines.append(f"**Test Cases Errored:** {total_errors} 💥\n")
    lines.append("---\n")
    
    # Results table
    lines.append("## 🧪 Test Results\n")
    lines.append("| Test Name | Status | Duration | Tests | Details |")
    lines.append("|-----------|--------|----------|-------|---------|")
    
    for r in results:
        name = r['name']
        status = r['status']
        emoji = {
            'PASSED': '✅',
            'FAILED': '❌',
            'ERROR': '💥',
            'IMPORT_ERROR': '📦',
            'TIMEOUT': '⏱️',
            'SKIPPED': '⏭️',
            'NO_TESTS': '❓'
        }.get(status, '❓')
        
        duration = r.get('duration', 0)
        passed = r.get('passed', 0)
        failed = r.get('failed', 0)
        errors = r.get('errors', 0)
        
        test_summary = f"{passed}✅ {failed}❌ {errors}💥" if total_tests > 0 else '-'
        details = r.get('reason', '') or r.get('error', '') or '-'
        
        lines.append(f"| {name} | {emoji} {status} | {duration:.1f}s | {test_summary} | {details[:50]} |")
    
    lines.append("\n---\n")
    
    # Errors and bugs section
    lines.append("## ❌ Errors and Bugs\n")
    
    errors_found = [r for r in results if r['status'] in ['FAILED', 'ERROR', 'IMPORT_ERROR', 'TIMEOUT']]
    
    if errors_found:
        lines.append(f"**Total Issues Found:** {len(errors_found)}\n")
        
        for idx, r in enumerate(errors_found, 1):
            lines.append(f"### {idx}. {r['name']} - {r['status']}\n")
            lines.append(f"**Status:** {r['status']}")
            lines.append(f"**File:** `{r['path']}`")
            
            if 'duration' in r:
                lines.append(f"**Duration:** {r['duration']:.1f}s")
            
            if 'error' in r:
                lines.append(f"\n**Error:**")
                lines.append(f"```")
                lines.append(r['error'])
                lines.append(f"```\n")
            
            if 'output' in r:
                output = r['output']
                # Extract error messages
                if 'ModuleNotFoundError' in output or 'ImportError' in output:
                    lines.append(f"\n**Import Error Details:**")
                    for line in output.split('\n'):
                        if 'ModuleNotFoundError' in line or 'ImportError' in line or 'No module named' in line:
                            lines.append(f"- {line.strip()}")
                    lines.append("")
                
                # Show relevant output (truncated)
                lines.append(f"\n**Output (last 100 lines):**")
                lines.append(f"```")
                output_lines = output.split('\n')
                if len(output_lines) > 100:
                    lines.append('\n'.join(output_lines[-100:]))
                else:
                    lines.append(output)
                lines.append(f"```\n")
            
            lines.append("---\n")
    else:
        lines.append("✅ No errors found! All tests passed successfully.\n")
    
    lines.append("---\n")
    
    # Recommendations
    lines.append("## 💡 Recommendations\n")
    
    if failed > 0:
        lines.append("### Action Items:\n")
        for r in errors_found:
            if r['status'] == 'IMPORT_ERROR':
                lines.append(f"- [ ] Install missing dependencies for `{r['name']}`")
            elif r['status'] == 'TIMEOUT':
                lines.append(f"- [ ] Optimize or break down `{r['name']}` (taking > 10 minutes)")
            elif r['status'] == 'FAILED':
                lines.append(f"- [ ] Fix failing assertions in `{r['name']}`")
            else:
                lines.append(f"- [ ] Debug errors in `{r['name']}`")
        lines.append("")
    
    if passed == total:
        lines.append("✅ All tests passing! System is ready for production.\n")
    elif passed > failed:
        lines.append("⚠️  Most tests passing, but some issues need attention.\n")
    else:
        lines.append("❌ Significant issues found. Address errors before proceeding.\n")
    
    lines.append("---\n")
    
    # System info
    lines.append("## 🖥️ System Information\n")
    lines.append(f"**Python:** {sys.version.split()[0]}")
    
    try:
        import torch
        lines.append(f"**PyTorch:** {torch.__version__}")
        if torch.cuda.is_available():
            lines.append(f"**GPU:** {torch.cuda.get_device_name(0)}")
            lines.append(f"**CUDA:** {torch.version.cuda}")
    except:
        pass
    
    lines.append("\n---\n")
    lines.append(f"\n*Generated by simple_test_runner.py*\n")
    
    # Write file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"\n✅ Report saved: {output_file}")

def main():
    print("="*70)
    print("🤖 TASK 9 - SIMPLE TEST RUNNER")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%I:%M %p')}\n")
    
    tests = [
        ('tests/task9/components/upscaler/test_upscaler_component.py', 'Upscaler Component'),
        ('tests/task9/components/temporal/test_temporal_simple.py', 'Temporal Consistency'),
        ('tests/task9/components/motion/test_motion_controller.py', 'Motion Controller'),
        ('tests/task9/integration/test_task9_simple.py', 'Simple Integration'),
        ('adapters/gurukul_lora/test_imports.py', 'Import Validation'),
        ('adapters/gurukul_lora/test_adapter.py', 'Adapter Functionality'),
    ]
    
    results = []
    
    for test_path, test_name in tests:
        result = run_single_test(test_path, test_name)
        results.append(result)
        time.sleep(1)  # Brief pause between tests
    
    # Generate report
    print(f"\n{'='*70}")
    print("📊 GENERATING REPORT")
    print("="*70)
    
    output_file = Path('AUTOMATION_TESTING_ERRORS_AND_BUGS.md')
    generate_report(results, output_file)
    
    # Summary
    print(f"\n{'='*70}")
    print("📈 SUMMARY")
    print("="*70)
    passed = sum(1 for r in results if r['status'] == 'PASSED')
    failed = sum(1 for r in results if r['status'] not in ['PASSED', 'SKIPPED'])
    print(f"✅ Passed: {passed}/{len(results)}")
    print(f"❌ Failed: {failed}/{len(results)}")
    print(f"\n✅ Report: {output_file.absolute()}")
    print(f"Completed: {datetime.now().strftime('%I:%M %p')}")
    print("="*70)

if __name__ == "__main__":
    main()
