#!/usr/bin/env python3
"""
Test runner script with custom configurations and reporting.
"""
import sys
import os
import argparse
from pathlib import Path
import pytest
import subprocess
from datetime import datetime
import json


class TestRunner:
    """Custom test runner with enhanced reporting and configuration."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_dir = Path(__file__).parent
        
    def run_tests(
        self, 
        test_path: str = None,
        markers: str = None,
        coverage: bool = True,
        verbose: bool = False,
        parallel: bool = False,
        html_report: bool = True,
        junit_xml: bool = False,
        timeout: int = 300
    ):
        """Run tests with specified configuration."""
        
        # Build pytest command
        cmd = ["python", "-m", "pytest"]
        
        # Test path
        if test_path:
            cmd.append(test_path)
        else:
            cmd.append(str(self.test_dir))
        
        # Markers
        if markers:
            cmd.extend(["-m", markers])
        
        # Verbosity
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        # Coverage
        if coverage:
            cmd.extend([
                "--cov=.",
                "--cov-report=term-missing",
                "--cov-fail-under=70"
            ])
            
            if html_report:
                cmd.append("--cov-report=html:htmlcov")
        
        # Parallel execution
        if parallel:
            try:
                import pytest_xdist
                cmd.extend(["-n", "auto"])
            except ImportError:
                print("Warning: pytest-xdist not installed, running serially")
        
        # JUnit XML report
        if junit_xml:
            cmd.extend(["--junit-xml", "test-results.xml"])
        
        # Timeout
        cmd.extend(["--timeout", str(timeout)])
        
        # Additional options
        cmd.extend([
            "--strict-markers",
            "--tb=short",
            "--disable-warnings"
        ])
        
        print(f"Running command: {' '.join(cmd)}")
        print(f"Working directory: {self.project_root}")
        
        # Execute tests
        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=False,
            text=True
        )
        
        return result.returncode
    
    def run_performance_tests(self):
        """Run performance tests with specific configuration."""
        print("Running performance tests...")
        return self.run_tests(
            markers="slow",
            verbose=True,
            parallel=True,
            timeout=600
        )
    
    def run_integration_tests(self):
        """Run integration tests."""
        print("Running integration tests...")
        return self.run_tests(
            markers="integration",
            verbose=True,
            timeout=300
        )
    
    def run_unit_tests(self):
        """Run unit tests."""
        print("Running unit tests...")
        return self.run_tests(
            markers="unit",
            verbose=False,
            parallel=True,
            timeout=120
        )
    
    def run_data_tests(self):
        """Run data validation tests."""
        print("Running data validation tests...")
        return self.run_tests(
            markers="data",
            verbose=True,
            timeout=180
        )
    
    def run_api_tests(self):
        """Run API tests."""
        print("Running API tests...")
        return self.run_tests(
            markers="api",
            verbose=True,
            timeout=240
        )
    
    def run_smoke_tests(self):
        """Run smoke tests (fast subset)."""
        print("Running smoke tests...")
        return self.run_tests(
            markers="unit and not slow",
            verbose=False,
            parallel=True,
            timeout=60
        )
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.project_root / f"test_report_{timestamp}.json"
        
        # Run different test suites and collect results
        results = {}
        
        test_suites = [
            ("unit", self.run_unit_tests),
            ("integration", self.run_integration_tests),
            ("data", self.run_data_tests),
            ("api", self.run_api_tests),
        ]
        
        for suite_name, test_func in test_suites:
            print(f"\n{'='*50}")
            print(f"Running {suite_name} tests")
            print(f"{'='*50}")
            
            start_time = datetime.now()
            exit_code = test_func()
            end_time = datetime.now()
            
            results[suite_name] = {
                "exit_code": exit_code,
                "duration": (end_time - start_time).total_seconds(),
                "status": "PASSED" if exit_code == 0 else "FAILED"
            }
        
        # Generate report
        report = {
            "timestamp": timestamp,
            "project_root": str(self.project_root),
            "test_results": results,
            "overall_status": "PASSED" if all(r["exit_code"] == 0 for r in results.values()) else "FAILED"
        }
        
        # Write report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*50}")
        print("TEST REPORT SUMMARY")
        print(f"{'='*50}")
        for suite_name, result in results.items():
            status_emoji = "✅" if result["status"] == "PASSED" else "❌"
            print(f"{status_emoji} {suite_name.upper()}: {result['status']} ({result['duration']:.1f}s)")
        
        print(f"\nFull report saved to: {report_file}")
        print(f"Overall status: {report['overall_status']}")
        
        return report
    
    def setup_test_environment(self):
        """Set up test environment with required dependencies."""
        print("Setting up test environment...")
        
        required_packages = [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "pytest-xdist>=3.0.0",
            "pytest-timeout>=2.1.0",
            "pandera>=0.15.0",
            "pytest-html>=3.1.0"
        ]
        
        print("Installing required test dependencies...")
        for package in required_packages:
            subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], check=False, capture_output=True)
        
        print("Test environment setup complete.")
    
    def validate_test_structure(self):
        """Validate test directory structure and configuration."""
        print("Validating test structure...")
        
        required_files = [
            "pytest.ini",
            "conftest.py",
            "tests/unit/__init__.py",
            "tests/integration/__init__.py",
            "tests/data/__init__.py",
            "tests/mocks/__init__.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            print("❌ Missing required test files:")
            for file_path in missing_files:
                print(f"  - {file_path}")
            return False
        
        print("✅ Test structure validation passed")
        return True
    
    def clean_test_artifacts(self):
        """Clean up test artifacts and cache."""
        print("Cleaning test artifacts...")
        
        artifacts_to_clean = [
            ".pytest_cache",
            "htmlcov",
            ".coverage",
            "test-results.xml",
            "__pycache__"
        ]
        
        for artifact in artifacts_to_clean:
            artifact_path = self.project_root / artifact
            if artifact_path.exists():
                if artifact_path.is_dir():
                    import shutil
                    shutil.rmtree(artifact_path, ignore_errors=True)
                else:
                    artifact_path.unlink()
        
        print("✅ Test artifacts cleaned")


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Enhanced test runner for trading project")
    
    parser.add_argument("--setup", action="store_true", help="Set up test environment")
    parser.add_argument("--validate", action="store_true", help="Validate test structure")
    parser.add_argument("--clean", action="store_true", help="Clean test artifacts")
    parser.add_argument("--report", action="store_true", help="Generate comprehensive test report")
    
    # Test execution options
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--data", action="store_true", help="Run data tests only")
    parser.add_argument("--api", action="store_true", help="Run API tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests")
    
    # Configuration options
    parser.add_argument("--path", type=str, help="Specific test path to run")
    parser.add_argument("--markers", type=str, help="Pytest markers to filter tests")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", "-p", action="store_true", help="Run tests in parallel")
    parser.add_argument("--timeout", type=int, default=300, help="Test timeout in seconds")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    exit_code = 0
    
    try:
        # Setup and validation commands
        if args.setup:
            runner.setup_test_environment()
            return 0
        
        if args.validate:
            success = runner.validate_test_structure()
            return 0 if success else 1
        
        if args.clean:
            runner.clean_test_artifacts()
            return 0
        
        if args.report:
            report = runner.generate_test_report()
            return 0 if report["overall_status"] == "PASSED" else 1
        
        # Test execution commands
        if args.unit:
            exit_code = runner.run_unit_tests()
        elif args.integration:
            exit_code = runner.run_integration_tests()
        elif args.data:
            exit_code = runner.run_data_tests()
        elif args.api:
            exit_code = runner.run_api_tests()
        elif args.performance:
            exit_code = runner.run_performance_tests()
        elif args.smoke:
            exit_code = runner.run_smoke_tests()
        else:
            # Run all tests with custom configuration
            exit_code = runner.run_tests(
                test_path=args.path,
                markers=args.markers,
                coverage=not args.no_coverage,
                verbose=args.verbose,
                parallel=args.parallel,
                timeout=args.timeout
            )
        
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        exit_code = 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        exit_code = 1
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)