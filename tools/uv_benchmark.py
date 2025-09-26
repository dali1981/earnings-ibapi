#!/usr/bin/env python3
"""UV vs pip performance benchmarking tool."""

import os
import sys
import time
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

class DependencyBenchmark:
    """Benchmark dependency installation performance."""
    
    def __init__(self):
        self.results = {}
        self.test_packages = [
            "fastapi>=0.111",
            "uvicorn[standard]>=0.30", 
            "pandas==2.2.2",
            "numpy==2.0.2",
            "pytest>=7.0.0",
            "celery>=5.3",
            "httpx>=0.27",
        ]
    
    def benchmark_uv_installation(self, temp_dir: Path) -> Dict:
        """Benchmark UV installation performance."""
        print("🔄 Benchmarking UV installation...")
        
        # Create UV project
        os.chdir(temp_dir)
        
        # Initialize UV project
        init_start = time.time()
        result = subprocess.run(["uv", "init", "--name", "benchmark-test"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"UV init failed: {result.stderr}")
            return {}
        init_time = time.time() - init_start
        
        # Add dependencies one by one
        add_times = []
        total_add_start = time.time()
        
        for package in self.test_packages:
            add_start = time.time()
            result = subprocess.run(["uv", "add", package], 
                                  capture_output=True, text=True)
            add_time = time.time() - add_start
            add_times.append(add_time)
            
            if result.returncode != 0:
                print(f"UV add {package} failed: {result.stderr}")
                continue
        
        total_add_time = time.time() - total_add_start
        
        # Sync installation (equivalent to pip install -r requirements.txt)
        sync_start = time.time()
        result = subprocess.run(["uv", "sync"], capture_output=True, text=True)
        sync_time = time.time() - sync_start
        
        if result.returncode != 0:
            print(f"UV sync failed: {result.stderr}")
            sync_time = float('inf')
        
        return {
            "tool": "uv",
            "init_time": init_time,
            "individual_add_times": add_times,
            "total_add_time": total_add_time,
            "sync_time": sync_time,
            "total_time": init_time + total_add_time + sync_time
        }
    
    def benchmark_pip_installation(self, temp_dir: Path) -> Dict:
        """Benchmark pip installation performance."""
        print("🔄 Benchmarking pip installation...")
        
        os.chdir(temp_dir)
        
        # Create virtual environment
        venv_start = time.time()
        result = subprocess.run([sys.executable, "-m", "venv", "venv"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Venv creation failed: {result.stderr}")
            return {}
        venv_time = time.time() - venv_start
        
        # Create requirements file
        req_file = temp_dir / "requirements.txt"
        with open(req_file, "w") as f:
            for package in self.test_packages:
                f.write(f"{package}\n")
        
        # Install dependencies
        pip_executable = temp_dir / "venv" / "bin" / "pip"
        if not pip_executable.exists():
            pip_executable = temp_dir / "venv" / "Scripts" / "pip.exe"  # Windows
        
        install_start = time.time()
        result = subprocess.run([str(pip_executable), "install", "-r", "requirements.txt"], 
                              capture_output=True, text=True)
        install_time = time.time() - install_start
        
        if result.returncode != 0:
            print(f"Pip install failed: {result.stderr}")
            install_time = float('inf')
        
        return {
            "tool": "pip",
            "venv_time": venv_time,
            "install_time": install_time,
            "total_time": venv_time + install_time
        }
    
    def run_benchmark(self) -> Dict:
        """Run complete benchmark comparison."""
        results = {}
        
        # UV benchmark
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            uv_dir = temp_path / "uv_test"
            uv_dir.mkdir()
            
            uv_results = self.benchmark_uv_installation(uv_dir)
            results["uv"] = uv_results
        
        # pip benchmark  
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pip_dir = temp_path / "pip_test"
            pip_dir.mkdir()
            
            pip_results = self.benchmark_pip_installation(pip_dir)
            results["pip"] = pip_results
        
        return results
    
    def print_results(self, results: Dict):
        """Print benchmark results."""
        print("\n" + "="*60)
        print("🏁 DEPENDENCY INSTALLATION BENCHMARK RESULTS")
        print("="*60)
        
        if "uv" in results:
            uv = results["uv"]
            print(f"\n🚀 UV Results:")
            print(f"  Init time:     {uv.get('init_time', 0):.2f}s")
            print(f"  Add deps time: {uv.get('total_add_time', 0):.2f}s")
            print(f"  Sync time:     {uv.get('sync_time', 0):.2f}s")
            print(f"  Total time:    {uv.get('total_time', 0):.2f}s")
        
        if "pip" in results:
            pip = results["pip"]
            print(f"\n🐍 pip Results:")
            print(f"  Venv time:     {pip.get('venv_time', 0):.2f}s")
            print(f"  Install time:  {pip.get('install_time', 0):.2f}s")
            print(f"  Total time:    {pip.get('total_time', 0):.2f}s")
        
        # Performance comparison
        if "uv" in results and "pip" in results:
            uv_total = results["uv"].get("total_time", float('inf'))
            pip_total = results["pip"].get("total_time", float('inf'))
            
            if uv_total < pip_total:
                speedup = pip_total / uv_total
                print(f"\n⚡ Performance Improvement:")
                print(f"  UV is {speedup:.1f}x faster than pip")
                print(f"  Time saved: {pip_total - uv_total:.1f} seconds")
            else:
                slowdown = uv_total / pip_total
                print(f"\n⚠️  Performance Analysis:")
                print(f"  pip is {slowdown:.1f}x faster than UV")
                print(f"  Additional time: {uv_total - pip_total:.1f} seconds")
        
        print("\n" + "="*60)


def main():
    """Run UV vs pip benchmark."""
    print("🏁 Starting UV vs pip performance benchmark...")
    print(f"Testing with {len(DependencyBenchmark().test_packages)} packages")
    
    benchmark = DependencyBenchmark()
    results = benchmark.run_benchmark()
    benchmark.print_results(results)
    
    # Save results for future reference
    import json
    results_file = Path("uv_benchmark_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"📊 Results saved to: {results_file}")


if __name__ == "__main__":
    main()