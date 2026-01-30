#!/usr/bin/env python3
"""
Performance Benchmark for MemoryForge.

Tests memory creation, search, and consolidation with various memory counts.
"""

import statistics
import tempfile
import time
from pathlib import Path
from typing import Callable
from uuid import uuid4

from memoryforge.config import Config
from memoryforge.models import Memory, MemoryType, MemorySource, Project
from memoryforge.storage.sqlite_db import SQLiteDatabase
from memoryforge.storage.qdrant_store import QdrantStore

# Try to import local embedding service
try:
    from memoryforge.core.embedding_factory import create_embedding_service
    HAS_EMBEDDING = True
except ImportError:
    HAS_EMBEDDING = False


def benchmark(name: str, func: Callable, iterations: int = 10) -> dict:
    """Run a benchmark and return timing statistics."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    return {
        "name": name,
        "iterations": iterations,
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def run_benchmarks(memory_counts: list = [100, 1000, 5000, 10000]):
    """Run full benchmark suite."""
    print("=" * 60)
    print("MemoryForge Performance Benchmark")
    print("=" * 60)
    print()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "bench.db"
        qdrant_path = Path(tmpdir) / "qdrant"
        
        # Initialize storage
        db = SQLiteDatabase(db_path)
        
        # Create project
        project = Project(name="benchmark", root_path=tmpdir)
        project = db.create_project(project)
        
        results = []
        
        for count in memory_counts:
            print(f"\n--- Benchmarking with {count} memories ---\n")
            
            # Clear previous data
            for mem in db.get_memories(project.id, limit=count + 1000):
                db.delete_memory(mem.id)
            
            # Benchmark: Memory Creation
            created_ids = []
            
            def create_memory():
                mem = Memory(
                    content=f"Test memory {uuid4()} with content about software architecture and design patterns",
                    type=MemoryType.NOTE,
                    source=MemorySource.MANUAL,
                    project_id=project.id,
                    confirmed=False,
                )
                db.create_memory(mem)
                created_ids.append(mem.id)
            
            result = benchmark(f"Create memory ({count} total)", create_memory, min(count, 100))
            results.append(result)
            print(f"  Create: {result['mean_ms']:.2f}ms mean, {result['stdev_ms']:.2f}ms stdev")
            
            # Bulk create to reach target count
            for _ in range(count - len(created_ids)):
                create_memory()
            
            # Benchmark: Memory Retrieval by ID
            test_ids = created_ids[:10]
            
            def get_memory():
                for mid in test_ids:
                    db.get_memory(mid)
            
            result = benchmark(f"Get memory by ID ({count} total)", get_memory, 10)
            results.append(result)
            print(f"  Get by ID: {result['mean_ms']:.2f}ms mean")
            
            # Benchmark: List memories
            def list_memories():
                db.get_memories(project.id, limit=50)
            
            result = benchmark(f"List 50 memories ({count} total)", list_memories, 10)
            results.append(result)
            print(f"  List 50: {result['mean_ms']:.2f}ms mean")
            
            # Benchmark: Count memories
            def count_memories():
                db.get_memory_count(project.id)
            
            result = benchmark(f"Count memories ({count} total)", count_memories, 20)
            results.append(result)
            print(f"  Count: {result['mean_ms']:.2f}ms mean")
            
            # Benchmark: Confirm memory
            unconfirmed = created_ids[-10:]
            confirm_idx = [0]
            
            def confirm_memory():
                if confirm_idx[0] < len(unconfirmed):
                    db.confirm_memory(unconfirmed[confirm_idx[0]])
                    confirm_idx[0] += 1
            
            result = benchmark(f"Confirm memory ({count} total)", confirm_memory, 10)
            results.append(result)
            print(f"  Confirm: {result['mean_ms']:.2f}ms mean")
        
        # Print summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"\n{'Operation':<40} {'Mean (ms)':<12} {'Stdev (ms)':<12}")
        print("-" * 64)
        
        for r in results:
            print(f"{r['name']:<40} {r['mean_ms']:<12.2f} {r['stdev_ms']:<12.2f}")


def run_embedding_benchmark():
    """Benchmark embedding performance if available."""
    if not HAS_EMBEDDING:
        print("Embedding service not available, skipping embedding benchmarks")
        return
    
    print("\n" + "=" * 60)
    print("Embedding Benchmark")
    print("=" * 60)
    
    try:
        config = Config()
        embedding_service = create_embedding_service(config)
        
        test_texts = [
            "Short text",
            "Medium length text about software architecture and design patterns used in enterprise applications",
            "A longer text that describes the complete software architecture including microservices, event-driven design, database choices like PostgreSQL and Redis, API gateway patterns, and containerization with Docker and Kubernetes for deployment" * 2,
        ]
        
        for text in test_texts:
            def embed():
                embedding_service.generate(text)
            
            result = benchmark(f"Embed ({len(text)} chars)", embed, 5)
            print(f"  {len(text)} chars: {result['mean_ms']:.2f}ms mean")
    
    except Exception as e:
        print(f"Failed to run embedding benchmark: {e}")


if __name__ == "__main__":
    import sys
    
    counts = [100, 1000]
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        counts = [100, 1000, 5000, 10000]
    
    run_benchmarks(counts)
    run_embedding_benchmark()
