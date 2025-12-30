#!/usr/bin/env python3
"""open_bench.py

Command-line interface for benchmarking pipelines in the Visual Pipeline and Platform Evaluation Tool.

This script allows you to run benchmarks and optimizations on AI pipelines from the command line,
without needing to use the web interface.

Usage:
    # Run a benchmark on the Smart NVR pipeline
    python open_bench.py benchmark --pipeline smartnvr --video /path/to/video.mp4 --device CPU --fps-floor 30

    # Run optimization to find best parameters
    python open_bench.py optimize --pipeline smartnvr --video /path/to/video.mp4 --device CPU
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

from benchmark import Benchmark
from optimize import PipelineOptimizer
from pipeline import PipelineLoader
from utils import prepare_video_and_constants

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default optimization parameter values
DEFAULT_BATCH_SIZES = [1, 2, 4]
DEFAULT_NIREQS = [1, 2, 4]


def _build_parameter_dict(device: str, batch_sizes: list, nireqs: list) -> dict:
    """Build parameter dictionary for pipeline execution.
    
    Args:
        device: Device name (CPU, GPU, NPU)
        batch_sizes: List of batch sizes to test
        nireqs: List of nireq values to test
        
    Returns:
        Dictionary of parameters with lists of values
    """
    return {
        "object_detection_device": [device],
        "object_detection_batch_size": batch_sizes,
        "object_detection_nireq": nireqs,
    }


def _get_detection_model_config(config: dict) -> tuple:
    """Extract detection model configuration from pipeline config.
    
    Returns:
        tuple: (detection_models list, detection_default string)
    """
    inference_config = config.get("parameters", {}).get("inference", {})
    detection_models = inference_config.get("detection_models", [])
    detection_default = inference_config.get("detection_model_default", "")
    return detection_models, detection_default


def _prepare_pipeline_constants(video_path: Path, detection_model: str, args) -> tuple:
    """Prepare constants and parameters for pipeline execution.
    
    Returns:
        tuple: (video_output_path, constants, param_grid)
    """
    return prepare_video_and_constants(
        input_video_player=str(video_path.absolute()),
        object_detection_model=detection_model,
        object_detection_device=args.device,
        object_detection_batch_size=getattr(args, 'batch_size', 1),
        object_detection_inference_interval=0.0,
        object_detection_nireq=getattr(args, 'nireq', 1),
        object_classification_model="Disabled",
        object_classification_device=args.device,
        object_classification_batch_size=1,
        object_classification_inference_interval=0.0,
        object_classification_reclassify_interval=0.0,
        object_classification_nireq=1,
        pipeline_watermark_enabled=False,
    )


def list_pipelines():
    """List all available pipelines."""
    try:
        pipelines = PipelineLoader.list()
        print("Available pipelines:")
        for pipeline_name in pipelines:
            try:
                config = PipelineLoader.config(pipeline_name)
                display_name = config.get("name", pipeline_name)
                enabled = config.get("metadata", {}).get("enabled", False)
                status = "✓" if enabled else "✗"
                print(f"  {status} {pipeline_name}: {display_name}")
            except Exception as e:
                print(f"  ? {pipeline_name}: (error loading config: {e})")
        return pipelines
    except Exception as e:
        logger.error(f"Error listing pipelines: {e}")
        return []


def run_benchmark(args):
    """Run a benchmark on the specified pipeline."""
    try:
        # Load the pipeline
        logger.info(f"Loading pipeline: {args.pipeline}")
        pipeline_cls, config = PipelineLoader.load(args.pipeline)
        
        # Check if video file exists
        video_path = Path(args.video)
        if not video_path.exists():
            logger.error(f"Video file not found: {args.video}")
            return 1
        
        # Prepare constants and parameters
        logger.info("Preparing pipeline configuration...")
        
        # Get default models from config
        detection_models, detection_default = _get_detection_model_config(config)
        
        if args.detection_model:
            if args.detection_model not in detection_models:
                logger.warning(f"Model '{args.detection_model}' not in available models: {detection_models}")
                logger.info(f"Using anyway: {args.detection_model}")
            detection_model = args.detection_model
        else:
            detection_model = detection_default
            logger.info(f"Using default detection model: {detection_model}")
        
        # Prepare video and constants
        video_output_path, constants, param_grid = _prepare_pipeline_constants(
            video_path, detection_model, args
        )
        
        # Build parameters for benchmark
        parameters = _build_parameter_dict(
            device=args.device,
            batch_sizes=[args.batch_size],
            nireqs=[args.nireq]
        )
        
        # Create and run benchmark
        logger.info(f"Starting benchmark with FPS floor: {args.fps_floor}, AI stream rate: {args.rate}%")
        benchmark = Benchmark(
            video_path=str(video_path.absolute()),
            pipeline_cls=pipeline_cls,
            fps_floor=args.fps_floor,
            rate=args.rate,
            parameters=parameters,
            constants=constants,
        )
        
        result = benchmark.run()
        total_streams, ai_streams, non_ai_streams, per_stream_fps = result
        
        # Display results
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        if total_streams > 0:
            print(f"✓ Success!")
            print(f"  Total Streams:     {total_streams}")
            print(f"  AI Streams:        {ai_streams}")
            print(f"  Non-AI Streams:    {non_ai_streams}")
            print(f"  Per-Stream FPS:    {per_stream_fps:.2f}")
            print(f"  Total FPS:         {per_stream_fps * total_streams:.2f}")
        else:
            print(f"✗ Failed to find a valid configuration")
            print(f"  The pipeline could not achieve the target FPS floor of {args.fps_floor}")
        print("="*60)
        
        return 0 if total_streams > 0 else 1
        
    except Exception as e:
        logger.error(f"Error running benchmark: {e}", exc_info=True)
        return 1


def run_optimize(args):
    """Run optimization to find the best parameters."""
    try:
        # Load the pipeline
        logger.info(f"Loading pipeline: {args.pipeline}")
        pipeline_cls, config = PipelineLoader.load(args.pipeline)
        
        # Check if video file exists
        video_path = Path(args.video)
        if not video_path.exists():
            logger.error(f"Video file not found: {args.video}")
            return 1
        
        # Prepare constants and parameters
        logger.info("Preparing pipeline configuration...")
        
        # Get default models from config
        detection_models, detection_default = _get_detection_model_config(config)
        
        detection_model = args.detection_model if args.detection_model else detection_default
        
        # Prepare video and constants
        video_output_path, constants, param_grid = _prepare_pipeline_constants(
            video_path, detection_model, args
        )
        
        # Build parameter grid for optimization
        param_grid = _build_parameter_dict(
            device=args.device,
            batch_sizes=args.batch_sizes if args.batch_sizes else DEFAULT_BATCH_SIZES,
            nireqs=args.nireqs if args.nireqs else DEFAULT_NIREQS
        )
        
        # Create and run optimizer
        logger.info(f"Starting optimization with parameter grid: {param_grid}")
        optimizer = PipelineOptimizer(
            pipeline=pipeline_cls,
            constants=constants,
            param_grid=param_grid,
            channels=args.channels,
        )
        
        optimizer.optimize()
        best_result = optimizer.evaluate()
        
        # Display results
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS")
        print("="*60)
        if best_result:
            print(f"✓ Best configuration found!")
            print(f"  Parameters:        {best_result.params}")
            print(f"  Total FPS:         {best_result.total_fps:.2f}")
            print(f"  Per-Stream FPS:    {best_result.per_stream_fps:.2f}")
            print(f"  Exit Code:         {best_result.exit_code}")
        else:
            print(f"✗ No valid configuration found")
        print("="*60)
        
        return 0 if best_result else 1
        
    except Exception as e:
        logger.error(f"Error running optimization: {e}", exc_info=True)
        return 1


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Command-line interface for Visual Pipeline and Platform Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available pipelines
  %(prog)s list

  # Run a benchmark
  %(prog)s benchmark --pipeline smartnvr --video /path/to/video.mp4 --device CPU --fps-floor 30

  # Run optimization
  %(prog)s optimize --pipeline smartnvr --video /path/to/video.mp4 --device GPU

  # Run with custom batch sizes and nireqs
  %(prog)s optimize --pipeline smartnvr --video /path/to/video.mp4 --device CPU \\
      --batch-sizes 1 2 4 8 --nireqs 1 2 4
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available pipelines')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run a benchmark on a pipeline')
    bench_parser.add_argument('--pipeline', required=True, help='Pipeline name (e.g., smartnvr)')
    bench_parser.add_argument('--video', required=True, help='Path to input video file')
    bench_parser.add_argument('--device', default='CPU', choices=['CPU', 'GPU', 'NPU'], 
                              help='Device to run inference on (default: CPU)')
    bench_parser.add_argument('--fps-floor', type=float, default=30.0, 
                              help='Minimum acceptable FPS per stream (default: 30.0)')
    bench_parser.add_argument('--rate', type=int, default=100, 
                              help='Percentage of AI streams (default: 100)')
    bench_parser.add_argument('--batch-size', type=int, default=1, 
                              help='Batch size for inference (default: 1)')
    bench_parser.add_argument('--nireq', type=int, default=1, 
                              help='Number of inference requests (default: 1)')
    bench_parser.add_argument('--detection-model', help='Object detection model to use')
    
    # Optimize command
    opt_parser = subparsers.add_parser('optimize', help='Optimize pipeline parameters')
    opt_parser.add_argument('--pipeline', required=True, help='Pipeline name (e.g., smartnvr)')
    opt_parser.add_argument('--video', required=True, help='Path to input video file')
    opt_parser.add_argument('--device', default='CPU', choices=['CPU', 'GPU', 'NPU'], 
                            help='Device to run inference on (default: CPU)')
    opt_parser.add_argument('--channels', type=int, default=1, 
                            help='Number of channels/streams (default: 1)')
    opt_parser.add_argument('--batch-size', type=int, default=1, 
                            help='Default batch size (default: 1)')
    opt_parser.add_argument('--batch-sizes', type=int, nargs='+', 
                            help='List of batch sizes to test (e.g., 1 2 4 8)')
    opt_parser.add_argument('--nireq', type=int, default=1, 
                            help='Default number of inference requests (default: 1)')
    opt_parser.add_argument('--nireqs', type=int, nargs='+', 
                            help='List of nireqs to test (e.g., 1 2 4)')
    opt_parser.add_argument('--detection-model', help='Object detection model to use')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_pipelines()
        return 0
    elif args.command == 'benchmark':
        return run_benchmark(args)
    elif args.command == 'optimize':
        return run_optimize(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
