# How to Run Benchmarks with open_bench.py

The `open_bench.py` script provides a command-line interface for benchmarking and optimizing AI pipelines without needing to use the web interface.

## Prerequisites

Before running benchmarks, ensure you have:

1. **Built the tool from source** or have the development environment set up. See [How to Build Source](how-to-build-source.md) for instructions.
2. **Python environment** with the required dependencies installed:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. **Video file** for testing (or use the included test videos)
4. **Models downloaded** - Run the models download script:
   ```bash
   bash models.sh
   ```

## Quick Start

### List Available Pipelines

To see which pipelines are available:

```bash
python open_bench.py list
```

Example output:
```
Available pipelines:
  ✓ smartnvr: Smart Network Video Recorder (NVR) Proxy Pipeline
  ✓ transportation2: Transportation Pipeline v2
```

### Run a Benchmark

Benchmark a pipeline to find the maximum number of streams that can achieve a target FPS:

```bash
python open_bench.py benchmark \
    --pipeline smartnvr \
    --video /path/to/video.mp4 \
    --device CPU \
    --fps-floor 30
```

**Parameters:**
- `--pipeline`: Name of the pipeline (e.g., `smartnvr`, `transportation2`)
- `--video`: Path to input video file
- `--device`: Device for inference - `CPU`, `GPU`, or `NPU` (default: `CPU`)
- `--fps-floor`: Minimum acceptable FPS per stream (default: `30.0`)
- `--rate`: Percentage of AI-enabled streams (default: `100`)
- `--batch-size`: Batch size for inference (default: `1`)
- `--nireq`: Number of inference requests (default: `1`)
- `--detection-model`: Object detection model to use (optional, uses pipeline default)

**Example Output:**
```
============================================================
BENCHMARK RESULTS
============================================================
✓ Success!
  Total Streams:     8
  AI Streams:        8
  Non-AI Streams:    0
  Per-Stream FPS:    30.45
  Total FPS:         243.60
============================================================
```

### Optimize Pipeline Parameters

Find the best combination of parameters for a given pipeline:

```bash
python open_bench.py optimize \
    --pipeline smartnvr \
    --video /path/to/video.mp4 \
    --device GPU \
    --batch-sizes 1 2 4 8 \
    --nireqs 1 2 4
```

**Parameters:**
- `--pipeline`: Name of the pipeline
- `--video`: Path to input video file
- `--device`: Device for inference - `CPU`, `GPU`, or `NPU`
- `--channels`: Number of video channels/streams (default: `1`)
- `--batch-size`: Default batch size (default: `1`)
- `--batch-sizes`: List of batch sizes to test (e.g., `1 2 4 8`)
- `--nireq`: Default number of inference requests (default: `1`)
- `--nireqs`: List of nireqs to test (e.g., `1 2 4`)
- `--detection-model`: Object detection model to use (optional)

**Example Output:**
```
============================================================
OPTIMIZATION RESULTS
============================================================
✓ Best configuration found!
  Parameters:        {'object_detection_device': 'GPU', 'object_detection_batch_size': 4, 'object_detection_nireq': 2}
  Total FPS:         245.32
  Per-Stream FPS:    245.32
  Exit Code:         0
============================================================
```

## Usage Examples

### Example 1: CPU Benchmark with Different FPS Floor

```bash
python open_bench.py benchmark \
    --pipeline smartnvr \
    --video videos/test_video.mp4 \
    --device CPU \
    --fps-floor 25 \
    --rate 100
```

### Example 2: GPU Optimization with Custom Parameters

```bash
python open_bench.py optimize \
    --pipeline smartnvr \
    --video videos/test_video.mp4 \
    --device GPU \
    --channels 4 \
    --batch-sizes 1 2 4 8 16 \
    --nireqs 1 2 4 8
```

### Example 3: NPU Benchmark with Specific Model

```bash
python open_bench.py benchmark \
    --pipeline smartnvr \
    --video videos/test_video.mp4 \
    --device NPU \
    --detection-model "YOLO v5s 416x416 (INT8)" \
    --fps-floor 30 \
    --batch-size 1 \
    --nireq 2
```

### Example 4: Mixed AI and Non-AI Streams

```bash
python open_bench.py benchmark \
    --pipeline smartnvr \
    --video videos/test_video.mp4 \
    --device CPU \
    --fps-floor 30 \
    --rate 50
```

This runs with 50% AI-enabled streams and 50% non-AI (recording-only) streams.

## Running in Docker

If you're using the Docker setup, you can run `open_bench.py` inside the container:

1. **Start the container**:
   ```bash
   source setup_env.sh -d cpu
   docker compose up -d
   ```

2. **Access the container**:
   ```bash
   docker exec -it vippet-cpu bash
   ```

3. **Run the benchmark inside the container**:
   ```bash
   # Navigate to the tool directory (default: /home/dlstreamer/vippet)
   cd $WORKDIR
   python open_bench.py benchmark \
       --pipeline smartnvr \
       --video videos/test_video.mp4 \
       --device CPU \
       --fps-floor 30
   ```

## Available Detection Models

The available detection models depend on the pipeline configuration. Common models include:

- `SSDLite MobileNet V2 (INT8)`
- `YOLO v5m 416x416 (INT8)`
- `YOLO v5s 416x416 (INT8)`
- `YOLO v5m 640x640 (INT8)`
- `YOLO v10s 640x640 (FP16)` (not supported on NPU)
- `YOLO v10m 640x640 (FP16)` (not supported on NPU)

To see which models are available for a specific pipeline, check the pipeline's `config.yaml` file in the `pipelines/<pipeline-name>/` directory.

## Troubleshooting

### Error: "Video file not found"

Ensure the video file path is correct and the file exists:
```bash
ls -la /path/to/video.mp4
```

### Error: "Pipeline not found"

List available pipelines to verify the name:
```bash
python open_bench.py list
```

### Error: "Model not supported on NPU"

Some models (like YOLO v10) don't support NPU. Choose a different model or device:
```bash
python open_bench.py benchmark \
    --pipeline smartnvr \
    --video videos/test_video.mp4 \
    --device CPU \
    --detection-model "YOLO v5s 416x416 (INT8)"
```

### Error: "No module named 'benchmark'"

Make sure you're running the script from the correct directory:
```bash
cd tools/visual-pipeline-and-platform-evaluation-tool
python open_bench.py --help
```

## Advanced Usage

### Custom Pipeline Development

If you're developing a custom pipeline:

1. Create a new directory under `pipelines/`
2. Add a `config.yaml` file with pipeline metadata
3. Create a `pipeline.py` with your pipeline class
4. Use `open_bench.py` to test your pipeline

### Scripting and Automation

You can use `open_bench.py` in automation scripts:

```bash
#!/bin/bash

# Run benchmarks across multiple devices
for device in CPU GPU NPU; do
    echo "Testing on $device..."
    python open_bench.py benchmark \
        --pipeline smartnvr \
        --video videos/test.mp4 \
        --device $device \
        --fps-floor 30
done
```

## Additional Resources

- [Get Started Guide](get-started.md)
- [Build from Source](how-to-build-source.md)
- [System Requirements](system-requirements.md)
- [Visual Pipeline and Platform Evaluation Tool Documentation](../README.md)
