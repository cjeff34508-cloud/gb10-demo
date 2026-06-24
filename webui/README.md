# GB10 Demo Suite - Streamlit WebUI

Interactive web interface for benchmarking model inference across multiple precisions (FP4-FP32).

## Features

✅ **Model Switching**: Switch between LLM, Vision, and HPC workloads  
✅ **Precision Benchmarking**: Compare FP4, FP8, FP16, FP32 latency & memory  
✅ **Real-time GPU Monitoring**: Live GPU memory tracking  
✅ **Interactive Charts**: Latency, memory, and speedup visualizations  
✅ **Results Export**: Download benchmark results as CSV  

## Installation

### Step 1: Install WebUI Dependencies

```bash
# Use the LLM venv (has most dependencies already)
source ~/gb10-demo/env/llm/bin/activate

# Install additional WebUI packages
pip install -r ~/gb10-demo/webui/requirements-webui.txt
```

### Step 2: Verify Setup

```bash
# Check that torch and CUDA are available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check that streamlit is installed
streamlit --version
```

## Running the WebUI

```bash
# Activate the LLM venv (or any venv with torch + streamlit)
source ~/gb10-demo/env/llm/bin/activate

# Start the Streamlit app
streamlit run ~/gb10-demo/webui/streamlit_app.py
```

The app will open at `http://localhost:8501` in your browser.

## Usage

### Basic Workflow

1. **Configure** (sidebar):
   - Select workload type (Vision, LLM, or HPC)
   - Choose model to benchmark
   - Select precisions (FP32, FP16, FP8, etc.)
   - Set number of runs and batch size

2. **Run** (sidebar):
   - Click **Run Benchmark** button
   - Watch progress in the **Benchmark** tab

3. **Analyze** (Results tab):
   - View latency and memory metrics
   - Compare speedup vs FP32
   - Download results as CSV

### Benchmark Types

#### Vision Workloads (CLIP, ViT)
- Image classification and embedding
- Models: `openai/clip-vit-base-patch32`, `google/vit-base-patch16-224`
- Benchmarks: Embedding latency, memory usage

#### LLM Workloads
- Text generation and inference
- Models: `gpt2`, `meta-llama/Llama-2-7b-hf`, `mistralai/Mistral-7B`
- Benchmarks: Token latency, generation speed

#### HPC Workloads
- Compute kernels and memory bandwidth
- Benchmarks: MatMul, Bandwidth test, Reduction operations
- Precisions: FP32, FP16, BF16, FP64

## Performance Tips

- **First Run**: Model download may take time. Check `~/gb10-demo/logs/model-download.log`
- **GPU Memory**: Monitor GPU memory in the sidebar. Reduce batch size if OOM
- **Warmup Runs**: Increase for more accurate measurements (default: 1)
- **Unload Models**: Models are automatically unloaded after each benchmark

## Troubleshooting

### "No GPU detected"
```bash
# Check CUDA
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### "Model not found"
- Run model downloader: `python ~/gb10-demo/scripts/model-downloader.py`
- Check models exist: `ls ~/gb10-demo/models/*/`

### "Out of Memory (OOM)"
- Reduce batch size (sidebar slider)
- Reduce number of benchmark runs
- Close other GPU-using applications
- Check: `nvidia-smi`

### Streamlit Errors
```bash
# Reinstall streamlit
pip install --upgrade streamlit

# Clear Streamlit cache
streamlit cache clear
```

## File Structure

```
~/gb10-demo/webui/
├── streamlit_app.py          # Main UI
├── helpers/
│   ├── benchmark_utils.py    # Timing, memory profiling
│   ├── llm_inference.py      # LLM benchmarks
│   ├── vision_inference.py   # Vision model benchmarks
│   ├── hpc_compute.py        # HPC benchmarks
│   └── __init__.py
├── requirements-webui.txt    # Dependencies
└── README.md                 # This file
```

## Example: Benchmarking CLIP across precisions

1. Start the app: `streamlit run streamlit_app.py`
2. **Settings**: Select "Vision (CLIP, ViT)" → `openai/clip-vit-base-patch32`
3. **Precisions**: Select [FP32, FP16, FP8] (uncheck FP4 if unsupported)
4. **Run**: Click **Run Benchmark** and watch progress
5. **Results**: View latency/memory charts, see speedup vs FP32
6. **Export**: Download CSV with detailed results

## Advanced Configuration

### Customize Models

Edit `~/gb10-demo/config/model-lists.json` and restart the app:

```json
{
  "llm": ["meta-llama/Llama-2-7b-hf", "google/flan-t5-large"],
  "vlm": ["openai/clip-vit-large-patch14"],
  "cnn": ["timm:resnet50.a1_in1k"]
}
```

### Add Custom Benchmarks

Create new benchmark classes in `helpers/custom_benchmarks.py`:

```python
from helpers.benchmark_utils import BenchmarkMetrics

class CustomBenchmark:
    @staticmethod
    def my_benchmark():
        metrics = BenchmarkMetrics("MyModel", "FP32")
        # ... implement benchmark ...
        return metrics
```

Then import and use in `streamlit_app.py`.

## Performance Expectations (GB10)

| Model | Precision | Latency | Memory | Speedup |
|-------|-----------|---------|--------|---------|
| CLIP-Base | FP32 | ~50ms | 350MB | 1.0× |
| CLIP-Base | FP16 | ~30ms | 200MB | 1.7× |
| CLIP-Base | FP8 | ~20ms | 120MB | 2.5× |
| MatMul 1000×1000 | FP32 | ~2.5ms | 8MB | 1.0× |
| MatMul 1000×1000 | FP16 | ~1.5ms | 4MB | 1.7× |

*Approximate values; actual results depend on GPU load and model size*

## Support

For issues or questions:
1. Check logs: `tail -f ~/gb10-demo/logs/model-download.log`
2. Review main README: `~/gb10-demo/README.md`
3. Check CUDA setup: `nvidia-smi`

---

**Created**: 2026-06-09  
**For**: NVIDIA GB10 Sales Demos
