# HybrIK Server

---
**[변경 요약] Visualization Backend 교체**
- 기존 PyTorch3D 기반 시각화를 **폐기**했습니다.
- 대체 도구: **trimesh + pyrender** (경량, Mac M1/M2 호환 용이).
- 추론/IK/관절 정확도에는 영향 없음. 렌더는 디버깅/검증 용도에 국한.
- 구현 위치: `hybrik-server/vis/` 유틸. 설정은 `settings.yaml` 참조.
---

Independent REST API server for 3D human pose estimation using HybrIK.

## Overview

HybrIK Server provides a REST API wrapper around the HybrIK model for 3D human pose and shape estimation. It offers standardized JSON APIs for single image analysis with configurable output formats.

## Features

- **3D Joint Estimation**: Extract 24-joint 3D pose coordinates from single images
- **Lightweight Visualization**: trimesh + pyrender backend (Mac M1/M2 optimized)
- **Debug Visualization**: matplotlib-based skeleton plotting for development
- **Mock Mode Support**: Fully functional without pretrained models for testing
- **Flexible Output**: Support for mm/normalized units and different coordinate frames
- **REST API**: Simple HTTP endpoints with JSON input/output
- **Multi-Platform**: CPU, CUDA, and MPS (Apple Silicon) support
- **Configurable**: YAML-based configuration for easy customization
- **Production Ready**: Docker support, health checks, and logging

## Quick Start

### 1. Setup Environment

**Option A: Using Conda (Recommended for Mac M1/M2)**
```bash
# Install Miniconda if not already installed
# Download from: https://docs.conda.io/en/latest/miniconda.html

# Create conda environment
conda create -n hybrik python=3.8
conda activate hybrik

# Clone and setup
git clone <this-repo>
cd hybrik-server

# Install dependencies
pip install -r requirements.txt
```

**Option B: Using Python venv**
```bash
# Clone and setup
git clone <this-repo>
cd hybrik-server

# Setup Python environment and dependencies
./scripts/setup.sh
```

### 2. Install HybrIK (Optional - Mock Mode Available)

**For Full Functionality:**
```bash
# Clone HybrIK repository (adjacent to hybrik-server)
cd ..
git clone https://github.com/jeffffffli/HybrIK.git
cd HybrIK

# Install HybrIK dependencies manually (due to version conflicts)
pip install numpy six terminaltables scipy cython matplotlib \
            pycocotools tqdm easydict chumpy pyyaml tb-nightly \
            future ffmpeg-python joblib
```

**For Testing/Development:**
Server works in Mock Mode without HybrIK installation - generates random but valid joint data for API testing.

### 3. Download Models

```bash
cd ../hybrik-server
mkdir -p pretrained_models

# Download HybrIK pretrained model
# TODO: Add specific download instructions
```

### 4. Configure

Edit `settings.yaml` to set correct paths:

```yaml
model:
  hybrik_cfg: "configs/hybrik_hrnet.yaml"
  hybrik_ckpt: "pretrained_models/hybrik_hrnet.pth"
  device: "cpu"  # or "cuda", "mps"
```

### 5. Run Server

```bash
# Development mode
./scripts/run.sh dev

# Production mode  
./scripts/run.sh prod
```

### 6. Test

```bash
# Health check
./scripts/healthcheck.sh

# Or manually
curl http://localhost:5002/health

# Comprehensive API test
python test_image_analysis.py
```

## API Reference

### Health Check

```http
GET/POST /health
```

Response:
```json
{
  "status": "ok",
  "version": {
    "app": "1.0.0",
    "hybrik": "latest", 
    "device": "cpu"
  },
  "ready": true
}
```

### Analyze Image

```http
POST /analyze-image
```

**Multipart Upload:**
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5002/analyze-image
```

**JSON Base64:**
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "base64_encoded_image"}' \
  http://localhost:5002/analyze-image
```

**Response:**
```json
{
  "joints3d": [[x, y, z], ...],
  "joints2d": [[u, v], ...],
  "confidence": [0.9, 0.8, ...],
  "debug_visualization": {
    "skeleton_plot": "./debug_vis/skeleton_plot_timestamp.png",
    "skeleton_render": "./debug_vis/skeleton_render_timestamp.png"
  },
  "meta": {
    "units": "mm",
    "coord_frame": "root-relative",
    "joint_order": "hybrik_24_joints",
    "latency_ms": 123.45,
    "version": {
      "app": "1.0.0",
      "hybrik": "latest",
      "model": "hybrik_hrnet.pth"
    }
  }
}
```

## Configuration

Key settings in `settings.yaml`:

```yaml
model:
  device: "cpu"           # cpu, cuda, mps
  hybrik_cfg: "path/to/config.yaml"
  hybrik_ckpt: "path/to/model.pth"

output:
  units: "mm"             # mm, norm
  coord_frame: "root-relative"  # root-relative, camera
  return_2d_joints: true
  return_smpl_params: false

visualization:
  backend: "trimesh+pyrender"  # Visualization backend
  debug_vis: false             # Enable debug visualization
  vis_output_dir: "./debug_vis" # Debug output directory

limits:
  max_image_resolution: 2048
  max_file_mb: 10
  request_timeout_ms: 30000
```

## Docker Deployment

```bash
# Build image
docker build -t hybrik-server .

# Run container
docker run -p 5002:5002 \
  -v $(pwd)/pretrained_models:/app/pretrained_models \
  -v $(pwd)/configs:/app/configs \
  hybrik-server
```

## Joint Ordering

The server returns 24 joints in the following order:

```
0: pelvis           12: neck
1: left_hip         13: left_collar  
2: right_hip        14: right_collar
3: spine1           15: head
4: left_knee        16: left_shoulder
5: right_knee       17: right_shoulder
6: spine2           18: left_elbow
7: left_ankle       19: right_elbow
8: right_ankle      20: left_wrist
9: spine3           21: right_wrist
10: left_foot       22: left_hand
11: right_foot      23: right_hand
```

## Development

### Project Structure

```
hybrik-server/
├── app.py                 # Flask REST API server
├── hybrik_runner.py       # HybrIK model wrapper
├── settings.yaml          # Configuration
├── requirements.txt       # Python dependencies
├── test_image_analysis.py # Comprehensive API test
├── Dockerfile            # Container definition
├── scripts/
│   ├── setup.sh          # Environment setup
│   ├── run.sh            # Server startup
│   └── healthcheck.sh    # Health testing
├── vis/                  # Visualization backend
│   ├── __init__.py       # Package initialization
│   ├── render_tm_pr.py   # trimesh + pyrender rendering
│   ├── mesh_utils.py     # SMPL mesh utilities
│   └── skeleton_plot.py  # matplotlib skeleton plots
├── debug_vis/            # Debug visualization output
├── pretrained_models/    # Model checkpoints
└── configs/              # Model configurations
```

### Adding Features

1. **New Endpoints**: Add routes in `app.py`
2. **Model Features**: Extend `hybrik_runner.py`  
3. **Configuration**: Update `settings.yaml` schema
4. **Tests**: Add test cases in `scripts/healthcheck.sh`

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: hybrik**
   - Server works in Mock Mode without HybrIK - generates test data
   - For real inference: Install HybrIK dependencies manually due to version conflicts
   - Check `hybrik_runner.py` logs for "using mock model for testing"

2. **PyTorch3D installation failures (Mac M1/M2)**
   - ✅ **RESOLVED**: Using trimesh + pyrender instead
   - No PyTorch3D installation required

3. **matplotlib threading errors**
   - ✅ **RESOLVED**: Using Agg backend for server environments
   - Debug visualizations work without GUI

4. **Model checkpoint not found**
   - Server runs in Mock Mode by default
   - Download pretrained models to `pretrained_models/` for real inference
   - Update `settings.yaml` with correct paths

5. **MPS not available (Mac M2)**
   - Ensure PyTorch has MPS support: `torch.backends.mps.is_available()`
   - Update to latest PyTorch version with `--index-url https://download.pytorch.org/whl/cpu`

6. **Out of memory**
   - Reduce `max_image_resolution` in settings
   - Use CPU instead of GPU for large images

7. **OpenCV version conflicts**
   - Install compatible version: `pip install opencv-python>=4.5.0`
   - Use conda environment for better dependency management

### Performance Tips

- **Mock Mode**: ~300ms response time for testing
- **Real Inference**: Enable warmup for reduced cold start latency
- **GPU Acceleration**: Use MPS (Mac M1/M2) or CUDA for better inference speed
- **Image Resolution**: Adjust limits based on hardware capabilities
- **Debug Visualization**: Disable in production (`debug_vis: false`)
- **Memory Usage**: Monitor usage with high-resolution inputs
- **Conda Environment**: Recommended for Mac M1/M2 compatibility

### Debug Features

- **Mock Mode**: Test API without pretrained models
- **Visualization Output**: Enable `debug_vis: true` in settings
- **Test Script**: Run `python test_image_analysis.py` for comprehensive testing
- **Health Monitoring**: Use `/health` endpoint for service monitoring
- **Structured Logging**: JSON-formatted logs for production debugging

## Contributing

1. Follow existing code structure and patterns
2. Update configuration schema when adding features
3. Add appropriate error handling and logging
4. Test with `scripts/healthcheck.sh`
5. Update documentation

## License

This project follows the same license as HybrIK. See HybrIK repository for details.

## Citation

If you use this server in your research, please cite the original HybrIK paper:

```bibtex
@article{li2021hybrik,
  title={HybrIK: A Hybrid Analytical-Neural Inverse Kinematics Solution for 3D Human Pose and Shape Estimation},
  author={Li, Jiefeng and Xu, Chao and Chen, Zhicun and Bian, Siyuan and Yang, Lixin and Lu, Cewu},
  journal={CVPR},
  year={2021}
}
```