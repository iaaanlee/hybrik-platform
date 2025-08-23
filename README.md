# HybrIK Server

Independent REST API server for 3D human pose estimation using HybrIK.

## Overview

HybrIK Server provides a REST API wrapper around the HybrIK model for 3D human pose and shape estimation. It offers standardized JSON APIs for single image analysis with configurable output formats.

## Features

- **3D Joint Estimation**: Extract 24-joint 3D pose coordinates from single images
- **Flexible Output**: Support for mm/normalized units and different coordinate frames
- **REST API**: Simple HTTP endpoints with JSON input/output
- **Multi-Platform**: CPU, CUDA, and MPS (Apple Silicon) support
- **Configurable**: YAML-based configuration for easy customization
- **Production Ready**: Docker support, health checks, and logging

## Quick Start

### 1. Setup Environment

```bash
# Clone and setup
git clone <this-repo>
cd hybrik-server

# Setup Python environment and dependencies
./scripts/setup.sh
```

### 2. Install HybrIK

```bash
# Clone HybrIK repository (adjacent to hybrik-server)
cd ..
git clone https://github.com/jeffffffli/HybrIK.git
cd HybrIK

# Install HybrIK
pip install -e .
```

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
curl http://localhost:8081/health
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
curl -X POST -F "file=@image.jpg" http://localhost:8081/analyze-image
```

**JSON Base64:**
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "base64_encoded_image"}' \
  http://localhost:8081/analyze-image
```

**Response:**
```json
{
  "joints3d": [[x, y, z], ...],
  "joints2d": [[u, v], ...],
  "confidence": [0.9, 0.8, ...],
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
docker run -p 8081:8081 \
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
├── Dockerfile            # Container definition
├── scripts/
│   ├── setup.sh          # Environment setup
│   ├── run.sh            # Server startup
│   └── healthcheck.sh    # Health testing
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
   - Ensure HybrIK is installed: `pip install -e ../HybrIK`

2. **Model checkpoint not found**
   - Download pretrained models to `pretrained_models/`
   - Update `settings.yaml` with correct paths

3. **MPS not available (Mac M2)**
   - Ensure PyTorch has MPS support: `torch.backends.mps.is_available()`
   - Update to latest PyTorch version

4. **Out of memory**
   - Reduce `max_image_resolution` in settings
   - Use CPU instead of GPU for large images

### Performance Tips

- Enable warmup for reduced cold start latency
- Use GPU/MPS for better inference speed
- Adjust image resolution limits based on hardware
- Monitor memory usage with high-resolution inputs

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