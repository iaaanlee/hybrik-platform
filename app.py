#!/usr/bin/env python3
"""
HybrIK Server - REST API Wrapper
Independent 3D human pose estimation service using HybrIK
"""

import os
import sys
import logging
import yaml
import time
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import numpy as np

from hybrik_runner import HybrIKRunner

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
hybrik_runner = None
config = None

def load_config():
    """Load configuration from settings.yaml"""
    global config
    config_path = os.path.join(os.path.dirname(__file__), 'settings.yaml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set logging level
    log_level = getattr(logging, config['logging']['level'].upper())
    logging.getLogger().setLevel(log_level)
    logger.info(f"Configuration loaded from {config_path}")
    return config

def initialize_hybrik():
    """Initialize HybrIK runner"""
    global hybrik_runner
    try:
        logger.info("Initializing HybrIK runner...")
        hybrik_runner = HybrIKRunner(config)
        logger.info("HybrIK runner initialized successfully")
        
        # Perform warmup if enabled
        if config['performance']['enable_warmup']:
            logger.info("Performing warmup...")
            hybrik_runner.warmup()
            logger.info("Warmup completed")
            
    except Exception as e:
        logger.error(f"Failed to initialize HybrIK: {e}")
        logger.error(traceback.format_exc())
        raise

def validate_image_input(file_data, max_size_mb=10, max_resolution=2048):
    """Validate image input"""
    try:
        # Check file size
        if len(file_data) > max_size_mb * 1024 * 1024:
            raise ValueError(f"File too large. Max size: {max_size_mb}MB")
        
        # Load and validate image
        image = Image.open(BytesIO(file_data))
        
        # Check resolution
        width, height = image.size
        if max(width, height) > max_resolution:
            raise ValueError(f"Image resolution too high. Max: {max_resolution}px")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    except Exception as e:
        raise ValueError(f"Invalid image: {e}")

@app.route('/health', methods=['GET', 'POST'])
def health_check():
    """Health check endpoint"""
    try:
        status = {
            'status': 'ok',
            'timestamp': time.time(),
            'version': {
                'app': '1.0.0',
                'hybrik': 'latest',
                'device': config['model']['device'] if config else 'unknown'
            },
            'ready': hybrik_runner is not None
        }
        
        return jsonify(status), 200
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    """Analyze single image for 3D pose estimation"""
    start_time = time.time()
    
    try:
        if hybrik_runner is None:
            raise RuntimeError("HybrIK runner not initialized")
        
        # Parse input - support both multipart and JSON base64
        image_data = None
        
        if 'file' in request.files:
            # Multipart file upload
            file = request.files['file']
            if file.filename == '':
                raise ValueError("No file selected")
            image_data = file.read()
            
        elif request.is_json and 'image_b64' in request.get_json():
            # Base64 JSON input
            json_data = request.get_json()
            try:
                image_data = base64.b64decode(json_data['image_b64'])
            except Exception as e:
                raise ValueError(f"Invalid base64 image: {e}")
        else:
            raise ValueError("No image data provided. Use 'file' parameter or 'image_b64' in JSON")
        
        # Validate input
        image = validate_image_input(
            image_data,
            max_size_mb=config['limits']['max_file_mb'],
            max_resolution=config['limits']['max_image_resolution']
        )
        
        # Run HybrIK inference
        result = hybrik_runner.analyze_image(image)
        
        # Add metadata
        processing_time = time.time() - start_time
        result['meta'] = {
            'units': config['output']['units'],
            'coord_frame': config['output']['coord_frame'],
            'joint_order': 'hybrik_24_joints',  # TODO: Document joint ordering
            'latency_ms': round(processing_time * 1000, 2),
            'version': {
                'app': '1.0.0',
                'hybrik': 'latest',
                'model': os.path.basename(config['model']['hybrik_ckpt'])
            }
        }
        
        # Log request if enabled
        if config['logging']['log_requests']:
            logger.info(f"Image analyzed - Size: {len(image_data)} bytes, "
                       f"Time: {processing_time:.3f}s")
        
        return jsonify(result), 200
        
    except ValueError as e:
        # Client error (4xx)
        logger.warning(f"Client error: {e}")
        return jsonify({
            'error': 'validation_error',
            'message': str(e)
        }), 400
        
    except Exception as e:
        # Server error (5xx)
        logger.error(f"Server error during image analysis: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'internal_error',
            'message': 'An internal server error occurred'
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'error': 'file_too_large',
        'message': f'File size exceeds maximum allowed ({config["limits"]["max_file_mb"]}MB)'
    }), 413

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'not_found',
        'message': 'Endpoint not found'
    }), 404

def main():
    """Main entry point"""
    try:
        logger.info("Starting HybrIK Server...")
        
        # Load configuration
        load_config()
        
        # Initialize HybrIK
        initialize_hybrik()
        
        # Start server
        host = config['server']['host']
        port = config['server']['port']
        
        logger.info(f"Server ready at http://{host}:{port}")
        logger.info("Available endpoints:")
        logger.info("  GET/POST /health - Health check")
        logger.info("  POST /analyze-image - 3D pose estimation")
        
        app.run(
            host=host,
            port=port,
            debug=False,
            threaded=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()