#!/usr/bin/env python3
"""
HybrIK Runner - Core inference wrapper
Handles model loading, preprocessing, inference, and postprocessing

NOTE: Visualization backend has been replaced from PyTorch3D to trimesh + pyrender
for Mac M1/M2 compatibility. PyTorch3D rendering is DEPRECATED/REMOVED.
"""

import os
import sys
import logging
import time
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

class HybrIKRunner:
    """HybrIK inference wrapper"""
    
    def __init__(self, config):
        self.config = config
        self.device = self._setup_device()
        self.model = None
        self.transform = None
        
        # Initialize model
        self._load_model()
        self._setup_transforms()
        
    def _setup_device(self):
        """Setup compute device"""
        device_config = self.config['model']['device'].lower()
        
        if device_config == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            
        elif device_config == 'mps' and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using Apple Metal Performance Shaders (MPS)")
            
        else:
            device = torch.device('cpu')
            logger.info("Using CPU device")
            
        return device
    
    def _load_model(self):
        """Load HybrIK model"""
        try:
            logger.info("Loading HybrIK model...")
            
            # Check if HybrIK is installed
            try:
                # Add HybrIK to path
                hybrik_path = '/Users/kihoonlee/self-workspace/skeleton_app/hybrik-platform/HybrIK'
                if hybrik_path not in sys.path:
                    sys.path.append(hybrik_path)
                
                from hybrik.models import builder
                from hybrik.utils.config import update_config
                from yacs.config import CfgNode as CN
            except ImportError as e:
                logger.error("HybrIK 라이브러리를 찾을 수 없습니다. 실제 분석을 위해 다음이 필요합니다:")
                logger.error("1. HybrIK 라이브러리 설치")
                logger.error("2. 사전 훈련된 모델 파일: pretrained_models/hybrik_hrnet.pth")
                logger.error("3. 설정 파일: configs/hybrik_hrnet.yaml")
                raise RuntimeError("HybrIK 모델을 로드할 수 없습니다. Mock 데이터 대신 실제 모델이 필요합니다.")
            
            # Load config
            cfg_path = self.config['model']['hybrik_cfg']
            if not os.path.exists(cfg_path):
                logger.error(f"필수 설정 파일이 없습니다: {cfg_path}")
                raise RuntimeError(f"HybrIK 설정 파일을 찾을 수 없습니다: {cfg_path}")
            
            cfg = update_config(cfg_path)
            
            # Build model
            self.model = builder.build_sppe(cfg.MODEL)
            
            # Load checkpoint (필수)
            ckpt_path = self.config['model']['hybrik_ckpt']
            if not os.path.exists(ckpt_path):
                logger.error(f"필수 모델 파일이 없습니다: {ckpt_path}")
                raise RuntimeError(f"HybrIK 사전 훈련된 모델을 찾을 수 없습니다: {ckpt_path}")
            
            logger.info(f"Loading checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
                
            logger.info("Checkpoint loaded successfully")
            
            # Move model to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("HybrIK model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load HybrIK model: {e}")
            raise
    
    def _get_default_config(self):
        """Get default HybrIK configuration"""
        from yacs.config import CfgNode as CN
        
        cfg = CN()
        cfg.MODEL = CN()
        cfg.MODEL.NAME = 'hybrik'
        cfg.MODEL.BACKBONE = 'hrnet'
        cfg.MODEL.NUM_JOINTS = 24
        cfg.MODEL.PRETRAINED = ''
        
        return cfg
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.transform = transforms.Compose([
            transforms.Resize((256, 192)),  # HybrIK standard input size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def warmup(self):
        """Perform warmup inference to reduce cold start latency"""
        try:
            logger.info("Starting warmup...")
            iterations = self.config['performance']['warmup_iterations']
            
            # Create dummy image
            dummy_image = Image.new('RGB', (256, 192), color='white')
            
            for i in range(iterations):
                logger.debug(f"Warmup iteration {i+1}/{iterations}")
                _ = self.analyze_image(dummy_image, log_performance=False)
            
            logger.info(f"Warmup completed ({iterations} iterations)")
            
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
    
    def analyze_image(self, image, log_performance=True):
        """
        Analyze image for 3D human pose estimation
        
        Args:
            image: PIL Image object
            log_performance: Whether to log performance metrics
            
        Returns:
            dict: Analysis results with 3D joints and optional 2D joints
        """
        start_time = time.time()
        
        try:
            # Check if model is loaded
            if self.model is None:
                raise RuntimeError("HybrIK 모델이 로드되지 않았습니다. 실제 모델 파일이 필요합니다.")
            # Store input image for 2D overlay visualization
            self._current_input_image = image
            
            # Preprocessing
            preprocess_start = time.time()
            input_tensor = self._preprocess_image(image)
            preprocess_time = time.time() - preprocess_start
            
            # Inference
            inference_start = time.time()
            with torch.no_grad():
                # Real HybrIK model inference
                results = self.model(input_tensor)
            inference_time = time.time() - inference_start
            
            # Postprocessing
            postprocess_start = time.time()
            output = self._postprocess_results(results)
            postprocess_time = time.time() - postprocess_start
            
            total_time = time.time() - start_time
            
            if log_performance and self.config['logging']['log_performance']:
                logger.info(f"Performance - Total: {total_time:.3f}s, "
                           f"Preprocess: {preprocess_time:.3f}s, "
                           f"Inference: {inference_time:.3f}s, "
                           f"Postprocess: {postprocess_time:.3f}s")
            
            return output
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def _preprocess_image(self, image):
        """Preprocess image for HybrIK input"""
        # Apply transforms
        input_tensor = self.transform(image)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.to(self.device)
        
        return input_tensor
    
    
    def _postprocess_results(self, results):
        """Postprocess model outputs to standard format"""
        # Convert tensors to numpy - use actual HybrIK output keys
        # pred_xyz_jts_24_struct is flattened (batch_size, 72), reshape to (batch_size, 24, 3)
        if 'pred_xyz_jts_24_struct' in results:
            joints_3d_flat = results['pred_xyz_jts_24_struct'][0].cpu().numpy()  # Remove batch dimension
            joints_3d = joints_3d_flat.reshape(24, 3)  # Reshape from (72,) to (24, 3)
        else:
            # Fallback to pred_xyz_jts_17 if available
            joints_3d_flat = results['pred_xyz_jts_17'][0].cpu().numpy() if 'pred_xyz_jts_17' in results else None
            joints_3d = joints_3d_flat.reshape(-1, 3) if joints_3d_flat is not None else None
        
        # HybrIK doesn't output joints_2d directly - we'll generate it in overlay
        joints_2d = None
        
        # Use scores (1 - sigma) as confidence if available
        confidence = results['scores'][0].cpu().numpy() if 'scores' in results else None
        
        # Apply coordinate system conversion
        if joints_3d is not None and self.config['output']['coord_frame'] == 'root-relative':
            # Make root-relative (subtract root joint position)
            if len(joints_3d) > 0:
                root_joint = joints_3d[0]  # Assume first joint is root
                joints_3d = joints_3d - root_joint
        
        # Apply unit conversion
        if joints_3d is not None and self.config['output']['units'] == 'mm':
            joints_3d = joints_3d * 1000  # Convert to millimeters
        
        # Clean invalid values
        if joints_3d is not None:
            joints_3d = np.nan_to_num(joints_3d, nan=0.0, posinf=0.0, neginf=0.0)
        if joints_2d is not None:
            joints_2d = np.nan_to_num(joints_2d, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Build output
        output = {
            'joints3d': joints_3d.tolist() if joints_3d is not None else []
        }
        
        # Add optional outputs based on configuration
        if self.config['output']['return_2d_joints'] and joints_2d is not None:
            output['joints2d'] = joints_2d.tolist()
            
        if confidence is not None:
            output['confidence'] = confidence.tolist()
        
        # TODO: Add SMPL parameters if requested
        if self.config['output']['return_smpl_params']:
            output['smpl'] = None  # Placeholder
            
        # TODO: Add mesh if requested  
        if self.config['output']['return_mesh']:
            output['mesh'] = None  # Placeholder
        
        # Add debug visualization if enabled
        if self.config.get('visualization', {}).get('debug_vis', False) and joints_3d is not None:
            self._generate_debug_visualization(joints_3d, output)
            
        # Add 2D overlay if enabled and have input image
        if self.config.get('visualization', {}).get('enable_2d_overlay', False):
            if hasattr(self, '_current_input_image') and joints_3d is not None:
                # For HybrIK, we generate 2D overlay from 3D joints
                self._generate_2d_overlay(self._current_input_image, joints_3d, output)
        
        return output
    
    def get_joint_names(self):
        """Get standard joint names in order"""
        # TODO: Return actual HybrIK joint names
        return [
            'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
            'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
            'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
        ]
    
    def _generate_debug_visualization(self, joints_3d, output):
        """
        Generate debug visualization using trimesh + pyrender backend.
        
        Args:
            joints_3d: Joint positions (J x 3)
            output: Output dictionary to add visualization paths
        """
        try:
            from vis.skeleton_plot import SkeletonPlotter, get_default_skeleton_edges_24
            from vis.render_tm_pr import TriMeshPyRenderBackend, get_default_skeleton_edges
            import os
            from datetime import datetime
            
            # Get output directory
            vis_output_dir = self.config.get('visualization', {}).get('vis_output_dir', './debug_vis')
            os.makedirs(vis_output_dir, exist_ok=True)
            
            # Generate timestamp for unique filenames
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 1. Generate matplotlib skeleton plot
            plotter = SkeletonPlotter()
            skeleton_edges = get_default_skeleton_edges_24()
            skeleton_plot_path = os.path.join(vis_output_dir, f'skeleton_plot_{timestamp}.png')
            
            success_plot = plotter.plot_skeleton_3d(
                joints_3d, 
                edges=skeleton_edges,
                output_path=skeleton_plot_path,
                title=f"3D Skeleton - {timestamp}"
            )
            
            if success_plot:
                output['debug_visualization'] = output.get('debug_visualization', {})
                output['debug_visualization']['skeleton_plot'] = skeleton_plot_path
            
            # 2. Generate trimesh+pyrender skeleton rendering (if available)
            try:
                renderer = TriMeshPyRenderBackend(width=800, height=600)
                render_edges = get_default_skeleton_edges()
                skeleton_render_path = os.path.join(vis_output_dir, f'skeleton_render_{timestamp}.png')
                
                success_render = renderer.render_skeleton_offscreen(
                    joints_3d,
                    edges=render_edges, 
                    output_path=skeleton_render_path
                )
                
                if success_render:
                    output['debug_visualization']['skeleton_render'] = skeleton_render_path
                    
            except Exception as e:
                logger.warning(f"3D skeleton rendering failed: {e}")
            
            logger.info(f"Debug visualization generated in {vis_output_dir}")
            
        except Exception as e:
            logger.warning(f"Debug visualization failed: {e}")
            # Don't fail the main inference due to visualization issues
    
    def _generate_2d_overlay(self, input_image, joints_3d, output):
        """
        Generate 2D overlay image with keypoints and skeleton
        
        Args:
            input_image: PIL Image object (original input)
            joints_3d: 3D joint positions (24 x 3) - will be projected to 2D
            output: Output dictionary to add overlay data
        """
        try:
            import cv2
            import numpy as np
            import base64
            from io import BytesIO
            
            # Convert PIL to OpenCV format
            img_array = np.array(input_image)
            if img_array.ndim == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Create a copy for drawing
            overlay_img = img_array.copy()
            
            # Project 3D joints to 2D by taking X and Y coordinates (drop Z)
            # HybrIK outputs 3D coordinates in camera space
            joints_2d = joints_3d[:, :2]  # Take only X and Y coordinates
            
            # Scale joints_2d from HybrIK input size (256x192) to original image size
            original_height, original_width = overlay_img.shape[:2]
            hybrik_height, hybrik_width = 256, 192  # HybrIK standard input size
            
            scale_x = original_width / hybrik_width
            scale_y = original_height / hybrik_height
            
            # Scale joint coordinates to original image size
            scaled_joints_2d = joints_2d.copy()
            scaled_joints_2d[:, 0] *= scale_x  # x coordinates
            scaled_joints_2d[:, 1] *= scale_y  # y coordinates
            
            logger.info(f"Scaling joints from {hybrik_width}x{hybrik_height} to {original_width}x{original_height}")
            logger.info(f"Scale factors: x={scale_x:.2f}, y={scale_y:.2f}")
            logger.info(f"First joint original: {joints_2d[0]}, scaled: {scaled_joints_2d[0]}")
            
            # HybrIK 24-joint skeleton connections (simplified)
            skeleton_connections = [
                # Spine connections
                (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),  # pelvis -> head
                # Left arm
                (12, 13), (13, 16), (16, 18), (18, 20), (20, 22),
                # Right arm  
                (12, 14), (14, 17), (17, 19), (19, 21), (21, 23),
                # Left leg
                (0, 1), (1, 4), (4, 7), (7, 10),
                # Right leg
                (0, 2), (2, 5), (5, 8), (8, 11)
            ]
            
            # Draw skeleton connections using scaled coordinates
            for connection in skeleton_connections:
                if connection[0] < len(scaled_joints_2d) and connection[1] < len(scaled_joints_2d):
                    pt1 = tuple(map(int, scaled_joints_2d[connection[0]]))
                    pt2 = tuple(map(int, scaled_joints_2d[connection[1]]))
                    
                    # Only draw if both points are valid (within image bounds)
                    if (0 <= pt1[0] < overlay_img.shape[1] and 0 <= pt1[1] < overlay_img.shape[0] and
                        0 <= pt2[0] < overlay_img.shape[1] and 0 <= pt2[1] < overlay_img.shape[0]):
                        cv2.line(overlay_img, pt1, pt2, (0, 255, 0), 3)  # Green lines, thicker for visibility
            
            # Draw keypoints using scaled coordinates
            for i, joint in enumerate(scaled_joints_2d):
                pt = tuple(map(int, joint))
                if 0 <= pt[0] < overlay_img.shape[1] and 0 <= pt[1] < overlay_img.shape[0]:
                    # Different colors for different joint types
                    if i == 0:  # pelvis
                        color = (255, 0, 0)  # Red
                    elif i == 15:  # head
                        color = (0, 0, 255)  # Blue
                    else:
                        color = (255, 255, 0)  # Yellow
                    
                    # Larger circles for better visibility
                    cv2.circle(overlay_img, pt, 6, color, -1)
                    cv2.circle(overlay_img, pt, 7, (0, 0, 0), 1)  # Black border
            
            # Convert back to RGB and encode to base64
            if overlay_img.ndim == 3:
                overlay_img_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
            else:
                overlay_img_rgb = overlay_img
                
            # Convert to PIL and then to base64
            from PIL import Image
            pil_img = Image.fromarray(overlay_img_rgb)
            buffer = BytesIO()
            pil_img.save(buffer, format='JPEG', quality=90)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Add to output
            if 'debug_visualization' not in output:
                output['debug_visualization'] = {}
            output['debug_visualization']['overlay_2d'] = img_base64
            
            logger.info("2D overlay image generated successfully")
            
        except Exception as e:
            logger.warning(f"2D overlay generation failed: {e}")
            # Don't fail the main inference due to visualization issues