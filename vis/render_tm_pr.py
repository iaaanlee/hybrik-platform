"""
NOTE: This module replaces PyTorch3D with trimesh + pyrender for lightweight visualization.
PyTorch3D rendering is DEPRECATED/REMOVED (Mac M1/M2 install & GPU support issues).

Trimesh + pyrender based offscreen/onscreen rendering utilities for HybrIK server.
Provides mesh visualization and 3D joint skeleton rendering capabilities.
"""

import numpy as np
import trimesh
import pyrender
import os
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TriMeshPyRenderBackend:
    """
    Lightweight visualization backend using trimesh + pyrender.
    Replaces PyTorch3D functionality for Mac M2 compatibility.
    """
    
    def __init__(self, width: int = 800, height: int = 600):
        """
        Initialize rendering backend.
        
        Args:
            width: Output image width
            height: Output image height
        """
        self.width = width
        self.height = height
        self.scene = None
        self.renderer = None
        
    def setup_scene(self) -> pyrender.Scene:
        """Setup basic scene with camera and lighting."""
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])
        
        # Add camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=self.width/self.height)
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        scene.add(camera, pose=camera_pose)
        
        # Add directional light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=camera_pose)
        
        self.scene = scene
        return scene
    
    def render_mesh_offscreen(self, vertices: np.ndarray, faces: np.ndarray, 
                            output_path: str) -> bool:
        """
        Render mesh to image file (offscreen).
        
        Args:
            vertices: Mesh vertices (V x 3)
            faces: Mesh faces (F x 3)
            output_path: Output image path
            
        Returns:
            bool: Success status
        """
        try:
            # Create trimesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Convert to pyrender mesh
            py_mesh = pyrender.Mesh.from_trimesh(mesh)
            
            # Setup scene
            scene = self.setup_scene()
            scene.add(py_mesh)
            
            # Render offscreen
            renderer = pyrender.OffscreenRenderer(self.width, self.height)
            color, depth = renderer.render(scene)
            
            # Save image
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            import imageio
            imageio.imwrite(output_path, color)
            
            renderer.delete()
            logger.info(f"Rendered mesh to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to render mesh: {e}")
            return False
    
    def render_mesh_viewer(self, vertices: np.ndarray, faces: np.ndarray) -> bool:
        """
        Display mesh in interactive viewer (onscreen).
        
        Args:
            vertices: Mesh vertices (V x 3) 
            faces: Mesh faces (F x 3)
            
        Returns:
            bool: Success status
        """
        try:
            # Create trimesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Convert to pyrender mesh
            py_mesh = pyrender.Mesh.from_trimesh(mesh)
            
            # Setup scene
            scene = self.setup_scene()
            scene.add(py_mesh)
            
            # Show in viewer
            pyrender.Viewer(scene, use_raymond_lighting=True)
            return True
            
        except Exception as e:
            logger.error(f"Failed to show mesh viewer: {e}")
            return False
    
    def render_skeleton_offscreen(self, joints3d: np.ndarray, 
                                edges: Optional[list] = None,
                                output_path: str = "skeleton.png") -> bool:
        """
        Render 3D skeleton joints and connections to image file.
        
        Args:
            joints3d: Joint positions (J x 3)
            edges: List of joint connection pairs [(parent, child), ...]
            output_path: Output image path
            
        Returns:
            bool: Success status
        """
        try:
            scene = self.setup_scene()
            
            # Add joint spheres
            for i, joint in enumerate(joints3d):
                sphere = trimesh.creation.uv_sphere(radius=0.02)
                sphere.visual.face_colors = [255, 0, 0, 255]  # Red joints
                sphere.apply_translation(joint)
                py_sphere = pyrender.Mesh.from_trimesh(sphere)
                scene.add(py_sphere)
            
            # Add skeleton edges if provided
            if edges:
                for parent_idx, child_idx in edges:
                    if parent_idx < len(joints3d) and child_idx < len(joints3d):
                        # Create cylinder between joints
                        start = joints3d[parent_idx]
                        end = joints3d[child_idx]
                        
                        # Calculate cylinder parameters
                        direction = end - start
                        length = np.linalg.norm(direction)
                        
                        if length > 0.001:  # Avoid degenerate cases
                            cylinder = trimesh.creation.cylinder(radius=0.01, height=length)
                            
                            # Orient cylinder
                            z_axis = np.array([0, 0, 1])
                            direction_normalized = direction / length
                            
                            # Rotation matrix to align z-axis with direction
                            if np.allclose(direction_normalized, z_axis):
                                rotation_matrix = np.eye(3)
                            elif np.allclose(direction_normalized, -z_axis):
                                rotation_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
                            else:
                                v = np.cross(z_axis, direction_normalized)
                                s = np.linalg.norm(v)
                                c = np.dot(z_axis, direction_normalized)
                                
                                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                                rotation_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
                            
                            cylinder.apply_transform(np.eye(4))
                            cylinder.apply_transform(np.vstack([
                                np.hstack([rotation_matrix, np.zeros((3, 1))]),
                                [0, 0, 0, 1]
                            ]))
                            
                            # Position at midpoint
                            midpoint = (start + end) / 2
                            cylinder.apply_translation(midpoint)
                            
                            cylinder.visual.face_colors = [0, 255, 0, 255]  # Green bones
                            py_cylinder = pyrender.Mesh.from_trimesh(cylinder)
                            scene.add(py_cylinder)
            
            # Render offscreen
            renderer = pyrender.OffscreenRenderer(self.width, self.height)
            color, depth = renderer.render(scene)
            
            # Save image
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            import imageio
            imageio.imwrite(output_path, color)
            
            renderer.delete()
            logger.info(f"Rendered skeleton to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to render skeleton: {e}")
            return False


def get_default_skeleton_edges() -> list:
    """
    Get default human skeleton edge connections for 24-joint model.
    Returns list of (parent, child) joint index pairs.
    """
    # Basic human skeleton topology (adjust based on your joint ordering)
    edges = [
        # Spine
        (0, 1), (1, 2), (2, 3),  # pelvis -> spine -> neck -> head
        
        # Left arm  
        (2, 4), (4, 5), (5, 6),  # spine -> left shoulder -> elbow -> wrist
        
        # Right arm
        (2, 7), (7, 8), (8, 9),  # spine -> right shoulder -> elbow -> wrist
        
        # Left leg
        (0, 10), (10, 11), (11, 12),  # pelvis -> left hip -> knee -> ankle
        
        # Right leg  
        (0, 13), (13, 14), (14, 15),  # pelvis -> right hip -> knee -> ankle
    ]
    
    return edges