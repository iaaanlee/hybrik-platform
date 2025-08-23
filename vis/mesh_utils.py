"""
NOTE: This module replaces PyTorch3D with trimesh + pyrender for lightweight visualization.
PyTorch3D rendering is DEPRECATED/REMOVED (Mac M1/M2 install & GPU support issues).

Mesh utilities providing standard interface for vertices/faces processing.
Handles SMPL mesh data extraction and preprocessing for visualization.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MeshProcessor:
    """
    Standard mesh processing utilities for HybrIK SMPL output.
    Provides consistent interface for mesh data handling.
    """
    
    def __init__(self):
        """Initialize mesh processor."""
        pass
    
    def extract_smpl_mesh(self, smpl_output: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract vertices and faces from SMPL model output.
        
        Args:
            smpl_output: SMPL model output containing vertices and faces
            
        Returns:
            Tuple[vertices, faces]: Mesh vertices (V x 3) and faces (F x 3)
        """
        try:
            # Extract vertices
            if 'vertices' in smpl_output:
                vertices = smpl_output['vertices']
            elif 'verts' in smpl_output:
                vertices = smpl_output['verts'] 
            else:
                raise ValueError("No vertices found in SMPL output")
            
            # Extract faces
            if 'faces' in smpl_output:
                faces = smpl_output['faces']
            elif 'f' in smpl_output:
                faces = smpl_output['f']
            else:
                # Use default SMPL faces if not provided
                logger.warning("No faces found in SMPL output, using default SMPL topology")
                faces = self.get_default_smpl_faces()
            
            # Convert to numpy arrays
            vertices = np.array(vertices).reshape(-1, 3)
            faces = np.array(faces).reshape(-1, 3)
            
            # Validate dimensions
            if vertices.shape[1] != 3:
                raise ValueError(f"Invalid vertices shape: {vertices.shape}")
            if faces.shape[1] != 3:
                raise ValueError(f"Invalid faces shape: {faces.shape}")
            
            logger.info(f"Extracted mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
            return vertices, faces
            
        except Exception as e:
            logger.error(f"Failed to extract SMPL mesh: {e}")
            raise
    
    def normalize_mesh(self, vertices: np.ndarray, faces: np.ndarray, 
                      scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize mesh for visualization.
        
        Args:
            vertices: Mesh vertices (V x 3)
            faces: Mesh faces (F x 3) 
            scale: Scaling factor
            
        Returns:
            Tuple[vertices, faces]: Normalized mesh data
        """
        try:
            # Center mesh at origin
            centroid = np.mean(vertices, axis=0)
            vertices_centered = vertices - centroid
            
            # Scale mesh
            if scale != 1.0:
                vertices_centered *= scale
            
            logger.info(f"Normalized mesh: centered at origin, scaled by {scale}")
            return vertices_centered, faces
            
        except Exception as e:
            logger.error(f"Failed to normalize mesh: {e}")
            return vertices, faces
    
    def validate_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> bool:
        """
        Validate mesh data integrity.
        
        Args:
            vertices: Mesh vertices (V x 3)
            faces: Mesh faces (F x 3)
            
        Returns:
            bool: Validation result
        """
        try:
            # Check shapes
            if vertices.ndim != 2 or vertices.shape[1] != 3:
                logger.error(f"Invalid vertices shape: {vertices.shape}")
                return False
                
            if faces.ndim != 2 or faces.shape[1] != 3:
                logger.error(f"Invalid faces shape: {faces.shape}")
                return False
            
            # Check face indices
            max_vertex_idx = vertices.shape[0] - 1
            if np.max(faces) > max_vertex_idx:
                logger.error(f"Face indices exceed vertex count: max_idx={np.max(faces)}, vertices={vertices.shape[0]}")
                return False
                
            if np.min(faces) < 0:
                logger.error(f"Negative face indices found: min_idx={np.min(faces)}")
                return False
            
            # Check for NaN/Inf values
            if np.any(~np.isfinite(vertices)):
                logger.error("Non-finite values found in vertices")
                return False
            
            logger.info("Mesh validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Mesh validation failed: {e}")
            return False
    
    def get_default_smpl_faces(self) -> np.ndarray:
        """
        Get default SMPL mesh face topology.
        
        Returns:
            np.ndarray: Default SMPL faces (F x 3)
        """
        # This is a placeholder - in real implementation, you would load
        # the actual SMPL face topology from SMPL model files
        logger.warning("Using placeholder SMPL faces - replace with actual SMPL topology")
        
        # Create a simple placeholder mesh (tetrahedron)
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3], 
            [0, 3, 1],
            [1, 3, 2]
        ])
        
        return faces
    
    def extract_joints_from_mesh(self, vertices: np.ndarray, 
                                joint_regressor: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract 3D joint positions from mesh vertices.
        
        Args:
            vertices: Mesh vertices (V x 3)
            joint_regressor: Joint regressor matrix (J x V)
            
        Returns:
            np.ndarray: Joint positions (J x 3)
        """
        try:
            if joint_regressor is not None:
                # Use joint regressor to compute joint positions
                joints = np.dot(joint_regressor, vertices)
                logger.info(f"Extracted {joints.shape[0]} joints using regressor")
                return joints
            else:
                # Use predefined joint indices (placeholder)
                logger.warning("No joint regressor provided, using vertex subsampling")
                
                # Simple subsampling (not accurate, for demo only)
                num_joints = min(24, vertices.shape[0])
                joint_indices = np.linspace(0, vertices.shape[0]-1, num_joints, dtype=int)
                joints = vertices[joint_indices]
                
                return joints
                
        except Exception as e:
            logger.error(f"Failed to extract joints: {e}")
            raise


def create_debug_mesh(num_vertices: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a simple debug mesh for testing visualization pipeline.
    
    Args:
        num_vertices: Number of vertices to generate
        
    Returns:
        Tuple[vertices, faces]: Debug mesh data
    """
    # Create a simple sphere-like mesh
    theta = np.linspace(0, 2*np.pi, int(np.sqrt(num_vertices)))
    phi = np.linspace(0, np.pi, int(np.sqrt(num_vertices)))
    
    vertices = []
    for p in phi:
        for t in theta:
            x = np.sin(p) * np.cos(t)
            y = np.sin(p) * np.sin(t)  
            z = np.cos(p)
            vertices.append([x, y, z])
    
    vertices = np.array(vertices)
    
    # Create simple triangular faces
    faces = []
    n = int(np.sqrt(num_vertices))
    for i in range(n-1):
        for j in range(n-1):
            # Create two triangles per quad
            v1 = i * n + j
            v2 = i * n + (j + 1)
            v3 = (i + 1) * n + j
            v4 = (i + 1) * n + (j + 1)
            
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])
    
    faces = np.array(faces)
    
    return vertices, faces