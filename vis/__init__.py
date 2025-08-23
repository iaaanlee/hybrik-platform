"""
Visualization utilities for HybrIK server.

NOTE: This package replaces PyTorch3D with trimesh + pyrender for lightweight visualization.
PyTorch3D rendering is DEPRECATED/REMOVED (Mac M1/M2 install & GPU support issues).

Modules:
- render_tm_pr: Trimesh + pyrender based mesh and skeleton rendering
- mesh_utils: Mesh processing utilities for SMPL data
- skeleton_plot: Matplotlib based skeleton visualization
"""

from .render_tm_pr import TriMeshPyRenderBackend, get_default_skeleton_edges
from .mesh_utils import MeshProcessor, create_debug_mesh
from .skeleton_plot import SkeletonPlotter, get_default_skeleton_edges_24

__all__ = [
    'TriMeshPyRenderBackend',
    'MeshProcessor', 
    'SkeletonPlotter',
    'get_default_skeleton_edges',
    'get_default_skeleton_edges_24',
    'create_debug_mesh'
]