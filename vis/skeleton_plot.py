"""
NOTE: This module replaces PyTorch3D with trimesh + pyrender for lightweight visualization.
PyTorch3D rendering is DEPRECATED/REMOVED (Mac M1/M2 install & GPU support issues).

3D joint skeleton plotting utilities using matplotlib as lightweight alternative.
Provides 2D projection visualization for debugging joint positions.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server environments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List, Tuple, Dict
import os
import logging

logger = logging.getLogger(__name__)


class SkeletonPlotter:
    """
    Lightweight 3D skeleton visualization using matplotlib.
    Alternative to heavy mesh rendering for joint position debugging.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize skeleton plotter.
        
        Args:
            figsize: Figure size (width, height)
        """
        self.figsize = figsize
        self.joint_colors = self._get_default_joint_colors()
        self.bone_color = 'green'
        self.joint_size = 50
        self.bone_width = 2
    
    def plot_skeleton_3d(self, joints3d: np.ndarray, 
                        edges: Optional[List[Tuple[int, int]]] = None,
                        output_path: Optional[str] = None,
                        title: str = "3D Skeleton",
                        show_labels: bool = False) -> bool:
        """
        Plot 3D skeleton with joints and connections.
        
        Args:
            joints3d: Joint positions (J x 3)
            edges: List of (parent, child) joint connections
            output_path: Save path (if None, display interactively)
            title: Plot title
            show_labels: Whether to show joint index labels
            
        Returns:
            bool: Success status
        """
        try:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot joints
            x, y, z = joints3d[:, 0], joints3d[:, 1], joints3d[:, 2]
            scatter = ax.scatter(x, y, z, c=range(len(joints3d)), 
                               cmap='viridis', s=self.joint_size, alpha=0.8)
            
            # Plot skeleton edges if provided
            if edges:
                for parent_idx, child_idx in edges:
                    if parent_idx < len(joints3d) and child_idx < len(joints3d):
                        parent_pos = joints3d[parent_idx]
                        child_pos = joints3d[child_idx]
                        
                        ax.plot([parent_pos[0], child_pos[0]],
                               [parent_pos[1], child_pos[1]], 
                               [parent_pos[2], child_pos[2]],
                               color=self.bone_color, linewidth=self.bone_width, alpha=0.7)
            
            # Add joint labels if requested
            if show_labels:
                for i, (x_pos, y_pos, z_pos) in enumerate(joints3d):
                    ax.text(x_pos, y_pos, z_pos, f'{i}', fontsize=8)
            
            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y') 
            ax.set_zlabel('Z')
            ax.set_title(title)
            
            # Add colorbar for joint indices
            plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
            
            # Set equal aspect ratio
            max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
            mid_x = (x.max()+x.min()) * 0.5
            mid_y = (y.max()+y.min()) * 0.5  
            mid_z = (z.max()+z.min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Save or show
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved 3D skeleton plot to {output_path}")
            else:
                plt.show()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to plot 3D skeleton: {e}")
            return False
    
    def plot_skeleton_2d_projections(self, joints3d: np.ndarray,
                                   edges: Optional[List[Tuple[int, int]]] = None,
                                   output_path: Optional[str] = None,
                                   title: str = "2D Skeleton Projections") -> bool:
        """
        Plot 2D projections of 3D skeleton (XY, XZ, YZ views).
        
        Args:
            joints3d: Joint positions (J x 3)
            edges: List of (parent, child) joint connections
            output_path: Save path (if None, display interactively)
            title: Plot title
            
        Returns:
            bool: Success status
        """
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            projections = [
                (0, 1, 'XY'),  # X-Y projection
                (0, 2, 'XZ'),  # X-Z projection  
                (1, 2, 'YZ')   # Y-Z projection
            ]
            
            for ax, (dim1, dim2, view_name) in zip(axes, projections):
                # Plot joints
                x, y = joints3d[:, dim1], joints3d[:, dim2]
                scatter = ax.scatter(x, y, c=range(len(joints3d)), 
                                   cmap='viridis', s=self.joint_size, alpha=0.8)
                
                # Plot skeleton edges if provided
                if edges:
                    for parent_idx, child_idx in edges:
                        if parent_idx < len(joints3d) and child_idx < len(joints3d):
                            parent_pos = joints3d[parent_idx]
                            child_pos = joints3d[child_idx]
                            
                            ax.plot([parent_pos[dim1], child_pos[dim1]],
                                   [parent_pos[dim2], child_pos[dim2]],
                                   color=self.bone_color, linewidth=self.bone_width, alpha=0.7)
                
                # Set labels and title
                ax.set_xlabel(['X', 'Y', 'Z'][dim1])
                ax.set_ylabel(['X', 'Y', 'Z'][dim2])
                ax.set_title(f'{view_name} View')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
            
            plt.suptitle(title)
            plt.tight_layout()
            
            # Save or show
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved 2D projection plots to {output_path}")
            else:
                plt.show()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to plot 2D projections: {e}")
            return False
    
    def plot_joint_comparison(self, joints3d_1: np.ndarray, joints3d_2: np.ndarray,
                            labels: Tuple[str, str] = ("Prediction", "Ground Truth"),
                            output_path: Optional[str] = None) -> bool:
        """
        Plot comparison between two sets of joint positions.
        
        Args:
            joints3d_1: First set of joint positions (J x 3)
            joints3d_2: Second set of joint positions (J x 3) 
            labels: Labels for the two sets
            output_path: Save path (if None, display interactively)
            
        Returns:
            bool: Success status
        """
        try:
            fig = plt.figure(figsize=(12, 5))
            
            # 3D comparison plot
            ax1 = fig.add_subplot(121, projection='3d')
            
            # Plot first set
            x1, y1, z1 = joints3d_1[:, 0], joints3d_1[:, 1], joints3d_1[:, 2]
            ax1.scatter(x1, y1, z1, c='red', s=self.joint_size, alpha=0.8, label=labels[0])
            
            # Plot second set
            x2, y2, z2 = joints3d_2[:, 0], joints3d_2[:, 1], joints3d_2[:, 2]
            ax1.scatter(x2, y2, z2, c='blue', s=self.joint_size, alpha=0.8, label=labels[1])
            
            # Draw connection lines between corresponding joints
            for i in range(min(len(joints3d_1), len(joints3d_2))):
                ax1.plot([joints3d_1[i, 0], joints3d_2[i, 0]],
                        [joints3d_1[i, 1], joints3d_2[i, 1]], 
                        [joints3d_1[i, 2], joints3d_2[i, 2]],
                        'gray', alpha=0.3, linewidth=1)
            
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.set_title('3D Joint Comparison')
            ax1.legend()
            
            # Error plot
            ax2 = fig.add_subplot(122)
            
            # Calculate per-joint errors
            errors = np.linalg.norm(joints3d_1 - joints3d_2, axis=1)
            joint_indices = range(len(errors))
            
            bars = ax2.bar(joint_indices, errors, alpha=0.7)
            ax2.set_xlabel('Joint Index')
            ax2.set_ylabel('L2 Error')
            ax2.set_title('Per-Joint L2 Error')
            ax2.grid(True, alpha=0.3)
            
            # Add error statistics
            mean_error = np.mean(errors)
            ax2.axhline(mean_error, color='red', linestyle='--', 
                       label=f'Mean: {mean_error:.3f}')
            ax2.legend()
            
            plt.tight_layout()
            
            # Save or show
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved joint comparison plot to {output_path}")
            else:
                plt.show()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to plot joint comparison: {e}")
            return False
    
    def _get_default_joint_colors(self) -> Dict[str, str]:
        """Get default color scheme for different joint types."""
        return {
            'head': 'red',
            'torso': 'blue', 
            'left_arm': 'green',
            'right_arm': 'orange',
            'left_leg': 'purple',
            'right_leg': 'brown'
        }


def get_default_skeleton_edges_24() -> List[Tuple[int, int]]:
    """
    Get default skeleton edge connections for 24-joint human model.
    
    Returns:
        List of (parent, child) joint index pairs
    """
    # Standard 24-joint skeleton topology
    # Adjust indices based on your specific joint ordering
    edges = [
        # Spine chain
        (0, 1),   # pelvis -> spine1
        (1, 2),   # spine1 -> spine2  
        (2, 3),   # spine2 -> neck
        (3, 4),   # neck -> head
        
        # Left arm chain
        (2, 5),   # spine2 -> left_shoulder
        (5, 6),   # left_shoulder -> left_elbow
        (6, 7),   # left_elbow -> left_wrist
        
        # Right arm chain  
        (2, 8),   # spine2 -> right_shoulder
        (8, 9),   # right_shoulder -> right_elbow
        (9, 10),  # right_elbow -> right_wrist
        
        # Left leg chain
        (0, 11),  # pelvis -> left_hip
        (11, 12), # left_hip -> left_knee
        (12, 13), # left_knee -> left_ankle
        (13, 14), # left_ankle -> left_foot
        
        # Right leg chain
        (0, 15),  # pelvis -> right_hip
        (15, 16), # right_hip -> right_knee  
        (16, 17), # right_knee -> right_ankle
        (17, 18), # right_ankle -> right_foot
        
        # Additional joints (hands, feet details)
        (7, 19),  # left_wrist -> left_hand
        (10, 20), # right_wrist -> right_hand
        (14, 21), # left_foot -> left_toe
        (18, 22), # right_foot -> right_toe
        (4, 23),  # head -> head_top
    ]
    
    return edges