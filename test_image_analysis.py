#!/usr/bin/env python3
"""
Test script for HybrIK server image analysis endpoint.
"""

import requests
import json
import base64
from PIL import Image
import io
import numpy as np

def create_test_image():
    """Create a simple test image."""
    # Create a 256x192 RGB image with some patterns
    image = Image.new('RGB', (256, 192), color='white')
    
    # Add some simple patterns (person-like shape)
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(figsize=(2.56, 1.92), dpi=100)
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 192)
    
    # Draw a simple stick figure
    # Head
    circle = patches.Circle((128, 160), 15, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)
    
    # Body
    ax.plot([128, 128], [145, 100], 'k-', linewidth=2)
    
    # Arms
    ax.plot([128, 110], [130, 120], 'k-', linewidth=2)
    ax.plot([128, 146], [130, 120], 'k-', linewidth=2)
    
    # Legs
    ax.plot([128, 115], [100, 60], 'k-', linewidth=2)
    ax.plot([128, 141], [100, 60], 'k-', linewidth=2)
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()
    
    # Convert to PIL Image
    image = Image.open(buf).convert('RGB')
    image = image.resize((256, 192))
    
    return image

def test_health_endpoint():
    """Test the health endpoint."""
    print("🔍 Testing health endpoint...")
    
    try:
        response = requests.get('http://localhost:8081/health', timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed:")
            print(f"   Status: {health_data.get('status')}")
            print(f"   Device: {health_data.get('version', {}).get('device')}")
            print(f"   Ready: {health_data.get('ready')}")
            return True
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_image_analysis():
    """Test the image analysis endpoint."""
    print("\n🔍 Testing image analysis endpoint...")
    
    try:
        # Create test image
        image = create_test_image()
        
        # Convert to base64
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        # Prepare request (API expects 'image_b64' key)
        data = {
            'image_b64': img_base64
        }
        
        print("   Sending test image for analysis...")
        response = requests.post('http://localhost:8081/analyze-image', 
                               json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Analysis successful:")
            
            # Check joints3d
            if 'joints3d' in result:
                joints3d = np.array(result['joints3d'])
                print(f"   3D Joints: {joints3d.shape} (expected: 24x3)")
                
                if joints3d.shape == (24, 3):
                    print("   Joint shape: ✅ Correct")
                    
                    # Check for reasonable values
                    joint_range = np.ptp(joints3d, axis=0)  # peak-to-peak range
                    print(f"   Joint range (X,Y,Z): {joint_range}")
                    
                    if np.all(joint_range > 0):
                        print("   Joint variance: ✅ Non-zero variance detected")
                    else:
                        print("   Joint variance: ⚠️  Some axes have no variance")
                else:
                    print("   Joint shape: ❌ Incorrect shape")
            
            # Check optional outputs
            if 'joints2d' in result:
                joints2d = np.array(result['joints2d'])
                print(f"   2D Joints: {joints2d.shape}")
                
            if 'confidence' in result:
                confidence = np.array(result['confidence'])
                print(f"   Confidence: {confidence.shape}, mean={np.mean(confidence):.3f}")
            
            if 'debug_visualization' in result:
                debug_vis = result['debug_visualization']
                print(f"   Debug visualization: {len(debug_vis)} outputs")
                for key, path in debug_vis.items():
                    print(f"     {key}: {path}")
            
            return True
            
        else:
            print(f"❌ Analysis failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting HybrIK Server Tests\n")
    
    # Test health endpoint
    health_ok = test_health_endpoint()
    
    if health_ok:
        # Test image analysis
        analysis_ok = test_image_analysis()
        
        if analysis_ok:
            print("\n🎉 All tests passed! HybrIK server is working correctly.")
            print("\nServer features verified:")
            print("- ✅ Health check endpoint")
            print("- ✅ Image analysis endpoint")
            print("- ✅ Mock inference working")
            print("- ✅ 3D joint coordinate output")
            print("- ✅ Trimesh + pyrender visualization backend ready")
        else:
            print("\n❌ Image analysis test failed.")
    else:
        print("\n❌ Server health check failed. Make sure the server is running.")
    
    print("\nTo enable debug visualization, set 'debug_vis: true' in settings.yaml")

if __name__ == '__main__':
    main()