# SPEED+ Tango Satellite Keypoint Visualizer

A Streamlit-based interactive tool to visualize and understand the SPEED+ Tango satellite keypoints, including transformation from pose (translation + rotation quaternion) to 3D keypoints with visibility computation.

## Features

- **Interactive 3D Visualization**: View keypoints in the camera frame with color-coded visibility status
- **Surface Rendering**: Visualize spacecraft surfaces (body, solar panels, antenna) as triangular meshes
- **Normal Vectors**: Display surface normals to understand orientation relative to camera viewpoint
- **Visibility Computation**: Automatic computation of keypoint visibility (visible, occluded, or not visible)
- **Detailed Keypoint Information**: View X, Y, Z coordinates and projected pixel locations for each keypoint
- **Adjustable Parameters**: Modify pose (translation + quaternion), camera intrinsics, and visualization options in real-time

## Installation & Usage

### Option 1: Docker (Recommended)

The easiest way to run the visualizer is using Docker Compose:

```bash
# Build and start the container
docker-compose up -d

# View logs (optional)
docker-compose logs -f

# Stop the container
docker-compose down
```

The app will be available at `http://localhost:8501`.

**Note**: The docker-compose.yml includes volume mounts for live editing. If you modify the Python files, Streamlit will automatically reload.

### Option 2: Local Python Installation

1. Install the required dependencies:
```bash
pip install -r requirements_visualizer.txt
```

2. Run the Streamlit application:
```bash
streamlit run keypoint_visualizer.py
```

The app will open in your browser at `http://localhost:8501`.

## Interface

### Sidebar Controls

**Pose Parameters:**
- **Translation (X, Y, Z)**: Position of the satellite in camera frame (meters)
- **Rotation Quaternion (w, x, y, z)**: Orientation of the satellite (automatically normalized)

**View-Relative Transform:**
- **Interactive Rotation**: Click and drag the 3D plot to rotate the view
  - Left-click drag: Orbit/rotate
  - Scroll wheel: Zoom in/out
  - Right-click drag: Pan
- **Offset X, Y, Z**: Translate satellite relative to view direction (meters, -2 to +2)

**Camera Intrinsics:**
- **fx, fy**: Focal lengths in pixels
- **cx, cy**: Principal point coordinates
- **Width, Height**: Image dimensions in pixels

**Visualization Options:**
- **Show Surfaces**: Toggle spacecraft surface mesh display
- **Show Rays (Ray-casting)**: Toggle ray visualization from camera to keypoints
- **Show Camera Point of View**: Toggle camera viewing direction vector
- **POV Vector Scale**: Adjust length of POV vector

### Main Display

**3D Visualization:**
- **Interactive Controls:**
  - 🖱️ Left-click drag to rotate viewpoint (camera orbits around satellite)
  - 🖱️ Scroll wheel to zoom in/out
  - 🖱️ Right-click drag to pan
- **Keypoints** color-coded by visibility:
  - 🟢 Green: Visible (ray from camera reaches keypoint without obstruction)
  - 🟠 Orange: Occluded (blocked by spacecraft body)
  - 🔴 Red: Not visible (outside image or behind camera)
- **Geometry:**
  - 🟡 Yellow edges: Main body surfaces (6 faces of rectangular prism)
  - 🟡 Yellow lines: Solar panels (KP1→KP8, KP2→KP9) and antenna (KP3→KP10)
- **Ray-casting Visualization:**
  - Grey solid rays (opacity 0.4): Visible keypoints (no intersection with surfaces)
  - Grey dashed rays (opacity 0.6, thicker): Occluded keypoints (intersect surface before reaching keypoint)
  - Grey dotted rays (opacity 0.3): Not visible (outside image bounds or behind camera)
- **Vectors:**
  - Cyan dashed arrow: Camera point of view vector (viewing direction along +Z axis)

**Keypoint Data Table:**
- Keypoint index (0-10)
- 3D coordinates in camera frame (X, Y, Z)
- 2D projected pixel coordinates (u, v)
- Visibility status

**Summary Statistics:**
- Count of visible, occluded, and not visible keypoints

## Keypoint Structure

The Tango satellite has 11 keypoints:

- **Keypoints 0-3**: Top corners of main body (z=0.3215m)
- **Keypoints 4-7**: Bottom corners of main body (z=0.0m)
- **Keypoint 8**: Left solar panel tip
- **Keypoint 9**: Right solar panel tip
- **Keypoint 10**: Antenna/appendage tip

Reference coordinates are defined in [reference_keypoints.txt](reference_keypoints.txt) and [yolo_visibility.py](yolo_visibility.py).

## Example Use Cases

1. **Understanding Pose-to-Keypoint Transformation**:
   - Adjust the translation and quaternion to see how the satellite orientation changes
   - Observe how keypoints transform from body frame to camera frame

2. **Interactive Satellite Manipulation**:
   - Drag the 3D plot to rotate and examine the satellite from any angle
   - Use translation sliders to adjust satellite position
   - See how transformations affect surface normals and visibility in real-time

3. **Visibility Analysis**:
   - Identify which keypoints are visible from a given viewpoint
   - Understand occlusion based on spacecraft geometry
   - Observe how rotation affects which surfaces are visible

4. **Surface Normal Analysis**:
   - Visualize which surfaces face the camera
   - Understand the relationship between surface orientation and visibility
   - See how normals update correctly with view-relative transformations

5. **Dataset Validation**:
   - Verify pose annotations from SPEED+ dataset
   - Debug keypoint projection issues
   - Fine-tune satellite poses using interactive controls

## Technical Details

### Coordinate Frames

- **Body Frame**: Satellite-fixed frame with origin at satellite center
- **Camera Frame**: Camera-fixed frame with Z-axis pointing forward (viewing direction)

### Transformation Pipeline

1. Load reference keypoints in body frame
2. **Apply view-relative translation** (if set):
   - Translate: `P_transformed = P_body + [tx, ty, tz]`
   - View rotation is handled interactively by dragging the 3D plot
3. Apply rotation matrix R (from quaternion) and translation t
4. Transform keypoints to camera frame: `P_camera = R @ P_transformed + t`
5. Project to 2D image plane using camera intrinsics
6. **Compute visibility using ray-casting**:
   - Cast ray from camera origin (0,0,0) to each keypoint
   - Check intersection with main body triangular faces using Möller–Trumbore algorithm
   - Keypoint is occluded if ray intersects any face before reaching the keypoint
   - Solar panels and antenna (represented as lines) do not occlude keypoints

### Visibility Flags

- **0**: Not visible (behind camera or outside image bounds)
- **1**: Occluded (inside image but hidden behind spacecraft body)
- **2**: Fully visible (inside image and on visible surface)

## Files

- [keypoint_visualizer.py](keypoint_visualizer.py): Main Streamlit application
- [yolo_visibility.py](yolo_visibility.py): Visibility computation and transformation functions
- [reference_keypoints.txt](reference_keypoints.txt): Reference 3D keypoint coordinates
- [requirements_visualizer.txt](requirements_visualizer.txt): Python dependencies
- [Dockerfile](Dockerfile): Docker image definition
- [docker-compose.yml](docker-compose.yml): Docker Compose configuration
- [.dockerignore](.dockerignore): Docker build exclusions

## Dependencies

- streamlit: Web application framework
- plotly: Interactive 3D plotting
- numpy: Numerical computations
