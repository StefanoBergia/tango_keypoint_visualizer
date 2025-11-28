"""
SPEED+ Tango Satellite Interactive Pose Control

This version uses horizontal sliders under the plot for snappy, responsive control.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from functools import lru_cache
from yolo_visibility import (
    tango_keypoints,
    get_tango_faces,
    get_tango_lines,
    quaternion_to_rotation_matrix,
    project_points,
    visibility_with_raycasting
)
#from keypoint_visualizer import create_3d_visualization, transform_keypoints_to_camera


# Cache expensive computations
@st.cache_data
def get_cached_faces():
    """Cache the face definitions since they never change."""
    return get_tango_faces()


@st.cache_data
def get_cached_lines():
    """Cache the line definitions since they never change."""
    return get_tango_lines()


@st.cache_data
def compute_satellite_state(tx, ty, tz, roll, pitch, yaw, fx, fy, cx, cy, W, H):
    """
    Precompute and cache all expensive operations for a given pose.

    This function is cached by Streamlit, so repeated calls with the same
    parameters will return instantly without recomputation.

    Args:
        tx, ty, tz: Translation in meters
        roll, pitch, yaw: Rotation in degrees
        fx, fy, cx, cy: Camera intrinsics
        W, H: Image dimensions

    Returns:
        tuple: (keypoints_camera, uv, Z, vis_flags, q)
    """
    # Convert Euler angles to quaternion
    roll_rad = np.deg2rad(roll)
    pitch_rad = np.deg2rad(pitch)
    yaw_rad = np.deg2rad(yaw)
    q = euler_to_quaternion(roll_rad, pitch_rad, yaw_rad)

    # Create camera matrix
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    t = np.array([tx, ty, tz])

    # Transform keypoints
    keypoints_body = tango_keypoints
    R = quaternion_to_rotation_matrix(q)
    keypoints_camera = transform_keypoints_to_camera(keypoints_body, R, t)

    # Project to 2D
    uv, Z = project_points(K, R, t, keypoints_body)

    # Compute visibility
    vis_flags = visibility_with_raycasting(uv, Z, W, H, keypoints_camera)

    return keypoints_camera, uv, Z, vis_flags, q


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) in radians to quaternion [w, x, y, z].

    Args:
        roll: Rotation around x-axis (radians)
        pitch: Rotation around y-axis (radians)
        yaw: Rotation around z-axis (radians)

    Returns:
        np.array: Quaternion [w, x, y, z]
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def create_2d_fov_plot(uv, vis_flags, W, H):
    """
    Create 2D field of view plot showing projected keypoints.

    Args:
        uv: (N, 2) projected pixel coordinates
        vis_flags: list of visibility flags
        W, H: image dimensions

    Returns:
        plotly figure
    """
    fig = go.Figure()

    # Draw image boundaries
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=W, y1=H,
        line=dict(color="cyan", width=3),
        fillcolor="rgba(0,100,200,0.1)"
    )

    # Color mapping for visibility
    colors = {0: 'red', 1: 'orange', 2: 'green'}
    symbols = {0: 'x', 1: 'circle-open', 2: 'circle'}

    # Plot keypoints
    for vis_flag in [0, 1, 2]:
        mask = np.array(vis_flags) == vis_flag
        if mask.any():
            indices = np.where(mask)[0]
            points = uv[mask]

            fig.add_trace(go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode='markers+text',
                marker=dict(size=12, color=colors[vis_flag], symbol=symbols[vis_flag], line=dict(width=2)),
                text=[f'KP{i}' for i in indices],
                textposition='top center',
                name={0: 'Not visible', 1: 'Occluded', 2: 'Visible'}[vis_flag],
                hovertemplate='<b>KP%{text}</b><br>u: %{x:.1f}<br>v: %{y:.1f}<extra></extra>'
            ))

    # Add center crosshair
    center_x, center_y = W/2, H/2
    fig.add_trace(go.Scatter(
        x=[center_x], y=[center_y],
        mode='markers',
        marker=dict(size=15, color='white', symbol='cross', line=dict(width=2, color='black')),
        name='Image Center',
        showlegend=False
    ))

    fig.update_layout(
        title='2D Camera View (Field of View)',
        xaxis_title='u (pixels)',
        yaxis_title='v (pixels)',
        xaxis=dict(range=[-W*0.1, W*1.1], constrain='domain'),
        yaxis=dict(range=[H*1.1, -H*0.1], scaleanchor='x', constrain='domain'),  # Inverted Y for image coordinates
        height=400,
        showlegend=True,
        legend=dict(x=1.05, y=1),
        uirevision='fov_view'  # Preserve zoom/pan state
    )

    return fig
def transform_keypoints_to_camera(keypoints_body, R, t):
    """
    Transform keypoints from body frame to camera frame.

    Args:
        keypoints_body: (N, 3) array of keypoints in body frame
        R: (3, 3) rotation matrix from body to camera frame
        t: (3,) translation vector

    Returns:
        (N, 3) array of keypoints in camera frame
    """
    return (keypoints_body @ R.T) + t.reshape(1, 3)


def create_3d_visualization(keypoints_camera, vis_flags,
                           show_surfaces=True, show_rays=True,
                           show_pov=True, pov_scale=1.0,
                           show_fov_cone=False, K=None, fov_depth=5.0):
    """
    Create 3D plotly visualization of keypoints with surfaces and ray-casting visualization.

    Args:
        keypoints_camera: (N, 3) keypoints in camera frame
        vis_flags: list of visibility flags (0, 1, or 2)
        show_surfaces: bool, whether to show surface triangles
        show_rays: bool, whether to show rays from camera to keypoints
        show_pov: bool, whether to show camera point of view vector
        pov_scale: float, scale factor for POV vector
        show_fov_cone: bool, whether to show field of view frustum
        K: (3, 3) camera intrinsic matrix (required if show_fov_cone=True)
        fov_depth: float, depth at which to draw the FOV rectangle

    Returns:
        plotly figure object
    """
    fig = go.Figure()

    # Get faces
    faces = get_tango_faces()

    # Color mapping for visibility
    colors = {0: 'red', 1: 'orange', 2: 'green'}
    color_names = {0: 'Not visible', 1: 'Occluded', 2: 'Visible'}

    # Plot keypoints with different colors based on visibility
    for vis_flag in [0, 1, 2]:
        mask = np.array(vis_flags) == vis_flag
        if mask.any():
            kps = keypoints_camera[mask]
            indices = np.where(mask)[0]

            fig.add_trace(go.Scatter3d(
                x=kps[:, 0],
                y=kps[:, 1],
                z=kps[:, 2],
                mode='markers+text',
                marker=dict(size=8, color=colors[vis_flag]),
                text=[f'KP{i}' for i in indices],
                textposition='top center',
                name=color_names[vis_flag],
                hovertemplate='<b>Keypoint %{text}</b><br>' +
                             'X: %{x:.3f}m<br>' +
                             'Y: %{y:.3f}m<br>' +
                             'Z: %{z:.3f}m<br>' +
                             f'Status: {color_names[vis_flag]}<extra></extra>'
            ))

    # Plot main body surfaces (triangles)
    if show_surfaces:
        for face in faces:
            # Get vertices in camera frame
            idx0, idx1, idx2 = face
            vertices = keypoints_camera[[idx0, idx1, idx2]]

            # Close the triangle
            x = list(vertices[:, 0]) + [vertices[0, 0]]
            y = list(vertices[:, 1]) + [vertices[0, 1]]
            z = list(vertices[:, 2]) + [vertices[0, 2]]

            # Yellow edges for high visibility
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color='yellow', width=3),
                opacity=1.0,
                showlegend=False,
                hoverinfo='skip'
            ))

        # Plot solar panels and antenna (lines)
        lines = get_tango_lines()
        for line in lines:
            idx0, idx1 = line
            p0 = keypoints_camera[idx0]
            p1 = keypoints_camera[idx1]

            fig.add_trace(go.Scatter3d(
                x=[p0[0], p1[0]],
                y=[p0[1], p1[1]],
                z=[p0[2], p1[2]],
                mode='lines',
                line=dict(color='yellow', width=4),
                showlegend=False,
                hovertemplate=f'<b>Line: KP{idx0} → KP{idx1}</b><extra></extra>'
            ))

    # Plot rays from camera to keypoints
    if show_rays:
        camera_origin = np.array([0.0, 0.0, 0.0])

        for i, keypoint in enumerate(keypoints_camera):
            vis_flag = vis_flags[i]

            # All rays in grey with different styles for visibility
            if vis_flag == 2:  # Visible
                ray_width = 1
                ray_dash = 'solid'
                ray_opacity = 0.4
            elif vis_flag == 1:  # Occluded
                ray_width = 2
                ray_dash = 'dash'
                ray_opacity = 0.6
            else:  # Not visible
                ray_width = 1
                ray_dash = 'dot'
                ray_opacity = 0.3

            # Draw ray from camera to keypoint
            fig.add_trace(go.Scatter3d(
                x=[camera_origin[0], keypoint[0]],
                y=[camera_origin[1], keypoint[1]],
                z=[camera_origin[2], keypoint[2]],
                mode='lines',
                line=dict(color='grey', width=ray_width, dash=ray_dash),
                opacity=ray_opacity,
                showlegend=False,
                hovertemplate=f'<b>Ray to KP{i}</b><br>Status: {["Not visible", "Occluded", "Visible"][vis_flag]}<extra></extra>'
            ))

    # Draw field of view frustum
    if show_fov_cone and K is not None:
        # Get image dimensions from K
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        W, H = cx * 2, cy * 2  # Approximate image dimensions

        # Camera origin
        cam_origin = np.array([0, 0, 0])

        # Compute 4 corners of image plane at fov_depth
        # Convert image corners to normalized coordinates, then to 3D rays
        corners_2d = np.array([
            [0, 0],      # Top-left
            [W, 0],      # Top-right
            [W, H],      # Bottom-right
            [0, H]       # Bottom-left
        ])

        # Unproject corners to 3D rays at fov_depth
        corners_3d = []
        for u, v in corners_2d:
            # Normalized image coordinates
            x_norm = (u - cx) / fx
            y_norm = (v - cy) / fy
            # 3D point at fov_depth
            point_3d = np.array([x_norm * fov_depth, y_norm * fov_depth, fov_depth])
            corners_3d.append(point_3d)

        corners_3d = np.array(corners_3d)

        # Draw lines from camera origin to each corner
        for i, corner in enumerate(corners_3d):
            fig.add_trace(go.Scatter3d(
                x=[cam_origin[0], corner[0]],
                y=[cam_origin[1], corner[1]],
                z=[cam_origin[2], corner[2]],
                mode='lines',
                line=dict(color='cyan', width=2, dash='dot'),
                showlegend=False if i > 0 else True,
                name='FOV Frustum' if i == 0 else None,
                hoverinfo='skip'
            ))

        # Draw rectangle at fov_depth (connecting the 4 corners)
        for i in range(4):
            next_i = (i + 1) % 4
            fig.add_trace(go.Scatter3d(
                x=[corners_3d[i][0], corners_3d[next_i][0]],
                y=[corners_3d[i][1], corners_3d[next_i][1]],
                z=[corners_3d[i][2], corners_3d[next_i][2]],
                mode='lines',
                line=dict(color='cyan', width=3),
                showlegend=False,
                hoverinfo='skip'
            ))

    # Plot camera point of view vector
    if show_pov:
        # Compute satellite centroid in camera frame
        centroid_camera = np.mean(keypoints_camera, axis=0)

        # Camera viewing direction in camera frame (Z-axis)
        camera_dir_camera = np.array([0, 0, 1])

        # Start point for POV vector (at centroid or offset backwards)
        pov_start = centroid_camera - camera_dir_camera * pov_scale
        pov_end = centroid_camera + camera_dir_camera * pov_scale * 0.5

        # Draw POV vector
        fig.add_trace(go.Scatter3d(
            x=[pov_start[0], pov_end[0]],
            y=[pov_start[1], pov_end[1]],
            z=[pov_start[2], pov_end[2]],
            mode='lines',
            line=dict(color='cyan', width=6, dash='dash'),
            name='Camera POV',
            hovertemplate='<b>Camera Point of View</b><br>' +
                         'Direction: (%.2f, %.2f, %.2f)<br>' % tuple(camera_dir_camera) +
                         'Camera looks along +Z axis<extra></extra>'
        ))

        # Add arrowhead for POV vector
        fig.add_trace(go.Cone(
            x=[pov_end[0]],
            y=[pov_end[1]],
            z=[pov_end[2]],
            u=[camera_dir_camera[0]],
            v=[camera_dir_camera[1]],
            w=[camera_dir_camera[2]],
            colorscale=[[0, 'cyan'], [1, 'cyan']],
            showscale=False,
            sizemode='absolute',
            sizeref=0.1,
            showlegend=False,
            hoverinfo='skip'
        ))

    # Set layout with turntable drag mode (rotates viewpoint/camera, not object)
    fig.update_layout(
        title='SPEED+ Tango Satellite Keypoints (Camera Frame)<br><sub>🖱️ Drag to rotate viewpoint | Scroll to zoom | Right-click drag to pan</sub>',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # Default viewing angle (only used on first render)
            )
        ),
        showlegend=True,
        height=700,
        dragmode='turntable',  # Rotate viewpoint/camera, not object
        uirevision='preserve_view'  # Preserve camera state across updates
    )

    return fig

def main():
    st.set_page_config(page_title="SPEED+ Interactive Control", layout="wide")

    st.title("SPEED+ Tango Satellite - Interactive Pose Control")
    st.markdown("""
    **Camera fixed at origin (0,0,0).** Adjust satellite pose with sliders below.
    💡 *Tip: Values are cached for faster updates when revisiting positions.*
    """)

    # Sidebar for camera and visualization options only
    st.sidebar.header("📷 Camera Parameters")
    fx = st.sidebar.number_input("fx", value=1015.0, step=10.0, format="%.1f")
    fy = st.sidebar.number_input("fy", value=1015.0, step=10.0, format="%.1f")
    cx = st.sidebar.number_input("cx", value=640.0, step=10.0, format="%.1f")
    cy = st.sidebar.number_input("cy", value=512.0, step=10.0, format="%.1f")
    W = st.sidebar.number_input("Width", value=1280, step=10)
    H = st.sidebar.number_input("Height", value=1024, step=10)

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    st.sidebar.header("👁️ Visualization")
    show_surfaces = st.sidebar.checkbox("Show Surfaces", value=True)
    show_rays = st.sidebar.checkbox("Show Rays", value=True)
    show_fov_cone = st.sidebar.checkbox("Show FOV Cone", value=True)

    # Get slider values from session state or defaults
    if 'tx' not in st.session_state:
        st.session_state.tx = 0.0
    if 'ty' not in st.session_state:
        st.session_state.ty = 0.0
    if 'tz' not in st.session_state:
        st.session_state.tz = 5.0
    if 'roll' not in st.session_state:
        st.session_state.roll = 0.0
    if 'pitch' not in st.session_state:
        st.session_state.pitch = 0.0
    if 'yaw' not in st.session_state:
        st.session_state.yaw = 0.0

    # Use values from session state for computations
    tx, ty, tz = st.session_state.tx, st.session_state.ty, st.session_state.tz
    roll, pitch, yaw = st.session_state.roll, st.session_state.pitch, st.session_state.yaw

    # Round slider values to reduce cache misses while maintaining precision
    # This creates a "snapping" effect that helps with caching
    tx = round(tx, 2)  # Round to 0.01m (10mm precision)
    ty = round(ty, 2)
    tz = round(tz, 2)
    roll = round(roll, 1)  # Round to 0.1° precision
    pitch = round(pitch, 1)
    yaw = round(yaw, 1)

    # Use cached computation - this will be instant for repeated poses
    keypoints_camera, uv, Z, vis_flags, q = compute_satellite_state(
        tx, ty, tz, roll, pitch, yaw,
        fx, fy, cx, cy, int(W), int(H)
    )

    # Reconstruct K and t for visualization
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    t = np.array([tx, ty, tz])

    # Show current pose values for debugging
    st.caption(f"Pose: t=({tx:.2f}, {ty:.2f}, {tz:.2f}), rotation=({roll:.1f}°, {pitch:.1f}°, {yaw:.1f}°)")

    # Display visualization in two rows
    # Row 1: 3D view (left) and 2D FOV (right)
    col1, col2 = st.columns([2, 1])

    with col1:
        # Compute FOV depth to extend to the furthest keypoint
        max_depth = np.max(keypoints_camera[:, 2]) * 1.5 if len(keypoints_camera) > 0 else 10.0

        # Create 3D visualization with origin marker and FOV cone
        # Force recreation by converting cached numpy arrays to fresh copies
        fig = create_3d_visualization(
            keypoints_camera.copy(),  # Fresh copy to ensure Streamlit detects change
            list(vis_flags),  # Convert to list to ensure detection
            show_surfaces=show_surfaces,
            show_rays=show_rays,
            show_pov=False,
            pov_scale=1.0,
            show_fov_cone=show_fov_cone,
            K=K.copy(),
            fov_depth=max_depth
        )

        # Add origin marker at (0,0,0)
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=12, color='red', symbol='diamond', line=dict(width=2, color='white')),
            name='Camera Origin',
            hovertemplate='<b>Camera Origin</b><br>Position: (0, 0, 0)<extra></extra>'
        ))

        # The uirevision is already set in create_3d_visualization
        # We don't need to override it, just ensure it stays consistent

        # Use config to improve performance
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'tango_satellite',
                'height': 1080,
                'width': 1920,
                'scale': 1
            }
        }

        st.plotly_chart(fig, use_container_width=True, key="main_plot", config=config)

    with col2:
        # 2D Field of View plot
        fov_fig = create_2d_fov_plot(uv, vis_flags, W, H)
        st.plotly_chart(fov_fig, use_container_width=True, key="fov_plot")

        # Stats
        st.markdown("### 📊 Stats")
        visible = sum(1 for v in vis_flags if v == 2)
        occluded = sum(1 for v in vis_flags if v == 1)
        not_visible = sum(1 for v in vis_flags if v == 0)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("✅", f"{visible}")
        with col_b:
            st.metric("⚠️", f"{occluded}")
        with col_c:
            st.metric("❌", f"{not_visible}")

    # Row 2: Control sliders at the bottom
    st.markdown("---")
    st.markdown("### 🎮 Controls")

    col_tx, col_ty, col_tz = st.columns(3)
    with col_tx:
        # Step size matches cache rounding (0.01m precision)
        st.session_state.tx = st.slider("X (Left/Right)", -5.0, 5.0, st.session_state.tx, 0.01, key="slider_tx")
    with col_ty:
        st.session_state.ty = st.slider("Y (Up/Down)", -5.0, 5.0, st.session_state.ty, 0.01, key="slider_ty")
    with col_tz:
        st.session_state.tz = st.slider("Z (Distance)", 0.5, 15.0, st.session_state.tz, 0.01, key="slider_tz")

    col_roll, col_pitch, col_yaw = st.columns(3)
    with col_roll:
        # Step size matches cache rounding (0.1° precision)
        st.session_state.roll = st.slider("Roll (°)", -180.0, 180.0, st.session_state.roll, 0.1, key="slider_roll")
    with col_pitch:
        st.session_state.pitch = st.slider("Pitch (°)", -180.0, 180.0, st.session_state.pitch, 0.1, key="slider_pitch")
    with col_yaw:
        st.session_state.yaw = st.slider("Yaw (°)", -180.0, 180.0, st.session_state.yaw, 0.1, key="slider_yaw")


if __name__ == "__main__":
    main()
