import numpy as np

def get_tango_faces():
    """
    Define the triangular faces of the Tango spacecraft main body for occlusion detection.

    The Tango spacecraft structure consists of:
    - Main body: rectangular prism (keypoints 0-7)
      * Keypoints 0-3: top face (z=0.3215m)
      * Keypoints 4-7: bottom face (z=0.0m)

    Solar panels and antenna are represented as lines (see get_tango_lines).

    Returns:
        list: List of triangular faces, where each face is defined by
              3 keypoint indices [idx1, idx2, idx3]
    """
    faces = [
        # ===== Main Body Faces Only =====

        # Top face (z = 0.3215m) - divided into 2 triangles
        [0, 1, 2],  # Triangle 1: top face
        [0, 2, 3],  # Triangle 2: top face

        # Bottom face (z = 0.0m) - divided into 2 triangles
        [4, 6, 5],  # Triangle 1: bottom face
        [4, 7, 6],  # Triangle 2: bottom face

        # Front face (positive y direction) - divided into 2 triangles
        [1, 2, 6],  # Triangle 1: front face
        [1, 6, 5],  # Triangle 2: front face

        # Back face (negative y direction) - divided into 2 triangles
        [0, 7, 4],  # Triangle 1: back face
        [0, 3, 7],  # Triangle 2: back face

        # Left face (negative x direction) - divided into 2 triangles
        [0, 1, 5],  # Triangle 1: left face
        [0, 5, 4],  # Triangle 2: left face

        # Right face (positive x direction) - divided into 2 triangles
        [2, 3, 7],  # Triangle 1: right face
        [2, 7, 6],  # Triangle 2: right face
    ]

    return faces


def get_tango_lines():
    """
    Define the line segments for solar panels and antenna.

    Returns:
        list: List of line segments, where each line is defined by
              2 keypoint indices [idx1, idx2]
    """
    lines = [
        # Left solar panel: k1 to k8
        [1, 8],

        # Right solar panel: k2 to k9
        [2, 9],

        # Antenna: k3 to k10
        [3, 10],
    ]

    return lines


def ray_triangle_intersection(ray_origin, ray_dir, v0, v1, v2, epsilon=1e-6):
    """
    Möller–Trumbore ray-triangle intersection algorithm.

    Args:
        ray_origin: (3,) ray origin point
        ray_dir: (3,) ray direction (normalized)
        v0, v1, v2: (3,) triangle vertices
        epsilon: small value for numerical stability

    Returns:
        (intersects, distance):
            - intersects: bool, True if ray intersects triangle
            - distance: float, distance along ray to intersection (if intersects)
    """
    edge1 = v1 - v0
    edge2 = v2 - v0

    h = np.cross(ray_dir, edge2)
    a = np.dot(edge1, h)

    # Ray is parallel to triangle
    if abs(a) < epsilon:
        return False, float('inf')

    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)

    # Intersection outside triangle
    if u < 0.0 or u > 1.0:
        return False, float('inf')

    q = np.cross(s, edge1)
    v = f * np.dot(ray_dir, q)

    # Intersection outside triangle
    if v < 0.0 or u + v > 1.0:
        return False, float('inf')

    # Compute distance along ray
    t = f * np.dot(edge2, q)

    # Intersection is behind ray origin
    if t < epsilon:
        return False, float('inf')

    return True, t


def is_keypoint_visible_raycasting(keypoint_idx, keypoints_camera, faces):
    """
    Check if a keypoint is visible using ray-casting.

    A keypoint is visible if a ray from the camera origin (0,0,0) to the keypoint
    does not intersect any triangular face before reaching the keypoint.

    Args:
        keypoint_idx: int, index of keypoint to check
        keypoints_camera: (N, 3) keypoints in camera frame
        faces: list of triangular faces [idx0, idx1, idx2]

    Returns:
        bool: True if keypoint is visible (not occluded)
    """
    camera_origin = np.array([0.0, 0.0, 0.0])
    keypoint = keypoints_camera[keypoint_idx]

    # Ray from camera to keypoint
    ray_dir = keypoint - camera_origin
    distance_to_keypoint = np.linalg.norm(ray_dir)

    # Degenerate case: keypoint at camera origin
    if distance_to_keypoint < 1e-6:
        return True

    ray_dir = ray_dir / distance_to_keypoint  # Normalize

    # Check intersection with all faces
    for face in faces:
        idx0, idx1, idx2 = face

        # Skip faces that contain this keypoint
        if keypoint_idx in face:
            continue

        v0 = keypoints_camera[idx0]
        v1 = keypoints_camera[idx1]
        v2 = keypoints_camera[idx2]

        intersects, t = ray_triangle_intersection(camera_origin, ray_dir, v0, v1, v2)

        # If ray intersects face before reaching keypoint, it's occluded
        if intersects and t < distance_to_keypoint - 1e-4:  # Small epsilon for numerical stability
            return False

    return True


def visibility_with_raycasting(uv, Z, W, H, keypoints_camera):
    """
    Compute YOLO11 pose visibility flags using ray-casting for occlusion detection.

    This function determines whether each keypoint is:
    - 0: Not visible (behind camera or outside image bounds)
    - 1: Occluded (inside image but hidden behind spacecraft body)
    - 2: Fully visible (inside image and directly reachable from camera)

    Args:
        uv (np.ndarray): (N, 2) array of projected 2D pixel coordinates
        Z (np.ndarray): (N,) array of depth values (z-coordinate in camera frame)
        W (int): Image width in pixels
        H (int): Image height in pixels
        keypoints_camera (np.ndarray): (N, 3) array of keypoint 3D positions in camera frame

    Returns:
        list: Visibility flags for each keypoint (0, 1, or 2)
    """
    flags = []
    faces = get_tango_faces()

    for i, ((u, v), z) in enumerate(zip(uv, Z)):
        # Check 1: Keypoint must be in front of camera (positive Z)
        in_front = z > 0

        # Check 2: Projected pixel must be inside image boundaries
        in_img = (0 <= u < W and 0 <= v < H)

        # If either check fails, keypoint is not visible
        if not (in_front and in_img):
            flags.append(0)  # Not visible
            continue

        # Check 3: Ray-casting occlusion test
        visible = is_keypoint_visible_raycasting(i, keypoints_camera, faces)

        if visible:
            flags.append(2)  # Fully visible
        else:
            flags.append(1)  # Occluded

    return flags


def visibility_with_tango_geometry(uv, Z, W, H, keypoints_camera, tango_keypoints, R):
    """
    Compute YOLO11 pose visibility flags using spacecraft geometry for occlusion detection.
    
    This function determines whether each keypoint is:
    - 0: Not visible (behind camera or outside image bounds)
    - 1: Occluded (inside image but hidden behind spacecraft body)
    - 2: Fully visible (inside image and on visible surface)
    
    The occlusion detection works by:
    1. Computing face normals for all spacecraft surfaces
    2. Determining which faces are visible from the camera viewpoint
    3. Checking if each keypoint belongs to a visible face
    
    Args:
        uv (np.ndarray): (N, 2) array of projected 2D pixel coordinates for each keypoint
        Z (np.ndarray): (N,) array of depth values (z-coordinate in camera frame) for each keypoint
        W (int): Image width in pixels
        H (int): Image height in pixels
        keypoints_camera (np.ndarray): (N, 3) array of keypoint 3D positions in camera frame
        tango_keypoints (np.ndarray): (N, 3) array of reference keypoint positions in spacecraft frame
        R (np.ndarray): (3, 3) rotation matrix transforming from spacecraft frame to camera frame
    
    Returns:
        list: Visibility flags for each keypoint (0, 1, or 2)
    
    Example:
        >>> uv = np.array([[320, 240], [640, 480], ...])  # 11 keypoints
        >>> Z = np.array([2.0, 2.1, ...])  # depths
        >>> vis = visibility_with_tango_geometry(uv, Z, 1280, 1024, kps_cam, kps_tango, R)
        >>> print(vis)  # [2, 2, 1, 0, ...]
    """
    flags = []
    
    # ===== Step 1: Determine camera viewing direction =====
    # The camera looks along the +Z axis in its own frame: [0, 0, 1]
    # We need this direction expressed in the spacecraft frame to compare with face normals
    # Transform: direction_spacecraft = R^T @ direction_camera
    camera_dir_spacecraft = R.T @ np.array([0, 0, 1])
    
    # ===== Step 2: Get spacecraft face definitions =====
    faces = get_tango_faces()
    
    # ===== Step 3: Determine which faces are visible from the camera =====
    face_visibility = []
    
    for face in faces:
        # Extract the 3 keypoint indices that define this triangular face
        idx0, idx1, idx2 = face
        
        # Get the 3D coordinates of the triangle vertices in spacecraft frame
        p0 = tango_keypoints[idx0]
        p1 = tango_keypoints[idx1]
        p2 = tango_keypoints[idx2]
        
        # Compute two edge vectors of the triangle
        v1 = p1 - p0  # Edge from vertex 0 to vertex 1
        v2 = p2 - p0  # Edge from vertex 0 to vertex 2
        
        # Compute face normal using cross product
        # The normal vector is perpendicular to the face surface
        normal = np.cross(v1, v2)
        
        # Normalize the normal vector to unit length
        norm_mag = np.linalg.norm(normal)
        if norm_mag > 1e-6:  # Check for degenerate triangles
            normal = normal / norm_mag
        else:
            # Degenerate triangle (all vertices collinear), mark as not visible
            face_visibility.append(False)
            continue
        
        # Check if face is visible using the dot product
        # If normal points toward camera (dot product < 0), the face is visible
        # If normal points away from camera (dot product > 0), the face is hidden
        dot = np.dot(normal, camera_dir_spacecraft)
        face_visibility.append(dot < 0)
    
    # ===== Step 4: Check visibility for each keypoint =====
    for i, ((u, v), z) in enumerate(zip(uv, Z)):
        
        # --- Basic visibility checks ---
        
        # Check 1: Keypoint must be in front of the camera (positive Z)
        in_front = z > 0
        
        # Check 2: Projected pixel must be inside image boundaries
        in_img = (0 <= u < W and 0 <= v < H)
        
        # If either check fails, keypoint is not visible
        if not (in_front and in_img):
            flags.append(0)  # Not visible
            continue
        
        # --- Occlusion check using face visibility ---
        
        # Check if this keypoint belongs to at least one visible face
        on_visible_face = False
        
        for face, is_visible in zip(faces, face_visibility):
            # If the keypoint index is part of this face AND the face is visible
            if i in face and is_visible:
                on_visible_face = True
                break
        
        # Assign visibility flag based on face membership
        if on_visible_face:
            flags.append(2)  # Fully visible
        else:
            flags.append(1)  # Occluded (on back side of spacecraft)
    
    return flags


# ===== Reference Tango Keypoints =====
# These are the 11 keypoints of the Tango spacecraft in its body frame (meters)
# Structure:
#   Keypoints 0-3: Top corners of main body (z=0.3215m)
#   Keypoints 4-7: Bottom corners of main body (z=0.0m)
#   Keypoint 8: Left solar panel tip
#   Keypoint 9: Right solar panel tip
#   Keypoint 10: Antenna/appendage tip

tango_keypoints = np.array([
    [-0.370, -0.385, 0.3215],  # 0: top-left-back
    [-0.370,  0.385, 0.3215],  # 1: top-left-front
    [ 0.370,  0.385, 0.3215],  # 2: top-right-front
    [ 0.370, -0.385, 0.3215],  # 3: top-right-back
    [-0.370, -0.264, 0.0000],  # 4: bottom-left-back
    [-0.370,  0.304, 0.0000],  # 5: bottom-left-front
    [ 0.370,  0.304, 0.0000],  # 6: bottom-right-front
    [ 0.370, -0.264, 0.0000],  # 7: bottom-right-back
    [-0.5427, 0.4877, 0.2535], # 8: left solar panel tip
    [ 0.5427, 0.4877, 0.2591], # 9: right solar panel tip
    [ 0.305, -0.579,  0.2515], # 10: antenna tip
])


# # ===== Usage Example =====
def create_yolo_label_with_occlusion(annotation, output_path):
    """
    Create a YOLO11 pose label file with accurate occlusion detection.
    
    Args:
        annotation (dict): SPEED+ annotation containing:
            - 'q_vbs2tango': quaternion [w, x, y, z]
            - 'r_Vo2To_vbs_true': translation [x, y, z]
            - 'fx', 'fy', 'cx', 'cy': camera intrinsics
            - 'width', 'height': image dimensions
        output_path (str): Path where to save the .txt label file
    
    Returns:
        None (writes YOLO format label to file)
    """
    # Extract pose parameters
    q = np.array(annotation['q_vbs2tango'])  # Quaternion [w, x, y, z]
    t = np.array(annotation['r_Vo2To_vbs_true'])  # Translation [x, y, z]
    
    # Build camera intrinsic matrix
    K = np.array([
        [annotation['fx'], 0, annotation['cx']],
        [0, annotation['fy'], annotation['cy']],
        [0, 0, 1]
    ])
    
    W = annotation['width']
    H = annotation['height']
    
    # Convert quaternion to rotation matrix
    R = quaternion_to_rotation_matrix(q)
    
    # Project 3D keypoints to 2D image plane
    uv, Z = project_points(K, R, t, tango_keypoints)
    
    # Transform keypoints to camera frame for geometry-based occlusion check
    keypoints_camera = (tango_keypoints @ R.T) + t.reshape(1, 3)
    
    # Compute visibility flags (0, 1, or 2) for each keypoint
    vis_flags = visibility_with_tango_geometry(
        uv, Z, W, H, 
        keypoints_camera, 
        tango_keypoints, 
        R
    )
    
    # Normalize pixel coordinates to [0, 1] range for YOLO format
    uv_norm = uv.copy()
    uv_norm[:, 0] /= W
    uv_norm[:, 1] /= H
    
    # Compute bounding box from visible keypoints only
    visible_mask = np.array(vis_flags) > 0  # Include both occluded (1) and visible (2)
    
    if not visible_mask.any():
        print(f"Warning: No visible keypoints, skipping label creation")
        return
    
    # Get pixel coordinates of visible keypoints
    visible_uv = uv[visible_mask]
    x_min, y_min = visible_uv.min(axis=0)
    x_max, y_max = visible_uv.max(axis=0)
    
    # Convert bounding box to YOLO format (normalized center + size)
    x_center = (x_min + x_max) / 2 / W
    y_center = (y_min + y_max) / 2 / H
    bbox_width = (x_max - x_min) / W
    bbox_height = (y_max - y_min) / H
    
    # Write YOLO11 pose label file
    # Format: <class> <x_center> <y_center> <width> <height> <x1> <y1> <v1> <x2> <y2> <v2> ...
    with open(output_path, 'w') as f:
        # Class ID (0 for spacecraft) + bounding box
        f.write(f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")
        
        # All 11 keypoints with their visibility flags
        for (x, y), v in zip(uv_norm, vis_flags):
            f.write(f" {x:.6f} {y:.6f} {v}")
        
        f.write("\n")


def project_points(K, R, t, P3d):
    """
    Project 3D points from spacecraft frame to 2D image plane.
    
    Args:
        K (np.ndarray): (3, 3) camera intrinsic matrix
        R (np.ndarray): (3, 3) rotation matrix (spacecraft to camera frame)
        t (np.ndarray): (3,) translation vector
        P3d (np.ndarray): (N, 3) 3D points in spacecraft frame
    
    Returns:
        tuple: (uv, Z) where
            - uv: (N, 2) pixel coordinates
            - Z: (N,) depth values in camera frame
    """
    # Transform points to camera frame: Pc = R @ P + t
    Pc = (P3d @ R.T) + t.reshape(1, 3)
    
    # Extract depth (Z coordinate in camera frame)
    Z = Pc[:, 2]
    
    # Avoid division by zero
    eps = 1e-9
    Zs = np.where(abs(Z) < eps, eps, Z)
    
    # Normalize by depth to get image plane coordinates
    x = Pc[:, 0] / Zs
    y = Pc[:, 1] / Zs
    
    # Apply camera intrinsics to get pixel coordinates
    uv = (K @ np.vstack([x, y, np.ones_like(x)])).T
    
    return uv[:, :2], Z


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to 3x3 rotation matrix.
    
    Args:
        q (np.ndarray): Quaternion in [w, x, y, z] format
    
    Returns:
        np.ndarray: (3, 3) rotation matrix
    """
    w, x, y, z = q
    return np.array([
        [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
    ])