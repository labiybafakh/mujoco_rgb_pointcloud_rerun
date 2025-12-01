#!/usr/bin/env python3
"""
Rerun demo: MuJoCo passive viewer + real-time Rerun point cloud visualization + RGB camera
This shows the MuJoCo simulation visuals, point clouds, and simulation data
"""

import numpy as np
import time
import mujoco as mj
import mujoco.viewer
import rerun as rr

def create_pointcloud_from_scene(model, data, camera_id=0, points_per_geom={'box': 100, 'sphere': 150, 'cylinder': 120}):
    """Create point cloud directly from MuJoCo scene geometry"""
    points = []
    colors = []
    
    # Get camera parameters
    if camera_id < model.ncam:
        cam_pos = data.cam_xpos[camera_id]
        cam_mat = data.cam_xmat[camera_id].reshape(3, 3)
    else:
        # Use a default viewpoint
        cam_pos = np.array([2.0, 2.0, 2.0])
        cam_mat = np.eye(3)
    
    # Sample points from all geometries in the scene
    for geom_id in range(model.ngeom):
        geom_type = model.geom_type[geom_id]
        geom_size = model.geom_size[geom_id]
        geom_pos = data.geom_xpos[geom_id]
        geom_mat = data.geom_xmat[geom_id].reshape(3, 3)
        
        # Sample points based on geometry type
        if geom_type == mj.mjtGeom.mjGEOM_BOX:
            # Sample points on box surface
            for i in range(points_per_geom['box']):
                # Random point on box surface
                face = np.random.randint(0, 6)
                if face < 2:  # x faces
                    local_point = np.array([(-1)**face * geom_size[0], 
                                          np.random.uniform(-geom_size[1], geom_size[1]),
                                          np.random.uniform(-geom_size[2], geom_size[2])])
                elif face < 4:  # y faces  
                    local_point = np.array([np.random.uniform(-geom_size[0], geom_size[0]),
                                          (-1)**(face-2) * geom_size[1],
                                          np.random.uniform(-geom_size[2], geom_size[2])])
                else:  # z faces
                    local_point = np.array([np.random.uniform(-geom_size[0], geom_size[0]),
                                          np.random.uniform(-geom_size[1], geom_size[1]),
                                          (-1)**(face-4) * geom_size[2]])
                
                # Transform to world coordinates
                world_point = geom_pos + geom_mat @ local_point
                points.append(world_point)
                
                # Use material color if available, otherwise default color
                if hasattr(model, 'mat_rgba') and model.geom_matid[geom_id] >= 0:
                    mat_id = model.geom_matid[geom_id]
                    rgba = model.mat_rgba[mat_id]
                    color = [int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)]
                else:
                    # Default colors based on geometry type
                    color = [150, 100, 50]  # Brown for boxes
                colors.append(color)
        
        elif geom_type == mj.mjtGeom.mjGEOM_SPHERE:
            # Sample points on sphere surface
            for i in range(points_per_geom['sphere']):
                # Random point on sphere surface
                phi = np.random.uniform(0, 2 * np.pi)
                theta = np.random.uniform(0, np.pi)
                radius = geom_size[0]
                
                local_point = radius * np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ])
                
                # Transform to world coordinates
                world_point = geom_pos + geom_mat @ local_point
                points.append(world_point)
                
                # Use material color if available, otherwise default color
                if hasattr(model, 'mat_rgba') and model.geom_matid[geom_id] >= 0:
                    mat_id = model.geom_matid[geom_id]
                    rgba = model.mat_rgba[mat_id]
                    color = [int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)]
                else:
                    # Default colors based on geometry type
                    color = [100, 150, 200]  # Blue for spheres
                colors.append(color)
        
        elif geom_type == mj.mjtGeom.mjGEOM_CYLINDER:
            # Sample points on cylinder surface
            for i in range(points_per_geom['cylinder']):
                # Random point on cylinder surface (sides + caps)
                if np.random.random() < 0.8:  # Side surface
                    theta = np.random.uniform(0, 2 * np.pi)
                    z = np.random.uniform(-geom_size[1], geom_size[1])
                    local_point = np.array([
                        geom_size[0] * np.cos(theta),
                        geom_size[0] * np.sin(theta),
                        z
                    ])
                else:  # End caps
                    theta = np.random.uniform(0, 2 * np.pi)
                    r = np.random.uniform(0, geom_size[0])
                    z = np.random.choice([-geom_size[1], geom_size[1]])
                    local_point = np.array([
                        r * np.cos(theta),
                        r * np.sin(theta),
                        z
                    ])
                
                # Transform to world coordinates
                world_point = geom_pos + geom_mat @ local_point
                points.append(world_point)
                
                # Use material color if available, otherwise default color
                if hasattr(model, 'mat_rgba') and model.geom_matid[geom_id] >= 0:
                    mat_id = model.geom_matid[geom_id]
                    rgba = model.mat_rgba[mat_id]
                    color = [int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)]
                else:
                    # Default colors based on geometry type
                    color = [200, 100, 100]  # Red for cylinders
                colors.append(color)
    
    return np.array(points), np.array(colors, dtype=np.uint8)

def uniform_downsample(points, colors, target_points):
    """
    Uniform downsampling - simple but preserves distribution
    
    Algorithm: Select every k-th point where k = total_points / target_points
    Equation: selected_indices = [0, k, 2k, 3k, ...] where k = ⌊N/M⌋
    Where N = total points, M = target points
    """
    if len(points) <= target_points:
        return points, colors
    
    step = len(points) // target_points
    return points[::step], colors[::step]

def farthest_point_sampling(points, colors, target_points):
    """
    Farthest Point Sampling - better preservation of geometry
    
    Algorithm: Iteratively select points that are farthest from already selected points
    Equations:
    1. d(i) = min(||p_i - s_j||) for all selected points s_j
    2. next_point = argmax(d(i)) for all unselected points i
    3. Repeat until target_points reached
    
    Where ||·|| is Euclidean distance: ||p_i - s_j|| = √(Σ(p_i_k - s_j_k)²)
    """
    if len(points) <= target_points:
        return points, colors
    
    n_points = len(points)
    sampled_indices = np.zeros(target_points, dtype=int)
    distances = np.full(n_points, np.inf)
    
    # Start with a random point
    sampled_indices[0] = np.random.randint(0, n_points)
    
    for i in range(1, target_points):
        # Update distances to the last sampled point
        last_point = points[sampled_indices[i-1]]
        new_distances = np.linalg.norm(points - last_point, axis=1)
        distances = np.minimum(distances, new_distances)
        
        # Select the farthest point
        sampled_indices[i] = np.argmax(distances)
        distances[sampled_indices[i]] = 0
    
    return points[sampled_indices], colors[sampled_indices]

def voxel_downsample(points, colors, voxel_size=0.02):
    """
    Voxel-based downsampling - removes nearby points
    
    Algorithm: Divide space into cubic voxels and average points within each voxel
    Equations:
    1. voxel_index = ⌊p/voxel_size⌋ for point p
    2. For each voxel V: p_avg = (1/|V|) * Σ(p_i) for all p_i ∈ V
    3. c_avg = (1/|V|) * Σ(c_i) for all colors c_i ∈ V
    
    Where ⌊·⌋ is floor operation and |V| is number of points in voxel V
    """
    if len(points) == 0:
        return points, colors
    
    # Create voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(int)
    
    # Create unique voxel keys
    voxel_keys = [tuple(idx) for idx in voxel_indices]
    
    # Group points by voxel and take average
    unique_voxels = {}
    for i, key in enumerate(voxel_keys):
        if key not in unique_voxels:
            unique_voxels[key] = []
        unique_voxels[key].append(i)
    
    # Average points and colors in each voxel
    downsampled_points = []
    downsampled_colors = []
    
    for indices in unique_voxels.values():
        # Average position
        avg_point = np.mean(points[indices], axis=0)
        # Average color
        avg_color = np.mean(colors[indices], axis=0).astype(np.uint8)
        
        downsampled_points.append(avg_point)
        downsampled_colors.append(avg_color)
    
    return np.array(downsampled_points), np.array(downsampled_colors)

def depth_to_pointcloud(rgb_image, depth_image, camera_intrinsics, depth_scale=1000.0):
    """Convert RGB-D images to point cloud"""
    if rgb_image is None or depth_image is None:
        return np.array([]), np.array([])
    
    height, width = depth_image.shape
    fx, fy = camera_intrinsics['fx'], camera_intrinsics['fy']
    cx, cy = camera_intrinsics['cx'], camera_intrinsics['cy']
    
    # Create coordinate matrices
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Filter valid depth values
    valid_depth = (depth_image > 0) & (depth_image < 100)  # Filter far points
    
    if not np.any(valid_depth):
        return np.array([]), np.array([])
    
    # Convert depth to real-world coordinates
    z = depth_image[valid_depth] / depth_scale
    x = (u[valid_depth] - cx) * z / fx
    y = (v[valid_depth] - cy) * z / fy
    
    # Stack to create 3D points
    points_3d = np.column_stack((x, y, z))
    
    # Get corresponding colors
    if len(rgb_image.shape) == 3:
        colors = rgb_image[valid_depth]
    else:
        colors = np.tile(rgb_image[valid_depth], (3, 1)).T
    
    return points_3d, colors

def generate_pointcloud_from_camera(model, data, renderer, camera_id=0, max_points=8000, points_per_geom={'box': 100, 'sphere': 150, 'cylinder': 120}, downsample_method='voxel'):
    """Generate real point cloud from scene geometry"""
    try:
        # Use geometry-based point cloud generation (more reliable than depth buffer)
        points_3d, colors = create_pointcloud_from_scene(model, data, camera_id, points_per_geom)
        
        # Apply downsampling if needed
        if len(points_3d) > max_points:
            if downsample_method == 'uniform':
                points_3d, colors = uniform_downsample(points_3d, colors, max_points)
            elif downsample_method == 'fps':
                points_3d, colors = farthest_point_sampling(points_3d, colors, max_points)
            elif downsample_method == 'voxel':
                points_3d, colors = voxel_downsample(points_3d, colors, voxel_size=0.02)
            else:
                # Fallback to uniform
                points_3d, colors = uniform_downsample(points_3d, colors, max_points)
        
        return points_3d, colors
        
    except Exception as e:
        print(f"Warning: Point cloud generation failed: {e}")
        return np.array([]), np.array([])

def capture_rgb_image(model, data, renderer, camera_id=0):
    """Capture RGB image from MuJoCo camera"""
    try:
        # Set camera
        if camera_id < model.ncam:
            # Render the scene
            renderer.update_scene(data, camera=camera_id)
            rgb_array = renderer.render()
            return rgb_array
        else:
            # Use free camera if no fixed camera available
            renderer.update_scene(data)
            rgb_array = renderer.render()
            return rgb_array
    except Exception as e:
        print(f"Warning: RGB capture failed: {e}")
        return None

def log_simulation_state(model, data, frame_count, renderer=None):
    """Log the current simulation state to Rerun"""
    # Log simulation time
    rr.set_time_sequence("frame", frame_count)
    rr.set_time_seconds("sim_time", data.time)
    
    # Capture and log RGB image
    if renderer is not None:
        rgb_image = capture_rgb_image(model, data, renderer)
        if rgb_image is not None:
            rr.log("simulation/camera/rgb", rr.Image(rgb_image))
    
    # Log body positions as 3D points
    body_positions = []
    body_colors = []
    
    for i in range(model.nbody):
        if i == 0:  # Skip world body
            continue
        
        pos = data.xpos[i]
        body_positions.append(pos)
        
        # Color based on body index
        color = [
            int(255 * (0.8 + 0.2 * np.sin(i * 1.0))),
            int(255 * (0.3 + 0.7 * np.sin(i * 1.5 + 1))),
            int(255 * (0.2 + 0.8 * np.sin(i * 2.0 + 3)))
        ]
        body_colors.append(color)
    
    if body_positions:
        rr.log("simulation/bodies", rr.Points3D(
            positions=body_positions,
            colors=body_colors,
            radii=0.05
        ))

def main():
    # Configuration for point cloud density
    POINT_CLOUD_CONFIG = {
        'max_points': 12000,  # Maximum total points for performance
        'points_per_geom': {
            'box': 200,      # Points per box geometry
            'sphere': 300,   # Points per sphere geometry  
            'cylinder': 250  # Points per cylinder geometry
        },
        'downsample_method': 'voxel'  # Options: 'uniform', 'fps', 'voxel'
    }
    
    model_path = "config/camera_environment.xml"
    
    # Initialize Rerun
    rr.init("MuJoCo Point Cloud Demo", spawn=True)
    
    # Set up the 3D view
    rr.log("simulation", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # Load model
    print(f"Loading model: {model_path}")
    model = mj.MjModel.from_xml_path(model_path)
    data = mj.MjData(model)
    print(f"Model loaded. Bodies: {model.nbody}, Cameras: {model.ncam}")
    
    # Create renderer for RGB capture
    renderer = mj.Renderer(model, height=480, width=640)
    print(f"Created renderer: 640x480")
    
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            
            frame_count = 0
            max_frames = 100000  # Run for 500 frames
            update_interval = 0.01  # Update every 0.1 seconds
            last_update_time = 0
            
            print(f"\nStarting real-time simulation + visualization...")
            print("Press Ctrl+C to stop")
            
            while viewer.is_running() and frame_count < max_frames:
                current_time = time.time()
                
                # Step simulation
                mj.mj_step(model, data)
                viewer.sync()
                frame_count += 1
                
                # Update visualization at intervals
                if current_time - last_update_time >= update_interval:
                    # Generate real point cloud from camera
                    points, colors = generate_pointcloud_from_camera(
                        model, data, renderer,
                        max_points=POINT_CLOUD_CONFIG['max_points'],
                        points_per_geom=POINT_CLOUD_CONFIG['points_per_geom'],
                        downsample_method=POINT_CLOUD_CONFIG['downsample_method']
                    )
                    
                    if len(points) > 0:
                        # Log point cloud to Rerun
                        rr.set_time_sequence("frame", frame_count)
                        rr.set_time_seconds("sim_time", data.time)
                        
                        rr.log("simulation/pointcloud", rr.Points3D(
                            positions=points,
                            colors=colors,
                            radii=0.01
                        ))
                        
                        # Log simulation state (including RGB image)
                        log_simulation_state(model, data, frame_count, renderer)
                        
                        last_update_time = current_time
                        
                        if frame_count % 50 == 0:  # Print progress every 50 frames
                            print(f"📸 Frame {frame_count}/{max_frames} - Time: {data.time:.2f}s - Points: {len(points)}")
                
                time.sleep(0.005)  # Small sleep for smooth animation
            
            print(f"\n🏁 Demo completed!")
            print(f"   📊 Total frames: {frame_count}")
            print(f"   ⏱️  Final simulation time: {data.time:.3f}s")
            print(f"   ✨ Check the Rerun viewer for the complete timeline")
            
    except KeyboardInterrupt:
        print("Interrupted...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    

if __name__ == "__main__":
    main()