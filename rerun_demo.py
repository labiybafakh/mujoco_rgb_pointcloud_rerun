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

def generate_pointcloud_from_simulation_state(model, data):
    """Generate point cloud data from current simulation state"""
    points = []
    colors = []
    
    for i in range(model.nbody):
        if i == 0:  # Skip world body
            continue
            
        pos = data.xpos[i].copy()
        
        # Generate points around each body
        for _ in range(40):  # Good amount for real-time visualization
            noise = np.random.normal(0, 0.02, 3)
            point_pos = pos + noise
            
            # Generate nice colors based on body index
            r = int(255 * (0.5 + 0.5 * np.sin(i * 1.1)))
            g = int(255 * (0.5 + 0.5 * np.sin(i * 1.7 + 2)))
            b = int(255 * (0.5 + 0.5 * np.sin(i * 2.3 + 4)))
            
            points.append(point_pos)
            colors.append([r, g, b])
    
    return np.array(points), np.array(colors, dtype=np.uint8)

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

    model_path = "config/camera_environment.xml"
    
    # Initialize Rerun
    rr.init("MuJoCo Point Cloud Demo", spawn=True)
    
    # Set up the 3D view
    rr.log("simulation", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    
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
                    # Generate and log point cloud
                    points, colors = generate_pointcloud_from_simulation_state(model, data)
                    
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