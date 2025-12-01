# MuJoCo Point Cloud Demo

Real-time point cloud visualization from MuJoCo simulations using Rerun.

## What it does

- Runs MuJoCo physics simulation in background
- Captures camera feed from the simulation  
- Generates point clouds from object positions
- Shows everything in a web viewer with timeline controls

## Requirements

- Python 3.13
- uv (package manager)
- macOS/Linux/Windows

Main dependencies:
- mujoco
- rerun-sdk
- numpy
- opencv-python

## Setup

```bash
uv sync
```

## Run

```bash
uv run mjpython rerun_demo.py
```

Opens a web browser with:
- Live camera feed from simulation
- 3D point cloud visualization
- Timeline to scrub through the data

## Files

- `rerun_demo.py` - main demo
- `config/camera_environment.xml` - MuJoCo scene setup

That's it. No fancy stuff, just works.