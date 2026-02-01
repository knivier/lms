# Kinera

Kinera is a real-time AI-powered fitness tracking application that uses computer vision to monitor your workout form, count reps, and provide instant feedback. Built with Tauri, React, and Python MediaPipe.

## Features

- Real-time pose detection and skeleton tracking via webcam
- Automatic rep counting for multiple exercises (squats, push-ups, bicep curls)
- Live form feedback based on joint angles and range of motion
- Session tracking with rest timers and set management
- Quality scoring for each rep using a trained neural network model
- Fullscreen workout interface with modern gradient UI

## Requirements

### System Dependencies
- Node.js 18+ and npm
- Rust (latest stable) and Cargo
- Python 3.8+
- Webcam

### Python Dependencies
Install Python packages from the root directory:
```bash
pip install -r requirements.txt
```

Required packages include:
- mediapipe (pose detection)
- opencv-python (camera and image processing)
- numpy, scipy (data processing)
- torch (neural network inference)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Kinera
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install frontend dependencies:
```bash
cd front
npm install
```

## Running the Application

### Development Mode

From the `front` directory, run:
```bash
cd front
npm run tauri dev
```

This will:
- Start the Vite development server
- Build the Tauri app
- Launch Kinera in fullscreen mode
- Automatically connect to your webcam when you start a workout

### Production Build

To create a production build:
```bash
cd front
npm run tauri build
```

The compiled application will be in `front/src-tauri/target/release/`.

## Camera Configuration

Camera settings can be configured in `cv/config.yaml`:

```yaml
camera_id: 0  # Change to your camera index (0, 1, 2, etc.)
camera_width: 1920
camera_height: 1080
camera_fps: 30
```

To test your camera setup, run the CV viewer directly:
```bash
python cv/cv-view.py
```

## Project Structure

```
Kinera/
├── front/                      # Tauri + React frontend
│   ├── src/
│   │   ├── pages/             # React components (WorkoutPicker, LiveSession, Settings)
│   │   └── App.jsx            # Main app with routing and na vigation
│   ├── src-tauri/             # Rust backend
│   │   └── src/lib.rs         # Tauri commands for CV integration
│   └── package.json
├── cv/                        # Computer vision pipeline
│   ├── cv.py                  # Main pose detection and angle calculation
│   ├── datahandler.py         # Rep detection and feedback generation
│   ├── cv-view.py            # Standalone CV viewer for testing
│   ├── config.yaml           # Camera and detection settings
│   ├── reps_log.jsonl        # Rep history (appended during sessions)
│   └── session_live.json     # Live metrics output
├── quantprocess/             # Machine learning models
│   ├── model.py              # Neural network architecture
│   ├── crouch_model.pth      # Trained quality scoring model
│   └── RepTracker.py         # Rep quality analysis
├── training-data/            # Exercise datasets for model training
├── workout_id.json           # Current workout state (written by frontend)
├── session_config.json       # Optional background scripts configuration
├── requirements.txt          # Python dependencies
└── README.md
```

## How It Works

1. **Frontend (Tauri + React)**: Provides the workout UI, manages sessions, and displays metrics
2. **CV Pipeline (Python)**: Captures camera frames, runs MediaPipe pose detection, calculates joint angles
3. **Rep Detection (datahandler.py)**: State machine that detects full rep cycles based on angle thresholds
4. **Feedback Generation**: Compares angles and tempo against target values, provides form feedback
5. **IPC Communication**: Frontend invokes Rust commands which start/stop the Python CV process and read rep data

### Supported Exercises

- **Squats**: Tracks knee angles (90-150°), depth, and extension
- **Push-ups**: Tracks elbow angles (100-150°), chest touch, and lockout
- **Bicep Curls**: Tracks elbow angles (70-135°), curl depth, and extension

## Session Configuration

Optional background scripts can be configured in `session_config.json` at the root:

```json
{
  "session_scripts": [
    "python ProcessedData/synthesizer.py",
    "python quantprocess/RepTracker.py"
  ]
}
```

These scripts start when you begin a workout and stop when you end it. Leave the array empty `[]` if you only want the CV pipeline.

## Troubleshooting

### Camera not detected
- Check camera permissions in your OS
- Verify camera_id in `cv/config.yaml` (try 0, 1, or 2)
- Test with: `python cv/cv-view.py`

### No reps detected
- Check terminal output for angle values and state machine transitions
- Adjust thresholds in `cv/datahandler.py` WORKOUT_TO_PARAMETERS
- Ensure you're in camera view and MediaPipe can see your full body

### Build errors
- Make sure Rust toolchain is up to date: `rustup update`
- Clear and reinstall: `cd front && rm -rf node_modules package-lock.json && npm install`

## Development

The app uses:
- **Frontend**: React 19, Material-UI, React Router
- **Backend**: Tauri 2, Rust
- **CV**: Python 3, MediaPipe, OpenCV
- **ML**: PyTorch for rep quality scoring

To modify detection parameters, edit `cv/datahandler.py` WORKOUT_TO_PARAMETERS.
To add new exercises, add entries to WORKOUT_TO_PARAMETERS and update the frontend workout list.

## License

See LICENSE file for details.
