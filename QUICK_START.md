# Quick Start Guide - Munajjam Installation

This guide will help you install Python, set up the environment, and test the Munajjam library with GPU support.

## Prerequisites

- Windows 10/11
- NVIDIA RTX GPU (for GPU acceleration)
- Internet connection (for downloading models and dependencies)

## Step 1: Install Python

If Python is not installed:

1. **Download Python 3.10+** from [python.org](https://www.python.org/downloads/)
   - Make sure to check **"Add Python to PATH"** during installation
   - Or use Microsoft Store: `winget install Python.Python.3.12`

2. **Verify installation:**
   ```powershell
   python --version
   ```
   Should show Python 3.10 or higher.

## Step 2: Run Setup Script

Open PowerShell in the project directory and run:

```powershell
.\setup_and_install.ps1
```

This script will:
- ✓ Check Python installation
- ✓ Detect your NVIDIA GPU
- ✓ Create a virtual environment
- ✓ Install PyTorch with CUDA support
- ✓ Install all dependencies
- ✓ Install Munajjam library

## Step 3: Test Installation

After installation completes, run the test script:

```powershell
.\venv\Scripts\python.exe test_installation.py
```

This will verify:
- ✓ All packages are installed correctly
- ✓ GPU/CUDA is detected and working
- ✓ Whisper model can be loaded
- ✓ Basic functionality works

## Step 4: Activate Virtual Environment

To use the library, activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

After activation, you can use Python directly:
```powershell
python test_installation.py
```

## Manual Installation (Alternative)

If the script doesn't work, you can install manually:

### 1. Create Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install PyTorch with CUDA
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Install Munajjam
```powershell
cd munajjam
pip install -e ".[faster-whisper]"
cd ..
```

## Testing with Sample Audio

To test transcription with a real audio file:

1. Place a WAV file in the `data/` directory named `001.wav` or `surah_001.wav`
2. Run the test script - it will automatically detect and transcribe it

Or use the example script:
```powershell
python munajjam\examples\basic_usage.py data\001.wav 1
```

## Troubleshooting

### Python not found
- Make sure Python is added to PATH
- Try restarting your terminal
- Use full path: `C:\Python312\python.exe`

### CUDA not detected
- Install NVIDIA drivers from [nvidia.com](https://www.nvidia.com/drivers)
- Install CUDA Toolkit 12.1+ from [developer.nvidia.com](https://developer.nvidia.com/cuda-downloads)
- Reinstall PyTorch: `pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu121`

### Model download fails
- Check internet connection
- The model will be downloaded from HuggingFace on first use
- Model size: ~500MB-1GB

### Out of memory errors
- Use smaller model: Set `MUNAJJAM_MODEL_ID="tarteel-ai/whisper-tiny-ar-quran"`
- Use CPU: Set `MUNAJJAM_DEVICE="cpu"`
- Close other applications using GPU

## Next Steps

- Read the [README.md](README.md) for usage examples
- Check [munajjam/README.md](munajjam/README.md) for API documentation
- See [munajjam/examples/](munajjam/examples/) for code examples

## Environment Variables

You can configure the library using environment variables:

```powershell
$env:MUNAJJAM_MODEL_ID="tarteel-ai/whisper-base-ar-quran"
$env:MUNAJJAM_DEVICE="cuda"
$env:MUNAJJAM_MODEL_TYPE="faster-whisper"  # or "transformers"
```

Or create a `.env` file in the project root:
```
MUNAJJAM_MODEL_ID=tarteel-ai/whisper-base-ar-quran
MUNAJJAM_DEVICE=cuda
MUNAJJAM_MODEL_TYPE=faster-whisper
```

