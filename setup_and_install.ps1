# Munajjam Setup and Installation Script for Windows
# This script sets up Python, virtual environment, and installs all dependencies with GPU support

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Munajjam Setup and Installation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python installation
Write-Host "Step 1: Checking Python installation..." -ForegroundColor Yellow
$pythonCmd = $null

# Try different Python commands
$pythonCommands = @("python", "python3", "py")
foreach ($cmd in $pythonCommands) {
    try {
        $version = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = $cmd
            Write-Host "   ✓ Found Python: $version" -ForegroundColor Green
            break
        }
    } catch {
        continue
    }
}

if (-not $pythonCmd) {
    Write-Host "   ✗ Python not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python 3.10 or higher:" -ForegroundColor Yellow
    Write-Host "   1. Download from: https://www.python.org/downloads/" -ForegroundColor White
    Write-Host "   2. Make sure to check 'Add Python to PATH' during installation" -ForegroundColor White
    Write-Host "   3. Restart your terminal and run this script again" -ForegroundColor White
    Write-Host ""
    Write-Host "Or install via Microsoft Store:" -ForegroundColor Yellow
    Write-Host "   winget install Python.Python.3.12" -ForegroundColor White
    exit 1
}

# Check Python version
$versionOutput = & $pythonCmd --version 2>&1
$versionMatch = $versionOutput -match "Python (\d+)\.(\d+)"
if ($versionMatch) {
    $majorVersion = [int]$matches[1]
    $minorVersion = [int]$matches[2]
    if ($majorVersion -lt 3 -or ($majorVersion -eq 3 -and $minorVersion -lt 10)) {
        Write-Host "   ✗ Python 3.10+ required. Found: $versionOutput" -ForegroundColor Red
        exit 1
    }
}

# Step 2: Check CUDA availability (for GPU support)
Write-Host ""
Write-Host "Step 2: Checking CUDA availability..." -ForegroundColor Yellow
$cudaAvailable = $false
try {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        $nvidiaOutput = & nvidia-smi --query-gpu=name --format=csv,noheader 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✓ NVIDIA GPU detected: $nvidiaOutput" -ForegroundColor Green
            $cudaAvailable = $true
        }
    }
} catch {
    Write-Host "   ⚠ NVIDIA GPU not detected or nvidia-smi not available" -ForegroundColor Yellow
    Write-Host "   Will install CPU version (you can install CUDA PyTorch later)" -ForegroundColor Yellow
}

# Step 3: Create virtual environment
Write-Host ""
Write-Host "Step 3: Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "   ⚠ Virtual environment already exists, removing it..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force venv
    Write-Host "   ✓ Removed existing virtual environment" -ForegroundColor Green
}

if (-not (Test-Path "venv")) {
    & $pythonCmd -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "   ✗ Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
    Write-Host "   ✓ Virtual environment created" -ForegroundColor Green
}

# Step 4: Activate virtual environment and upgrade pip
Write-Host ""
Write-Host "Step 4: Activating virtual environment and upgrading pip..." -ForegroundColor Yellow
$venvPython = "venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "   ✗ Virtual environment Python not found" -ForegroundColor Red
    exit 1
}

& $venvPython -m pip install --upgrade pip
Write-Host "   ✓ pip upgraded" -ForegroundColor Green

# Step 5: Install PyTorch with CUDA support
Write-Host ""
Write-Host "Step 5: Installing PyTorch with CUDA support..." -ForegroundColor Yellow
if ($cudaAvailable) {
    Write-Host "   Installing PyTorch with CUDA 12.1 support..." -ForegroundColor White
    & $venvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
} else {
    Write-Host "   Installing PyTorch (CPU version)..." -ForegroundColor White
    Write-Host "   Note: You can install CUDA version later if you have NVIDIA GPU" -ForegroundColor Yellow
    & $venvPython -m pip install torch torchvision torchaudio
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "   ✗ Failed to install PyTorch" -ForegroundColor Red
    exit 1
}
Write-Host "   ✓ PyTorch installed" -ForegroundColor Green

# Step 6: Install project dependencies
Write-Host ""
Write-Host "Step 6: Installing project dependencies..." -ForegroundColor Yellow
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "   ✗ Failed to install dependencies" -ForegroundColor Red
    exit 1
}
Write-Host "   ✓ Dependencies installed" -ForegroundColor Green

# Step 7: Install munajjam in development mode
Write-Host ""
Write-Host "Step 7: Installing munajjam library in development mode..." -ForegroundColor Yellow
Set-Location munajjam
& ..\venv\Scripts\python.exe -m pip install -e ".[faster-whisper]"
if ($LASTEXITCODE -ne 0) {
    Write-Host "   ✗ Failed to install munajjam" -ForegroundColor Red
    Set-Location ..
    exit 1
}
Set-Location ..
Write-Host "   ✓ Munajjam library installed" -ForegroundColor Green

# Step 8: Verify installation
Write-Host ""
Write-Host "Step 8: Verifying installation..." -ForegroundColor Yellow
& $venvPython -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
& $venvPython -c "import munajjam; print(f'Munajjam version: {munajjam.__version__ if hasattr(munajjam, \"__version__\") else \"installed\"}')"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installation Complete! ✓" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the virtual environment, run:" -ForegroundColor Yellow
Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To test the installation, run:" -ForegroundColor Yellow
Write-Host "   .\venv\Scripts\python.exe test_installation.py" -ForegroundColor White
Write-Host ""

