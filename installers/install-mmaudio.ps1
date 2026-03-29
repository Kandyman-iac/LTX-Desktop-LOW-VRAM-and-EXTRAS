<#
.SYNOPSIS
    Installs MMAudio into WSL Ubuntu-24.04 for use with RADICAL LTX Desktop.

.DESCRIPTION
    MMAudio is a video-to-audio AI model (~1.5 GB model weights) developed by
    Ho Kei Cheng (hkchengrex). Given a video clip and an optional text prompt it
    synthesises a matching audio track using a flow-matching diffusion model.

    This script automates the full WSL-side installation:
      - Ensures WSL and the target distro are present
      - Installs Miniconda3 inside WSL if absent
      - Clones the MMAudio repository
      - Creates a dedicated conda environment with the correct PyTorch + CUDA build
      - Installs all Python dependencies

    PREREQUISITES
    -------------
    - Windows 10/11 with WSL2 enabled  (winget install --id Microsoft.WSL)
    - Ubuntu-24.04 distro installed     (wsl --install -d Ubuntu-24.04)
    - An NVIDIA GPU with CUDA 12.x drivers installed on the host
    - At least 6 GB free VRAM and ~4 GB free disk space in WSL
    - Internet access (GitHub + PyPI + conda-forge)

    Model weights (~1.5 GB total, four files) are downloaded automatically on
    first inference — they are NOT downloaded by this script.

.PARAMETER WslDistro
    Name of the WSL distro to install into. Default: Ubuntu-24.04

.PARAMETER WslUser
    Linux username to run commands as inside WSL. Default: mike_hunt

.PARAMETER InstallDir
    Absolute Linux path where the MMAudio repo will be cloned. Default: /home/mike_hunt/MMAudio

.PARAMETER CondaEnv
    Name for the conda environment that will be created. Default: mmaudio

.EXAMPLE
    .\install-mmaudio.ps1
    # Installs with all defaults.

.EXAMPLE
    .\install-mmaudio.ps1 -WslDistro "Ubuntu-22.04" -WslUser "bob" -InstallDir "/home/bob/MMAudio"
#>

[CmdletBinding()]
param(
    [string]$WslDistro  = "Ubuntu-24.04",
    [string]$WslUser    = "mike_hunt",
    [string]$InstallDir = "/home/mike_hunt/MMAudio",
    [string]$CondaEnv   = "mmaudio"
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Banner {
    Write-Host ""
    Write-Host "  ╔══════════════════════════════════════════════════════╗" -ForegroundColor Magenta
    Write-Host "  ║       RADICAL LTX Desktop — MMAudio Installer        ║" -ForegroundColor Magenta
    Write-Host "  ╚══════════════════════════════════════════════════════╝" -ForegroundColor Magenta
    Write-Host ""
}

function Write-Step {
    param([int]$Number, [string]$Title)
    Write-Host ""
    Write-Host "  [$Number/5] $Title" -ForegroundColor Cyan
    Write-Host "  $('─' * 54)" -ForegroundColor DarkGray
}

function Write-OK   { param([string]$Msg) Write-Host "  [OK]  $Msg" -ForegroundColor Green  }
function Write-Info { param([string]$Msg) Write-Host "  [..] $Msg"  -ForegroundColor Yellow }
function Write-Err  { param([string]$Msg) Write-Host "  [!!] $Msg"  -ForegroundColor Red    }

# Runs a shell command inside the WSL distro as $WslUser.
# Streams output live and throws on non-zero exit code.
function Invoke-Wsl {
    param([string]$Cmd)
    Write-Host "  >>> $Cmd" -ForegroundColor DarkGray
    wsl -d $WslDistro -u $WslUser bash -c $Cmd
    if ($LASTEXITCODE -ne 0) {
        throw "WSL command failed (exit $LASTEXITCODE): $Cmd"
    }
}

# Same as Invoke-Wsl but captures and returns stdout silently (no live stream).
function Invoke-WslCapture {
    param([string]$Cmd)
    $out = wsl -d $WslDistro -u $WslUser bash -c $Cmd 2>$null
    return $out
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

Write-Banner

# ── Step 1: Verify WSL + distro ────────────────────────────────────────────
Write-Step 1 "Checking WSL installation and distro"

$wslExe = Get-Command wsl -ErrorAction SilentlyContinue
if (-not $wslExe) {
    Write-Err "WSL is not installed or not on PATH."
    Write-Host ""
    Write-Host "  Install WSL with:" -ForegroundColor White
    Write-Host "    winget install --id Microsoft.WSL --source winget" -ForegroundColor Yellow
    Write-Host "  Then install the Ubuntu-24.04 distro:" -ForegroundColor White
    Write-Host "    wsl --install -d Ubuntu-24.04" -ForegroundColor Yellow
    Write-Host "  Reboot and re-run this script." -ForegroundColor White
    exit 1
}
Write-OK "wsl.exe found at $($wslExe.Source)"

# Get list of installed distros (strip VT escape codes that WSL sometimes emits)
$distroList = (wsl --list --quiet 2>$null) -replace '\x1B\[[0-9;]*m', '' |
              ForEach-Object { $_.Trim() } |
              Where-Object { $_ -ne '' }

if ($distroList -notcontains $WslDistro) {
    Write-Err "Distro '$WslDistro' is not installed."
    Write-Host ""
    Write-Host "  Available distros:" -ForegroundColor White
    $distroList | ForEach-Object { Write-Host "    - $_" -ForegroundColor DarkGray }
    Write-Host ""
    Write-Host "  Install '$WslDistro' with:" -ForegroundColor White
    Write-Host "    wsl --install -d $WslDistro" -ForegroundColor Yellow
    Write-Host "  Then create user '$WslUser' inside it before re-running." -ForegroundColor White
    exit 1
}
Write-OK "Distro '$WslDistro' is installed"

# Confirm the user exists inside WSL
$userCheck = Invoke-WslCapture "id -u $WslUser 2>&1"
if ($LASTEXITCODE -ne 0 -or $userCheck -match "no such user") {
    Write-Err "Linux user '$WslUser' does not exist in '$WslDistro'."
    Write-Host "  Create it first:  wsl -d $WslDistro -- adduser $WslUser" -ForegroundColor Yellow
    exit 1
}
Write-OK "Linux user '$WslUser' exists (uid=$($userCheck.Trim()))"

# ── Step 2: Ensure Miniconda3 is available inside WSL ─────────────────────
Write-Step 2 "Checking for conda / Miniconda3 in WSL"

$condaCheck = Invoke-WslCapture "bash -lc 'conda --version 2>/dev/null || echo __MISSING__'"
if ($condaCheck -match "__MISSING__" -or $condaCheck -match "command not found") {
    Write-Info "conda not found — installing Miniconda3 into WSL..."

    # Download the Miniconda installer via WSL curl, install silently, then init
    $minicondaUrl = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    $minicondaInstaller = "/tmp/Miniconda3-latest-Linux-x86_64.sh"

    Invoke-Wsl "curl -fsSL '$minicondaUrl' -o '$minicondaInstaller'"
    Invoke-Wsl "bash '$minicondaInstaller' -b -p /home/$WslUser/miniconda3"
    Invoke-Wsl "rm -f '$minicondaInstaller'"

    # Initialise conda for bash (writes to ~/.bashrc)
    Invoke-Wsl "/home/$WslUser/miniconda3/bin/conda init bash"

    Write-OK "Miniconda3 installed at /home/$WslUser/miniconda3"
} else {
    Write-OK "conda found: $($condaCheck.Trim())"
}

# Convenience alias — all subsequent conda calls go through the full path so
# we don't rely on .bashrc being sourced in non-interactive shells.
$condaBin = Invoke-WslCapture "bash -lc 'which conda 2>/dev/null || echo /home/$WslUser/miniconda3/bin/conda'"
$condaBin  = $condaBin.Trim()
Write-Info "Using conda binary: $condaBin"

# ── Step 3: Clone the MMAudio repository ──────────────────────────────────
Write-Step 3 "Cloning MMAudio repository"

$repoUrl = "https://github.com/hkchengrex/MMAudio"
$gitCheck = Invoke-WslCapture "test -d '$InstallDir/.git' && echo EXISTS || echo MISSING"

if ($gitCheck.Trim() -eq "EXISTS") {
    Write-OK "Repo already cloned at $InstallDir — skipping clone"
    Write-Info "Pulling latest changes..."
    Invoke-Wsl "cd '$InstallDir' && git pull --ff-only"
} else {
    Write-Info "Cloning $repoUrl into $InstallDir ..."
    # Ensure the parent directory exists
    $parentDir = ($InstallDir -split "/")[0..($InstallDir.Split("/").Count - 2)] -join "/"
    Invoke-Wsl "mkdir -p '$parentDir'"
    Invoke-Wsl "git clone '$repoUrl' '$InstallDir'"
    Write-OK "Repository cloned successfully"
}

# ── Step 4: Create conda env and install dependencies ─────────────────────
Write-Step 4 "Setting up conda environment '$CondaEnv'"

$envCheck = Invoke-WslCapture "$condaBin env list 2>/dev/null | grep -w '$CondaEnv' | head -1"
if ($envCheck -match $CondaEnv) {
    Write-OK "Conda environment '$CondaEnv' already exists — skipping creation"
} else {
    Write-Info "Creating conda environment '$CondaEnv' with Python 3.10..."
    Invoke-Wsl "$condaBin create -y -n '$CondaEnv' python=3.10"
    Write-OK "Environment '$CondaEnv' created"
}

# Install PyTorch 2.5.1 + CUDA 12.1
Write-Info "Installing PyTorch 2.5.1 + CUDA 12.1 (this may take several minutes)..."
Invoke-Wsl "$condaBin run -n '$CondaEnv' pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121"
Write-OK "PyTorch 2.5.1+cu121 installed"

# Install MMAudio package itself (editable mode so the repo IS the package)
Write-Info "Installing MMAudio and its dependencies..."
Invoke-Wsl "$condaBin run -n '$CondaEnv' pip install -e '$InstallDir'"

# Install any extra requirements listed in requirements.txt if present
$reqCheck = Invoke-WslCapture "test -f '$InstallDir/requirements.txt' && echo YES || echo NO"
if ($reqCheck.Trim() -eq "YES") {
    Write-Info "Installing extra requirements from requirements.txt..."
    Invoke-Wsl "$condaBin run -n '$CondaEnv' pip install -r '$InstallDir/requirements.txt'"
    Write-OK "requirements.txt dependencies installed"
} else {
    Write-Info "No requirements.txt found — skipping extra deps"
}

# Quick sanity check — can we import torch inside the env?
Write-Info "Running import sanity check..."
Invoke-Wsl "$condaBin run -n '$CondaEnv' python -c 'import torch; print(\"torch\", torch.__version__, \"CUDA\", torch.cuda.is_available())'"
Write-OK "Python environment is functional"

# ── Step 5: Success ────────────────────────────────────────────────────────
Write-Step 5 "Installation complete"

Write-Host ""
Write-Host "  MMAudio has been installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "  Install directory : $InstallDir" -ForegroundColor White
Write-Host "  Conda environment : $CondaEnv"   -ForegroundColor White
Write-Host "  WSL distro        : $WslDistro"  -ForegroundColor White
Write-Host ""
Write-Host "  MODEL WEIGHTS (~1.5 GB)" -ForegroundColor Yellow
Write-Host "  ─────────────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host "  Four checkpoint files are downloaded automatically on" -ForegroundColor White
Write-Host "  first inference.  They are NOT downloaded by this"    -ForegroundColor White
Write-Host "  script.  Ensure you have a stable internet connection" -ForegroundColor White
Write-Host "  and ~1.5 GB free disk space in WSL when you first run." -ForegroundColor White
Write-Host ""
Write-Host "  NEXT STEPS" -ForegroundColor Yellow
Write-Host "  ─────────────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host "  1. Open RADICAL LTX Desktop." -ForegroundColor White
Write-Host "  2. In Settings, set MMAudio WSL distro to: $WslDistro" -ForegroundColor White
Write-Host "  3. Confirm conda env name is:              $CondaEnv" -ForegroundColor White
Write-Host "  4. Generate a video, then click 'Add Audio'" -ForegroundColor White
Write-Host "     — weights will download on that first call." -ForegroundColor White
Write-Host ""
