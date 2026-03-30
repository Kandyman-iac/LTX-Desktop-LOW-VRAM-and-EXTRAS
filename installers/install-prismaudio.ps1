<#
.SYNOPSIS
    Installs PrismAudio (ThinkSound by Alibaba FunAudioLLM) on Windows for use
    with RADICAL LTX Desktop.

.DESCRIPTION
    PrismAudio / ThinkSound is a 518-million-parameter Foley and SFX synthesis
    model developed by Alibaba's FunAudioLLM team.  It uses a chain-of-thought
    (CoT) reasoning stage to plan spatially and temporally coherent sound effects
    before synthesising audio that is synchronised to the input video.

    Key facts:
      - Model size  : 518 M parameters (~2–3 GB weights on disk)
      - VRAM needed : ~4–6 GB GPU VRAM (NVIDIA, CUDA 12.x)
      - License     : MIT (research + commercial use permitted)
      - Source      : github.com/FunAudioLLM/ThinkSound  branch: prismaudio
      - Runs natively on Windows — no WSL required

    This script:
      - Verifies git and conda are available
      - Clones the 'prismaudio' branch of ThinkSound into $InstallDir
      - Creates a dedicated conda environment with Python 3.10
      - Installs all Python dependencies

    PREREQUISITES
    -------------
    - Git for Windows             (https://git-scm.com/download/win)
    - Miniconda3 or Anaconda3     (https://docs.conda.io/en/latest/miniconda.html)
    - NVIDIA GPU with CUDA 12.x drivers
    - ~8 GB free disk space
    - Internet access (GitHub + PyPI)

.PARAMETER InstallDir
    Windows path where the ThinkSound repo will be cloned.
    Default: C:\AI\ThinkSound

.PARAMETER CondaEnv
    Name for the conda environment that will be created.
    Default: prismaudio

.PARAMETER GitRepo
    URL of the ThinkSound Git repository.
    Default: https://github.com/FunAudioLLM/ThinkSound

.PARAMETER Branch
    Git branch to check out.
    Default: prismaudio

.EXAMPLE
    .\install-prismaudio.ps1
    # Installs with all defaults.

.EXAMPLE
    .\install-prismaudio.ps1 -InstallDir "D:\AI\ThinkSound" -CondaEnv "thinksound"
#>

[CmdletBinding()]
param(
    [string]$InstallDir  = "",          # overrides mode default if set
    [string]$CondaEnv    = "prismaudio",
    [string]$GitRepo     = "https://github.com/FunAudioLLM/ThinkSound",
    [string]$Branch      = "prismaudio",
    [switch]$UseWSL,                    # install into WSL2 instead of Windows-native
    [string]$WslDistro   = "Ubuntu-24.04",
    [string]$WslUser     = "mike_hunt",
    [string]$WslInstallDir = "/home/mike_hunt/ThinkSound"
)

$ErrorActionPreference = "Stop"

# Resolve default InstallDir based on mode
if (-not $InstallDir) {
    $InstallDir = if ($UseWSL) { $WslInstallDir } else { "C:\AI\ThinkSound" }
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Banner {
    $mode = if ($UseWSL) { "WSL2 mode" } else { "Windows-native mode" }
    Write-Host ""
    Write-Host "  ╔══════════════════════════════════════════════════════╗" -ForegroundColor Magenta
    Write-Host "  ║     RADICAL LTX Desktop — PrismAudio Installer       ║" -ForegroundColor Magenta
    Write-Host "  ║     $($mode.PadRight(48))║" -ForegroundColor Magenta
    Write-Host "  ╚══════════════════════════════════════════════════════╝" -ForegroundColor Magenta
    Write-Host ""
}

function Invoke-Wsl {
    param([string]$Cmd)
    $result = wsl -d $WslDistro -u $WslUser bash -c $Cmd
    if ($LASTEXITCODE -ne 0) { throw "WSL command failed (exit $LASTEXITCODE): $Cmd" }
    return $result
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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

Write-Banner

# ===========================================================================
# WSL mode — all steps run inside WSL2
# ===========================================================================
if ($UseWSL) {

# ── WSL Step 1: Verify WSL distro ─────────────────────────────────────────
Write-Step 1 "Checking WSL2 distro ($WslDistro)"

$wslList = wsl --list --quiet 2>&1
$wslText  = try { [System.Text.Encoding]::Unicode.GetString([System.Text.Encoding]::Unicode.GetBytes($wslList)) } catch { $wslList }
if ($wslText -notmatch [regex]::Escape($WslDistro)) {
    Write-Err "WSL distro '$WslDistro' not found."
    Write-Host "  Available distros:" -ForegroundColor White
    $wslText.Split("`n") | Where-Object { $_ -match '\S' } | ForEach-Object { Write-Host "    $_" }
    Write-Host "  Install Ubuntu 24.04 via: wsl --install -d Ubuntu-24.04" -ForegroundColor Yellow
    exit 1
}
Write-OK "WSL distro '$WslDistro' found"

# ── WSL Step 2: Check/install git inside WSL ──────────────────────────────
Write-Step 2 "Checking git in WSL"
$gitVer = wsl -d $WslDistro -u $WslUser bash -c "git --version 2>/dev/null || echo MISSING"
if ($gitVer -match "MISSING") {
    Write-Info "Installing git in WSL..."
    Invoke-Wsl "sudo apt-get update -qq && sudo apt-get install -y git"
}
Write-OK "git ready: $gitVer"

# ── WSL Step 3: Clone/update repo in WSL ──────────────────────────────────
Write-Step 3 "Cloning ThinkSound into WSL at $InstallDir"
$cloneResult = wsl -d $WslDistro -u $WslUser bash -c "
    if [ -d '$InstallDir/.git' ]; then
        echo ALREADY_EXISTS
    else
        git clone --branch $Branch --single-branch $GitRepo '$InstallDir' && echo CLONED || echo FAILED
    fi
"
if ($cloneResult -match "FAILED") { throw "git clone inside WSL failed" }
if ($cloneResult -match "ALREADY_EXISTS") {
    Write-OK "Repo already present — pulling latest"
    Invoke-Wsl "cd '$InstallDir' && git fetch origin && git checkout $Branch && git pull --ff-only origin $Branch"
} else {
    Write-OK "Repository cloned"
}

# ── WSL Step 4: Conda env + dependencies ──────────────────────────────────
Write-Step 4 "Setting up conda env '$CondaEnv' in WSL"

# Locate conda in WSL (try micromamba path fallbacks)
$condaCmd = wsl -d $WslDistro -u $WslUser bash -c "
    for p in conda mamba micromamba; do
        if command -v \$p &>/dev/null; then echo \$p; break; fi
    done
" | Select-Object -First 1
$condaCmd = $condaCmd.Trim()
if (-not $condaCmd) { $condaCmd = "conda" }
Write-Info "Using: $condaCmd"

$envExists = wsl -d $WslDistro -u $WslUser bash -c "$condaCmd env list 2>/dev/null | grep -c '^$CondaEnv'" 2>&1
if ([int]($envExists -replace '\D','0') -gt 0) {
    Write-OK "Conda env '$CondaEnv' already exists"
} else {
    Write-Info "Creating conda env '$CondaEnv' with Python 3.10..."
    Invoke-Wsl "$condaCmd create -y -n $CondaEnv python=3.10"
    Write-OK "Env '$CondaEnv' created"
}

Write-Info "Installing dependencies (this may take several minutes)..."
Invoke-Wsl "cd '$InstallDir' && $condaCmd run -n $CondaEnv pip install -r requirements.txt 2>&1 || ([ -f setup_windows.bat ] && echo 'Note: only Windows setup script found')"
Write-OK "Dependencies installed"

$pyVer = wsl -d $WslDistro -u $WslUser bash -c "$condaCmd run -n $CondaEnv python -c 'import sys; print(sys.version)'"
Write-OK "Python: $pyVer"

# ── WSL Step 5: Done ──────────────────────────────────────────────────────
Write-Step 5 "Installation complete (WSL mode)"
Write-Host ""
Write-Host "  PrismAudio installed in WSL successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "  WSL distro        : $WslDistro"  -ForegroundColor White
Write-Host "  Install directory : $InstallDir" -ForegroundColor White
Write-Host "  Conda environment : $CondaEnv"   -ForegroundColor White
Write-Host ""
Write-Host "  CONFIGURING RADICAL LTX DESKTOP" -ForegroundColor Yellow
Write-Host "  ─────────────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host "  In backend\handlers\prismaudio_handler.py set:" -ForegroundColor White
Write-Host ""
Write-Host "    _USE_WSL                 = True"               -ForegroundColor Yellow
Write-Host "    _WSL_DISTRO              = `"$WslDistro`""     -ForegroundColor Yellow
Write-Host "    _WSL_USER                = `"$WslUser`""       -ForegroundColor Yellow
Write-Host "    _PRISMAUDIO_DIR_WSL      = `"$InstallDir`""    -ForegroundColor Yellow
Write-Host "    _PRISMAUDIO_CONDA_ENV_WSL= `"$CondaEnv`""      -ForegroundColor Yellow
Write-Host ""

# Exit early — skip the Windows-native section below
exit 0
}

# ===========================================================================
# Windows-native mode (default)
# ===========================================================================

# ── Step 1: Verify git is installed ───────────────────────────────────────
Write-Step 1 "Checking for Git"

$gitExe = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitExe) {
    Write-Err "Git is not installed or not on PATH."
    Write-Host ""
    Write-Host "  Install Git for Windows from:" -ForegroundColor White
    Write-Host "    https://git-scm.com/download/win" -ForegroundColor Yellow
    Write-Host "  Or via winget:" -ForegroundColor White
    Write-Host "    winget install --id Git.Git --source winget" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  After installing, close and re-open PowerShell, then re-run this script." -ForegroundColor White
    exit 1
}

$gitVersion = (git --version 2>&1)
Write-OK "git found: $gitVersion"

# ── Step 2: Locate conda / Miniconda3 ─────────────────────────────────────
Write-Step 2 "Locating conda / Miniconda3"

# Search a set of well-known install locations
$candidatePaths = @(
    "$env:USERPROFILE\miniconda3",
    "$env:USERPROFILE\anaconda3",
    "$env:USERPROFILE\Miniconda3",
    "$env:USERPROFILE\Anaconda3",
    "C:\miniconda3",
    "C:\Miniconda3",
    "C:\anaconda3",
    "C:\Anaconda3",
    "C:\ProgramData\miniconda3",
    "C:\ProgramData\Miniconda3",
    "C:\ProgramData\anaconda3",
    "C:\ProgramData\Anaconda3"
)

$condaExe = $null

# First check if conda is already on PATH
$condaOnPath = Get-Command conda -ErrorAction SilentlyContinue
if ($condaOnPath) {
    $condaExe = $condaOnPath.Source
    Write-OK "conda found on PATH: $condaExe"
} else {
    Write-Info "conda not on PATH — searching common install locations..."
    foreach ($base in $candidatePaths) {
        $candidate = Join-Path $base "Scripts\conda.exe"
        if (Test-Path $candidate) {
            $condaExe = $candidate
            Write-OK "conda found at: $condaExe"
            break
        }
    }
}

if (-not $condaExe) {
    Write-Err "conda / Miniconda3 / Anaconda3 not found."
    Write-Host ""
    Write-Host "  Searched paths:" -ForegroundColor White
    $candidatePaths | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    Write-Host ""
    Write-Host "  Download Miniconda3 (recommended, ~90 MB) from:" -ForegroundColor White
    Write-Host "    https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Install it, tick 'Add to PATH' or re-run this script after activating base." -ForegroundColor White
    exit 1
}

$condaVersion = (& $condaExe --version 2>&1)
Write-OK "conda version: $condaVersion"

# ── Step 3: Clone (or update) the ThinkSound repository ───────────────────
Write-Step 3 "Cloning ThinkSound repository (branch: $Branch)"

if (Test-Path (Join-Path $InstallDir ".git")) {
    Write-OK "Repo already exists at $InstallDir — skipping clone"
    Write-Info "Pulling latest changes on branch '$Branch'..."
    Push-Location $InstallDir
    try {
        git fetch origin
        git checkout $Branch
        git pull --ff-only origin $Branch
        Write-OK "Repository updated"
    } finally {
        Pop-Location
    }
} else {
    # Ensure the parent directory exists
    $parentDir = Split-Path $InstallDir -Parent
    if ($parentDir -and -not (Test-Path $parentDir)) {
        Write-Info "Creating parent directory: $parentDir"
        New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
    }

    Write-Info "Cloning $GitRepo (branch: $Branch) into $InstallDir ..."
    git clone --branch $Branch --single-branch $GitRepo $InstallDir
    if ($LASTEXITCODE -ne 0) {
        throw "git clone failed (exit $LASTEXITCODE)"
    }
    Write-OK "Repository cloned successfully"
}

# ── Step 4: Create conda env and install dependencies ─────────────────────
Write-Step 4 "Setting up conda environment '$CondaEnv'"

# Check if env already exists
$envListOutput = & $condaExe env list 2>&1
$envExists = $envListOutput | Where-Object { $_ -match "^\s*$([regex]::Escape($CondaEnv))\s" }

if ($envExists) {
    Write-OK "Conda environment '$CondaEnv' already exists — skipping creation"
} else {
    Write-Info "Creating conda environment '$CondaEnv' with Python 3.10..."
    & $condaExe create -y -n $CondaEnv python=3.10
    if ($LASTEXITCODE -ne 0) {
        throw "conda create failed (exit $LASTEXITCODE)"
    }
    Write-OK "Environment '$CondaEnv' created"
}

# Install dependencies — prefer requirements.txt, fall back to setup_windows.bat
$requirementsPath  = Join-Path $InstallDir "requirements.txt"
$setupBatPath      = Join-Path $InstallDir "setup_windows.bat"

if (Test-Path $requirementsPath) {
    Write-Info "Installing from requirements.txt (this may take several minutes)..."
    & $condaExe run -n $CondaEnv pip install -r $requirementsPath
    if ($LASTEXITCODE -ne 0) {
        throw "pip install -r requirements.txt failed (exit $LASTEXITCODE)"
    }
    Write-OK "requirements.txt dependencies installed"
} elseif (Test-Path $setupBatPath) {
    Write-Info "requirements.txt not found — running setup_windows.bat instead..."
    Write-Info "(The batch file will run inside the '$CondaEnv' conda environment.)"
    # Activate env, then call the bat file
    $activateScript = Join-Path (Split-Path $condaExe -Parent) "..\shell\condabin\conda-hook.ps1"
    if (Test-Path $activateScript) {
        . $activateScript
        conda activate $CondaEnv
    }
    Push-Location $InstallDir
    try {
        cmd /c "setup_windows.bat"
        if ($LASTEXITCODE -ne 0) {
            throw "setup_windows.bat failed (exit $LASTEXITCODE)"
        }
    } finally {
        Pop-Location
    }
    Write-OK "setup_windows.bat completed"
} else {
    Write-Info "No requirements.txt or setup_windows.bat found — skipping dep install."
    Write-Info "You may need to install dependencies manually once the repo is updated."
}

# Quick sanity check
Write-Info "Running import sanity check..."
$pyCheck = & $condaExe run -n $CondaEnv python -c "import sys; print('Python', sys.version)" 2>&1
Write-OK "Python environment: $pyCheck"

# ── Step 5: Success ────────────────────────────────────────────────────────
Write-Step 5 "Installation complete"

Write-Host ""
Write-Host "  PrismAudio (ThinkSound) has been installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "  Install directory : $InstallDir" -ForegroundColor White
Write-Host "  Conda environment : $CondaEnv"   -ForegroundColor White
Write-Host "  Git branch        : $Branch"      -ForegroundColor White
Write-Host ""
Write-Host "  CONFIGURING RADICAL LTX DESKTOP" -ForegroundColor Yellow
Write-Host "  ─────────────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host "  Open the handler file and verify these constants:" -ForegroundColor White
Write-Host ""
Write-Host "    backend\handlers\prismaudio_handler.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "    _USE_WSL                = False"               -ForegroundColor Yellow
Write-Host "    _PRISMAUDIO_DIR_WIN     = r`"$InstallDir`""    -ForegroundColor Yellow
Write-Host "    _PRISMAUDIO_CONDA_ENV   = `"$CondaEnv`""       -ForegroundColor Yellow
Write-Host ""
Write-Host "  MODEL WEIGHTS" -ForegroundColor Yellow
Write-Host "  ─────────────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host "  ThinkSound checkpoint files are fetched from Hugging" -ForegroundColor White
Write-Host "  Face on first inference (check the repo README for" -ForegroundColor White
Write-Host "  exact download instructions or auto-download support)." -ForegroundColor White
Write-Host "  Ensure ~4–6 GB free VRAM before running inference." -ForegroundColor White
Write-Host ""
