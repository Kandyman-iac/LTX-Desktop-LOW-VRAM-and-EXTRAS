<#
.SYNOPSIS
    Meta-installer for RADICAL LTX Desktop optional AI tools.

.DESCRIPTION
    Runs one or more tool-specific installer scripts (install-mmaudio.ps1,
    install-prismaudio.ps1) with a guided menu or via command-line switches.

    Each tool installer is invoked in a child process.  Failure of one tool
    does NOT abort installation of the others.  A summary table is printed
    at the end showing which tools succeeded or failed.

.PARAMETER MMAudio
    Install MMAudio (video-to-audio, WSL/Linux).

.PARAMETER PrismAudio
    Install PrismAudio / ThinkSound (Foley SFX, Windows-native).

.PARAMETER All
    Install all available tools (equivalent to -MMAudio -PrismAudio).

.EXAMPLE
    .\install-all.ps1
    # Shows an interactive menu.

.EXAMPLE
    .\install-all.ps1 -All
    # Installs every tool without prompting.

.EXAMPLE
    .\install-all.ps1 -MMAudio
    # Installs only MMAudio.
#>

[CmdletBinding()]
param(
    [switch]$MMAudio,
    [switch]$PrismAudio,
    [switch]$All
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

function Write-Banner {
    Clear-Host
    Write-Host ""
    Write-Host "  ╔════════════════════════════════════════════════════════════╗" -ForegroundColor Magenta
    Write-Host "  ║                                                            ║" -ForegroundColor Magenta
    Write-Host "  ║        ██████╗  █████╗ ██████╗ ██╗ ██████╗ █████╗ ██╗     ║" -ForegroundColor Magenta
    Write-Host "  ║        ██╔══██╗██╔══██╗██╔══██╗██║██╔════╝██╔══██╗██║     ║" -ForegroundColor Magenta
    Write-Host "  ║        ██████╔╝███████║██║  ██║██║██║     ███████║██║     ║" -ForegroundColor Magenta
    Write-Host "  ║        ██╔══██╗██╔══██║██║  ██║██║██║     ██╔══██║██║     ║" -ForegroundColor Magenta
    Write-Host "  ║        ██║  ██║██║  ██║██████╔╝██║╚██████╗██║  ██║███████╗║" -ForegroundColor Magenta
    Write-Host "  ║        ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝║" -ForegroundColor Magenta
    Write-Host "  ║                                                            ║" -ForegroundColor Cyan
    Write-Host "  ║            RADICAL LTX Desktop — Tool Installer            ║" -ForegroundColor Cyan
    Write-Host "  ║                                                            ║" -ForegroundColor Cyan
    Write-Host "  ╚════════════════════════════════════════════════════════════╝" -ForegroundColor Magenta
    Write-Host ""
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-OK      { param([string]$Msg) Write-Host "  [OK]      $Msg" -ForegroundColor Green   }
function Write-Info    { param([string]$Msg) Write-Host "  [..]      $Msg" -ForegroundColor Yellow  }
function Write-Err     { param([string]$Msg) Write-Host "  [FAILED]  $Msg" -ForegroundColor Red     }
function Write-Skip    { param([string]$Msg) Write-Host "  [SKIP]    $Msg" -ForegroundColor DarkGray }
function Write-Divider             { Write-Host "  $('─' * 60)" -ForegroundColor DarkGray }

# ---------------------------------------------------------------------------
# Tool registry — add future tools here
# ---------------------------------------------------------------------------

# Each entry: Name, Description, Script (relative to this file's directory), Switch name
$tools = @(
    [PSCustomObject]@{
        Id          = 1
        Name        = "MMAudio"
        Description = "Video-to-audio AI (WSL/Linux, ~1.5 GB weights, needs WSL2 + CUDA)"
        Script      = "install-mmaudio.ps1"
        SwitchName  = "MMAudio"
    },
    [PSCustomObject]@{
        Id          = 2
        Name        = "PrismAudio"
        Description = "Foley/SFX AI — ThinkSound 518M (Windows-native, ~4–6 GB VRAM)"
        Script      = "install-prismaudio.ps1"
        SwitchName  = "PrismAudio"
    }
)

# ---------------------------------------------------------------------------
# Determine which tools to install
# ---------------------------------------------------------------------------

Write-Banner

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Build the set of tool IDs to run
$selectedIds = [System.Collections.Generic.List[int]]::new()

if ($All) {
    # All switch — add every tool
    foreach ($t in $tools) { $selectedIds.Add($t.Id) }
    Write-Info "-All switch detected — all tools will be installed."
    Write-Host ""
} elseif ($MMAudio -or $PrismAudio) {
    # Individual switches
    if ($MMAudio)    { $selectedIds.Add(1) }
    if ($PrismAudio) { $selectedIds.Add(2) }
    Write-Info "Tools selected via command-line switches."
    Write-Host ""
} else {
    # Interactive menu
    Write-Host "  Which tools would you like to install?" -ForegroundColor White
    Write-Host ""
    foreach ($t in $tools) {
        Write-Host "  [$($t.Id)] $($t.Name)" -ForegroundColor Cyan
        Write-Host "      $($t.Description)" -ForegroundColor DarkGray
        Write-Host ""
    }
    Write-Host "  [A] All of the above" -ForegroundColor Yellow
    Write-Host "  [Q] Quit / cancel"    -ForegroundColor DarkGray
    Write-Host ""

    $choice = (Read-Host "  Enter selection (e.g. 1, 2, 1 2, or A)").Trim().ToUpper()

    if ($choice -eq "Q" -or $choice -eq "") {
        Write-Host ""
        Write-Skip "No selection made — exiting."
        Write-Host ""
        exit 0
    }

    if ($choice -eq "A") {
        foreach ($t in $tools) { $selectedIds.Add($t.Id) }
    } else {
        # Parse space- or comma-separated numbers
        $tokens = $choice -split '[\s,]+' | Where-Object { $_ -match '^\d+$' }
        foreach ($tok in $tokens) {
            $id = [int]$tok
            if ($tools | Where-Object { $_.Id -eq $id }) {
                if ($selectedIds -notcontains $id) { $selectedIds.Add($id) }
            } else {
                Write-Info "Unknown tool ID '$tok' — ignored."
            }
        }
    }

    if ($selectedIds.Count -eq 0) {
        Write-Err "No valid tools selected."
        exit 1
    }

    Write-Host ""
    Write-Info "Selected tools:"
    foreach ($id in $selectedIds) {
        $t = $tools | Where-Object { $_.Id -eq $id }
        Write-Host "   - $($t.Name)" -ForegroundColor White
    }
    Write-Host ""
    $confirm = (Read-Host "  Proceed with installation? [Y/n]").Trim().ToUpper()
    if ($confirm -eq "N") {
        Write-Skip "Installation cancelled by user."
        Write-Host ""
        exit 0
    }
}

# ---------------------------------------------------------------------------
# Run installers
# ---------------------------------------------------------------------------

# Results table: tool name → "SUCCESS" | "FAILED" | "SKIPPED"
$results = [ordered]@{}

foreach ($t in $tools) {
    $results[$t.Name] = "SKIPPED"
}

Write-Host ""
Write-Divider
Write-Host "  Starting installation of $($selectedIds.Count) tool(s)..." -ForegroundColor White
Write-Divider

foreach ($id in $selectedIds) {
    $tool       = $tools | Where-Object { $_.Id -eq $id }
    $scriptPath = Join-Path $scriptDir $tool.Script

    Write-Host ""
    Write-Host "  ┌─────────────────────────────────────────────────────┐" -ForegroundColor Cyan
    Write-Host "  │  Installing: $($tool.Name.PadRight(39))│" -ForegroundColor Cyan
    Write-Host "  └─────────────────────────────────────────────────────┘" -ForegroundColor Cyan
    Write-Host ""

    if (-not (Test-Path $scriptPath)) {
        Write-Err "Installer script not found: $scriptPath"
        $results[$tool.Name] = "FAILED"
        continue
    }

    try {
        # Run the child script in a new PowerShell process so its
        # $ErrorActionPreference = "Stop" and param bindings are isolated.
        $proc = Start-Process -FilePath "powershell.exe" `
                              -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass",
                                            "-File", "`"$scriptPath`"" `
                              -Wait -PassThru -NoNewWindow

        if ($proc.ExitCode -ne 0) {
            throw "Child process exited with code $($proc.ExitCode)"
        }

        $results[$tool.Name] = "SUCCESS"
        Write-Host ""
        Write-OK "$($tool.Name) installation completed."

    } catch {
        $results[$tool.Name] = "FAILED"
        Write-Host ""
        Write-Err "$($tool.Name) installation FAILED: $_"
        Write-Host "  See output above for details." -ForegroundColor DarkGray
    }
}

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

Write-Host ""
Write-Divider
Write-Host "  INSTALLATION SUMMARY" -ForegroundColor White
Write-Divider
Write-Host ""

$colWidth = ($results.Keys | Measure-Object -Property Length -Maximum).Maximum + 2

foreach ($name in $results.Keys) {
    $status = $results[$name]
    $padded = $name.PadRight($colWidth)
    switch ($status) {
        "SUCCESS" { Write-Host "  $padded  SUCCESS" -ForegroundColor Green   }
        "FAILED"  { Write-Host "  $padded  FAILED"  -ForegroundColor Red     }
        "SKIPPED" { Write-Host "  $padded  skipped" -ForegroundColor DarkGray }
    }
}

Write-Host ""
Write-Divider

$failed  = ($results.Values | Where-Object { $_ -eq "FAILED"  }).Count
$success = ($results.Values | Where-Object { $_ -eq "SUCCESS" }).Count
$skipped = ($results.Values | Where-Object { $_ -eq "SKIPPED" }).Count

if ($failed -gt 0 -and $success -eq 0) {
    Write-Host ""
    Write-Err "All selected installations failed. Check errors above."
    Write-Host ""
    exit 1
} elseif ($failed -gt 0) {
    Write-Host ""
    Write-Host "  $success tool(s) installed successfully, $failed failed, $skipped skipped." -ForegroundColor Yellow
    Write-Host "  Review the error output above for the failed tool(s)." -ForegroundColor Yellow
    Write-Host ""
    exit 1
} else {
    Write-Host ""
    Write-Host "  $success tool(s) installed successfully. $skipped skipped." -ForegroundColor Green
    Write-Host ""
    Write-Host "  Open RADICAL LTX Desktop and enjoy the new capabilities!" -ForegroundColor Cyan
    Write-Host ""
    exit 0
}
