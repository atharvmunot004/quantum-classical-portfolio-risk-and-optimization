# Run QAOA optimization in a detached terminal window
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Start the process in a new PowerShell window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m src.quantum_optimization.QAOA_for_CVaR.main" -WindowStyle Normal

Write-Host "QAOA optimization started in a new terminal window."
Write-Host "You can close this window - the optimization will continue running."
