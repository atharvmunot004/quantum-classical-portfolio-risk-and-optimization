@echo off
REM Run QAOA optimization in a detached terminal window
cd /d "%~dp0"
start "QAOA Optimization" cmd /k "python -m src.quantum_optimization.QAOA_for_CVaR.main"
echo QAOA optimization started in a new terminal window.
echo You can close this window - the optimization will continue running.
