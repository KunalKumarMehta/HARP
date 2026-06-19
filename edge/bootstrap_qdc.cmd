@echo off
REM HARP one-command QDC provisioner. Run from the repo root in the SSH/cmd session:
REM     edge\bootstrap_qdc.cmd
REM Pass-through flags:
REM     edge\bootstrap_qdc.cmd -SkipRun
REM     edge\bootstrap_qdc.cmd -QairtZip C:\path\to\qairt-2.45.zip
REM     edge\bootstrap_qdc.cmd -QpmExe C:\path\to\QPM3.x.exe   (auto-detected from repo root)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap_qdc.ps1" %*
