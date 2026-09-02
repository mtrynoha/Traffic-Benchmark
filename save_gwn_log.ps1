# Run this AFTER Graph WaveNet training completes to save output to the log file.
# The background task output file path is printed when you run run_in_background.
# Adjust $srcFile to match your actual temp output file path if different.

param(
    [string]$srcFile = "C:\Users\tryno\AppData\Local\Temp\claude\c--Documents-thesis\4156f05d-2682-4833-abb0-7ba8628e4ea0\tasks\b6af717p3.output"
)

$destFile = "logs/graph-wavenet_pems-bay.log"

if (Test-Path $srcFile) {
    Copy-Item $srcFile $destFile -Force
    Write-Host "Saved GWN log to $destFile"
} else {
    Write-Host "Source file not found: $srcFile"
}
