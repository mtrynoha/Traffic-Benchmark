# Train DCRNN on PEMS-BAY, 3 runs
# Run from repo root: .\run_dcrnn_bay.ps1
# Checkpoints -> models/bay_best_run{0,1,2}.tar
# Test metrics appended to log after each run via --TEST_ONLY

$logFile = "logs/dcrnn_pems-bay.log"
$config  = "methods/DCRNN/data/model/dcrnn_bay.yaml"

"Starting DCRNN PEMS-BAY training $(Get-Date)" | Tee-Object -FilePath $logFile

for ($r = 0; $r -lt 3; $r++) {
    "=== Run $r / train ===" | Tee-Object -Append -FilePath $logFile

    python methods/DCRNN/dcrnn_train_pytorch.py `
        --config_filename=$config 2>&1 | Tee-Object -Append -FilePath $logFile

    "=== Run $r / test ===" | Tee-Object -Append -FilePath $logFile

    # Load best checkpoint and evaluate per-horizon test metrics
    python methods/DCRNN/dcrnn_train_pytorch.py `
        --config_filename=$config `
        --LOAD_INITIAL True `
        --TEST_ONLY True 2>&1 | Tee-Object -Append -FilePath $logFile

    # Rename checkpoint so next run doesn't overwrite
    Move-Item "models/bay_best.tar" "models/bay_best_run$r.tar" -Force
    "Checkpoint saved as models/bay_best_run$r.tar" | Tee-Object -Append -FilePath $logFile
}

"Finished DCRNN PEMS-BAY $(Get-Date)" | Tee-Object -Append -FilePath $logFile
