# Detached orchestration: trains DGCRN then DCRNN on PEMS-BAY, unattended.
# Designed to be launched via a Windows Scheduled Task so it survives VSCode closing.
# Progress: logs/AUTORUN_STATUS.txt (high-level), logs/dgcrn_pems-bay.log, logs/dcrnn_pems-bay.log

$ErrorActionPreference = "Continue"
$root   = "c:\Documents\thesis\code\Traffic-Benchmark_Fork"
$python = "C:\Users\tryno\.pyenv\pyenv-win\versions\3.6.8\python.exe"
$status = "$root\logs\AUTORUN_STATUS.txt"
Set-Location $root

function Note($msg) { "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg | Out-File -FilePath $status -Append -Encoding utf8 }

"=== AUTORUN START {0} ===" -f (Get-Date) | Out-File -FilePath $status -Encoding utf8

# ---------------- DGCRN: ALREADY COMPLETED (3 runs done 2026-06-07) - SKIPPED ----------------
# Checkpoints: save/expDGCRN_pemsbay_{0,1,2}.pth ; log preserved: logs/dgcrn_pems-bay.log
Note "DGCRN: skipped (already completed)"

# ---------------- DCRNN (batch 64; run 0 already trained & preserved) ----------------
$config = "methods/DCRNN/data/model/dcrnn_bay.yaml"
"=== DCRNN PEMS-BAY {0} ===" -f (Get-Date) | Out-File -FilePath "$root\logs\dcrnn_pems-bay.log" -Encoding utf8

# Runs 0 and 1: trained earlier (converged val_mae ~1.63), preserved. Just evaluate them.
foreach ($pr in 0, 1) {
    Note "DCRNN: run $pr test (preserved checkpoint)"
    "=== Run $pr / test (preserved) ===" | Out-File -FilePath "$root\logs\dcrnn_pems-bay.log" -Append -Encoding utf8
    Copy-Item "$root\models\bay_best_run$pr.tar" "$root\models\bay_best.tar" -Force
    & $python methods/DCRNN/dcrnn_train_pytorch.py --config_filename=$config --LOAD_INITIAL True --TEST_ONLY True *>&1 |
        Out-File -FilePath "$root\logs\dcrnn_pems-bay.log" -Append -Encoding utf8
}

# Run 2 only: train fresh.
for ($r = 2; $r -lt 3; $r++) {
    Note "DCRNN: run $r training"
    "=== Run $r / train ===" | Out-File -FilePath "$root\logs\dcrnn_pems-bay.log" -Append -Encoding utf8
    & $python methods/DCRNN/dcrnn_train_pytorch.py --config_filename=$config *>&1 |
        Out-File -FilePath "$root\logs\dcrnn_pems-bay.log" -Append -Encoding utf8

    Note "DCRNN: run $r test"
    "=== Run $r / test ===" | Out-File -FilePath "$root\logs\dcrnn_pems-bay.log" -Append -Encoding utf8
    & $python methods/DCRNN/dcrnn_train_pytorch.py --config_filename=$config --LOAD_INITIAL True --TEST_ONLY True *>&1 |
        Out-File -FilePath "$root\logs\dcrnn_pems-bay.log" -Append -Encoding utf8

    if (Test-Path "$root\models\bay_best.tar") {
        Move-Item "$root\models\bay_best.tar" "$root\models\bay_best_run$r.tar" -Force
        Note "DCRNN: run $r checkpoint -> models/bay_best_run$r.tar"
    } else {
        Note "DCRNN: run $r WARNING - bay_best.tar not found"
    }
}

Note "ALL DONE"
"=== AUTORUN COMPLETE {0} ===" -f (Get-Date) | Out-File -FilePath $status -Append -Encoding utf8
