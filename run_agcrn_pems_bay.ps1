# Detached orchestration: trains AGCRN on PEMS-BAY, 3 runs (seeds 10/11/12), unattended.
# Launch via Scheduled Task so it survives VSCode closing.
# Progress: logs/AUTORUN_STATUS_AGCRN.txt ; full log: logs/agcrn_pems-bay.log
# Checkpoints copied to save/agcrn_pemsbay_run{0,1,2}.pth

$ErrorActionPreference = "Continue"
$root   = "c:\Documents\thesis\code\Traffic-Benchmark_Fork"
$python = "C:\Users\tryno\.pyenv\pyenv-win\versions\3.6.8\python.exe"
$status = "$root\logs\AUTORUN_STATUS_AGCRN.txt"
$expdir = "$root\methods\AGCRN\model\experiments\PEMS-BAY"
Set-Location $root

function Note($msg) { "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg | Out-File -FilePath $status -Append -Encoding utf8 }

"=== AGCRN AUTORUN START {0} ===" -f (Get-Date) | Out-File -FilePath $status -Encoding utf8
"=== AGCRN PEMS-BAY {0} ===" -f (Get-Date) | Out-File -FilePath "$root\logs\agcrn_pems-bay.log" -Encoding utf8

for ($r = 0; $r -lt 3; $r++) {
    $seed = 10 + $r
    Note "AGCRN: run $r training (seed $seed)"
    "=== Run $r (seed $seed) ===" | Out-File -FilePath "$root\logs\agcrn_pems-bay.log" -Append -Encoding utf8

    & $python methods/AGCRN/model/Run_PEMS-BAY.py --device cuda:0 --seed $seed *>&1 |
        Out-File -FilePath "$root\logs\agcrn_pems-bay.log" -Append -Encoding utf8

    # Find newest experiment dir that has a saved checkpoint = the run that just finished
    $latest = Get-ChildItem $expdir -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName 'best_model.pth') } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latest) {
        Copy-Item (Join-Path $latest.FullName 'best_model.pth') "$root\save\agcrn_pemsbay_run$r.pth" -Force
        Note "AGCRN: run $r checkpoint -> save/agcrn_pemsbay_run$r.pth (from $($latest.Name))"
    } else {
        Note "AGCRN: run $r WARNING - no best_model.pth found"
    }
}

Note "ALL AGCRN DONE"
"=== AGCRN AUTORUN COMPLETE {0} ===" -f (Get-Date) | Out-File -FilePath $status -Append -Encoding utf8
