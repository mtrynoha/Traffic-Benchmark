# Detached orchestration: trains AGCRN on METR-LA, 3 runs (seeds 10/11/12), unattended.
# Launch via Scheduled Task so it survives VSCode closing.
# Progress: logs/AUTORUN_STATUS_AGCRN_METRLA.txt ; full log: logs/agcrn_metr-la.log
# Checkpoints copied to save/agcrn_metrla_run{0,1,2}.pth

$ErrorActionPreference = "Continue"
$root   = "c:\Documents\thesis\code\Traffic-Benchmark_Fork"
$python = "C:\Users\tryno\.pyenv\pyenv-win\versions\3.6.8\python.exe"
$status = "$root\logs\AUTORUN_STATUS_AGCRN_METRLA.txt"
$expdir = "$root\methods\AGCRN\model\experiments\METR-LA"
Set-Location $root

function Note($msg) { "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg | Out-File -FilePath $status -Append -Encoding utf8 }

"=== AGCRN METR-LA AUTORUN START {0} ===" -f (Get-Date) | Out-File -FilePath $status -Encoding utf8
"=== AGCRN METR-LA {0} ===" -f (Get-Date) | Out-File -FilePath "$root\logs\agcrn_metr-la.log" -Encoding utf8

for ($r = 0; $r -lt 3; $r++) {
    $seed = 10 + $r
    Note "AGCRN METR-LA: run $r training (seed $seed)"
    "=== Run $r (seed $seed) ===" | Out-File -FilePath "$root\logs\agcrn_metr-la.log" -Append -Encoding utf8

    & $python methods/AGCRN/model/Run_METR-LA.py --device cuda:0 --seed $seed *>&1 |
        Out-File -FilePath "$root\logs\agcrn_metr-la.log" -Append -Encoding utf8

    $latest = Get-ChildItem $expdir -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName 'best_model.pth') } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latest) {
        Copy-Item (Join-Path $latest.FullName 'best_model.pth') "$root\save\agcrn_metrla_run$r.pth" -Force
        Note "AGCRN METR-LA: run $r checkpoint -> save/agcrn_metrla_run$r.pth (from $($latest.Name))"
    } else {
        Note "AGCRN METR-LA: run $r WARNING - no best_model.pth found"
    }
}

Note "ALL AGCRN METR-LA DONE"
"=== AGCRN METR-LA AUTORUN COMPLETE {0} ===" -f (Get-Date) | Out-File -FilePath $status -Append -Encoding utf8
