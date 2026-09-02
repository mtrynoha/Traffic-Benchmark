# Train DGCRN on PEMS-BAY, 3 runs
# Run from repo root: .\run_dgcrn_bay.ps1
# Checkpoints -> save/expDGCRN_pemsbay_{0,1,2}.pth
# Summary (mean+std across runs) printed at end

$logFile = "logs/dgcrn_pems-bay.log"

"Starting DGCRN PEMS-BAY training $(Get-Date)" | Tee-Object -FilePath $logFile

python methods/DGCRN/train.py `
    --adj_data data/sensor_graph/adj_mx_bay.pkl `
    --data data/PEMS-BAY `
    --num_nodes 325 `
    --runs 3 `
    --epochs 110 `
    --print_every 10 `
    --batch_size 64 `
    --tolerance 100 `
    --expid DGCRN_pemsbay `
    --cl_decay_steps 5500 `
    --rnn_size 96 `
    --device cuda:0 2>&1 | Tee-Object -Append -FilePath $logFile

"Finished DGCRN PEMS-BAY $(Get-Date)" | Tee-Object -Append -FilePath $logFile
