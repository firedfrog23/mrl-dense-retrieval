# run_all.ps1
# Sequential runner for phases 1 through 6.
# Halts on the first script that returns a non-zero exit code.
#
# Usage:
#   .\run_all.ps1
#
# If PowerShell blocks the script, run once:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

$scripts = @(
    "01_train_baseline.py",
    "02_train_mrl.py",
    "03_fit_pca.py",
    "04_evaluate.py",
    "05_make_tables.py",
    "06_make_plots.py"
)

$startTime = Get-Date
$totalScripts = $scripts.Count
$sep = "=" * 60

for ($i = 0; $i -lt $totalScripts; $i++) {
    $s = $scripts[$i]
    $stepStart = Get-Date

    Write-Host ""
    Write-Host $sep -ForegroundColor Cyan
    Write-Host "[$($i + 1)/$totalScripts] $s" -ForegroundColor Cyan
    Write-Host $sep -ForegroundColor Cyan

    python $s

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "FAILED: $s (exit code $LASTEXITCODE)" -ForegroundColor Red
        Write-Host "Pipeline halted. Fix the error, then re-run this script." -ForegroundColor Yellow
        Write-Host "Completed phases keep their outputs and caches, so resuming is fast." -ForegroundColor Yellow
        exit $LASTEXITCODE
    }

    $elapsed = (Get-Date) - $stepStart
    Write-Host ""
    Write-Host "OK: $s (took $($elapsed.ToString('mm\:ss')))" -ForegroundColor Green
}

$totalElapsed = (Get-Date) - $startTime
Write-Host ""
Write-Host $sep -ForegroundColor Green
Write-Host "ALL DONE in $($totalElapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
Write-Host $sep -ForegroundColor Green
Write-Host ""
Write-Host "Outputs:" -ForegroundColor Green
Write-Host "  outputs\results\eval_results.csv" -ForegroundColor Green
Write-Host "  outputs\tables\*.tex" -ForegroundColor Green
Write-Host "  outputs\figures\*.pdf" -ForegroundColor Green