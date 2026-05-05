param(
    [ValidateSet("demo", "final", "midseason", "midseason-final", "report")]
    [string]$Mode = "demo"
)

$ErrorActionPreference = "Stop"

Write-Host "NBA Championship Prediction Project Runner"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found. Install Python 3.10+ and rerun."
    exit 1
}

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if ($Mode -eq "demo") {
    Write-Host "Running fast demo..."
    python run_research_study.py `
        --epochs-mlp 5 `
        --epochs-attention 5 `
        --lr 0.001 `
        --seed 42 `
        --output-dir demo_outputs/results `
        --report-path demo_outputs/demo_report.md `
        --figures-dir demo_outputs/figures

    Write-Host "Demo complete."
    Write-Host "Demo report: demo_outputs/demo_report.md"
    Write-Host "Demo metrics: demo_outputs/results/model_comparison.csv"
}

if ($Mode -eq "final") {
    Write-Host "Running full final experiment..."
    python run_research_study.py `
        --epochs-mlp 150 `
        --epochs-attention 200 `
        --lr 0.001 `
        --seed 42 `
        --output-dir results/research_study `
        --report-path docs/final_report.md `
        --figures-dir docs/figures

    Write-Host "Final run complete."
    Write-Host "Final report: docs/final_report.md"
    Write-Host "Metrics: results/research_study/model_comparison.csv"
    Write-Host "Figures: docs/figures/"
}

if ($Mode -eq "midseason") {
    Write-Host "Running fast mid-season demo..."
    python run_midseason_study.py `
        --epochs-mlp 5 `
        --epochs-attention 5 `
        --lr 0.001 `
        --seed 42 `
        --output-dir demo_outputs/midseason_results `
        --report-path demo_outputs/midseason_demo_report.md `
        --figures-dir demo_outputs/figures_midseason

    Write-Host "Mid-season demo complete."
    Write-Host "Mid-season demo report: demo_outputs/midseason_demo_report.md"
    Write-Host "Mid-season demo metrics: demo_outputs/midseason_results/model_comparison.csv"
}

if ($Mode -eq "midseason-final") {
    Write-Host "Running full mid-season experiment..."
    python run_midseason_study.py `
        --epochs-mlp 150 `
        --epochs-attention 200 `
        --lr 0.001 `
        --seed 42 `
        --output-dir results/midseason_study `
        --report-path docs/midseason_report.md `
        --figures-dir docs/figures_midseason

    Write-Host "Mid-season final run complete."
    Write-Host "Mid-season report: docs/midseason_report.md"
    Write-Host "Metrics: results/midseason_study/model_comparison.csv"
    Write-Host "Figures: docs/figures_midseason/"
}

if ($Mode -eq "report") {
    Write-Host "Main artifacts:"
    Write-Host "docs/final_report.md"
    Write-Host "results/research_study/summary_metrics.json"
    Write-Host "results/research_study/model_comparison.csv"
    Write-Host "results/research_study/fold_metrics.csv"
    Write-Host "results/research_study/predictions.csv"
    Write-Host "docs/figures/research_confusion_matrices.png"
    Write-Host "docs/figures/research_attention_weights.png"
    Write-Host "docs/figures/research_tsne.png"
    Write-Host "data/processed/features_midseason.csv"
    Write-Host "docs/midseason_report.md"
    Write-Host "results/midseason_study/summary_metrics.json"
    Write-Host "results/midseason_study/model_comparison.csv"
    Write-Host "results/midseason_study/fold_metrics.csv"
    Write-Host "results/midseason_study/predictions.csv"
    Write-Host "docs/figures_midseason/midseason_confusion_matrices.png"
    Write-Host "docs/figures_midseason/midseason_attention_weights.png"
    Write-Host "docs/figures_midseason/midseason_tsne.png"

    if (Test-Path "docs/final_report.md") {
        try {
            Start-Process "docs/final_report.md"
        }
        catch {
            Write-Warning "Could not open docs/final_report.md automatically."
        }
    }
}
