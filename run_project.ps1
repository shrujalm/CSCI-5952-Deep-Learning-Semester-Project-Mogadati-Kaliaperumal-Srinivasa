param(
    [ValidateSet("demo", "showcase", "final", "midseason", "midseason-final", "report")]
    [string]$Mode = "demo",
    [switch]$NoPause
)

# Syntax-check on execution-policy-restricted Windows hosts:
# powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Command .\run_project.ps1 -Syntax"

$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Title)

    Write-Host ""
    Write-Host "============================================================"
    Write-Host $Title
    Write-Host "============================================================"
}

function Write-Cue {
    param([string]$Text)

    Write-Host ""
    Write-Host "Presenter cue:"
    Write-Host "  $Text"
}

function Wait-Showcase {
    param([string]$Message = "Press Enter when you are ready for the next step")

    if (-not $NoPause) {
        [void](Read-Host $Message)
    }
}

function Format-Percent {
    param([object]$Value)

    return "{0:N1}%" -f ([double]$Value * 100)
}

function Convert-PlayoffLabel {
    param([object]$Value)

    $labels = @(
        "Missed Playoffs",
        "First Round Exit",
        "Second Round Exit",
        "Conference Finals",
        "Finals Loss",
        "Champion"
    )

    try {
        $index = [int]$Value
        if ($index -ge 0 -and $index -lt $labels.Count) {
            return $labels[$index]
        }
    }
    catch {
        return $Value
    }

    return $Value
}

function Show-ModelComparison {
    param(
        [string]$Path,
        [string]$Title
    )

    if (-not (Test-Path $Path)) {
        Write-Warning "Missing metrics file: $Path"
        return
    }

    Write-Host ""
    Write-Host $Title

    Import-Csv $Path |
        Sort-Object -Property @{ Expression = { [double]$_.accuracy }; Descending = $true } |
        ForEach-Object {
            [pscustomobject]@{
                Model = $_.model_name
                Accuracy = (Format-Percent $_.accuracy)
                "Macro F1" = (Format-Percent $_.f1_macro)
                "Top-2 Accuracy" = (Format-Percent $_.top2_accuracy)
            }
        } |
        Format-Table -AutoSize
}

function Show-UseCasePredictions {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        Write-Warning "Missing predictions file: $Path"
        return
    }

    $predictions = Import-Csv $Path
    $targets = @(
        @{ Season = "2023-24"; Team = "BOS"; Case = "Champion signal" },
        @{ Season = "2023-24"; Team = "DEN"; Case = "Strong contender" },
        @{ Season = "2023-24"; Team = "DAL"; Case = "Finals run that was hard to forecast" },
        @{ Season = "2023-24"; Team = "ATL"; Case = "Missed-playoffs separation" },
        @{ Season = "2016-17"; Team = "GSW"; Case = "Historical champion" }
    )

    $rows = foreach ($target in $targets) {
        $row = $predictions |
            Where-Object {
                $_.SEASON -eq $target["Season"] -and
                $_.TEAM -eq $target["Team"]
            } |
            Select-Object -First 1

        if ($row) {
            [pscustomobject]@{
                Case = $target["Case"]
                Season = $row.SEASON
                Team = $row.TEAM
                Actual = (Convert-PlayoffLabel $row.PLAYOFF_RESULT)
                "Random Forest" = (Convert-PlayoffLabel $row.RANDOM_FOREST_PRED)
                "MLP" = (Convert-PlayoffLabel $row.MLP_BASELINE_PRED)
                "Attention" = (Convert-PlayoffLabel $row.ATTENTION_MODEL_PRED)
            }
        }
    }

    Write-Host ""
    Write-Host "Concrete held-out prediction examples"

    foreach ($row in $rows) {
        Write-Host ""
        Write-Host ("- {0}: {1} {2}" -f $row.Case, $row.Season, $row.Team)
        Write-Host ("  Actual outcome: {0}" -f $row.Actual)
        Write-Host ("  Random Forest:  {0}" -f $row."Random Forest")
        Write-Host ("  MLP Baseline:   {0}" -f $row.MLP)
        Write-Host ("  Attention:      {0}" -f $row.Attention)
    }
}

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

if ($Mode -eq "showcase") {
    Write-Section "Showcase Mode: Testing, Evaluation, and Prediction"
    Write-Cue "Say: This project predicts how far NBA teams go in the playoffs from regular-season team stats and top-eight rotation player features."
    Wait-Showcase

    Write-Section "1. Lightweight Validation"
    Write-Cue "Say: Before showing results, I validate that the runner is syntactically valid and the Python project compiles."
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Command .\run_project.ps1 -Syntax"
    python -m compileall -q src run_pipeline.py run_research_study.py run_midseason_study.py
    Write-Host "Validation passed: PowerShell syntax and Python compile checks completed."
    Wait-Showcase

    Write-Section "2. Live Evaluation Run"
    Write-Cue "Say: This live run rebuilds the dataset and executes the evaluation pipeline. I keep the neural epochs low so the demo finishes during the presentation."
    Wait-Showcase

    python run_research_study.py `
        --epochs-mlp 5 `
        --epochs-attention 5 `
        --lr 0.001 `
        --seed 42 `
        --output-dir demo_outputs/showcase_results `
        --report-path demo_outputs/showcase_report.md `
        --figures-dir demo_outputs/showcase_figures

    Write-Host "Live showcase run complete."
    Show-ModelComparison "demo_outputs/showcase_results/model_comparison.csv" "Live demo metrics from this run"
    Write-Cue "Say: The live run proves the pipeline works end to end. Because it only trains neural models for five epochs, the neural scores are intentionally not the final research numbers."
    Wait-Showcase

    Write-Section "3. Final Research Evaluation"
    Write-Cue "Say: Now I compare against the committed full training artifacts, where the neural models were trained longer and evaluated across every held-out season."
    Show-ModelComparison "results/research_study/model_comparison.csv" "Full regular-season evaluation"
    Show-ModelComparison "results/midseason_study/model_comparison.csv" "Mid-season evaluation"
    Write-Cue "Say: Full-season features perform better because they contain more information. The mid-season run is harder, but it demonstrates the same problem under a trade-deadline-style use case."
    Wait-Showcase

    Write-Section "4. Concrete Use Cases"
    Write-Cue "Say: Instead of only showing aggregate metrics, I can point to held-out team-seasons and show what the models predicted."
    Show-UseCasePredictions "results/research_study/predictions.csv"
    Write-Cue "Say: These examples show both strengths and limits: the model separates many missed-playoff teams well, can identify some champions, and still struggles with rare Finals outcomes."
    Wait-Showcase

    Write-Section "5. Artifacts to Show"
    Write-Host "Generated during this showcase:"
    Write-Host "  demo_outputs/showcase_report.md"
    Write-Host "  demo_outputs/showcase_results/model_comparison.csv"
    Write-Host "  demo_outputs/showcase_results/predictions.csv"
    Write-Host "  demo_outputs/showcase_figures/"
    Write-Host ""
    Write-Host "Final committed artifacts:"
    Write-Host "  docs/final_report.md"
    Write-Host "  docs/final_report.pdf"
    Write-Host "  docs/real_results_summary.md"
    Write-Host "  results/research_study/"
    Write-Host "  results/midseason_study/"
    Write-Cue "Say: My takeaway is that postseason depth is learnable, but exact champion prediction is still difficult because champion and finals-loss classes are tiny."
    Write-Host ""
    Write-Host "Showcase complete."
    exit 0
}

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
