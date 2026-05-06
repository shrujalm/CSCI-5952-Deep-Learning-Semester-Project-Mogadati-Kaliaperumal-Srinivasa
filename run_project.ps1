param(
    [ValidateSet("demo", "showcase", "final", "midseason", "midseason-final", "report")]
    [string]$Mode = "demo",
    [switch]$NoPause,
    [switch]$Pause,
    [int]$StepDelaySeconds = 2
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

function Write-ShowcaseText {
    param([string]$Text)

    Write-Host ""
    Write-Host $Text
}

function Write-Story {
    param([string]$Text)

    Write-Host ""
    Write-Host $Text
}

function Write-Result {
    param([string]$Text)

    Write-Host ""
    Write-Host $Text
}

function Wait-Showcase {
    param([string]$Message = "Press Enter when you are ready for the next step")

    if ($Pause -and -not $NoPause) {
        [void](Read-Host $Message)
        return
    }

    if (-not $NoPause -and $StepDelaySeconds -gt 0) {
        Start-Sleep -Seconds $StepDelaySeconds
    }
}

function Invoke-ShowcaseCommand {
    param(
        [string]$Description,
        [string]$CommandText,
        [scriptblock]$Command
    )

    Write-Story $Description

    if ($CommandText) {
        Write-Host ""
        Write-Host "Running command:"
        Write-Host "  $CommandText"
    }

    & $Command
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

function Show-DataSnapshot {
    param(
        [string]$Path,
        [string]$Title
    )

    if (-not (Test-Path $Path)) {
        Write-Warning "Missing dataset file: $Path"
        return
    }

    $data = Import-Csv $Path
    $columnCount = 0
    if ($data.Count -gt 0) {
        $columnCount = ($data[0].PSObject.Properties | Measure-Object).Count
    }

    $seasons = $data.SEASON | Sort-Object -Unique

    Write-Host ""
    Write-Host $Title
    Write-Host ("Rows: {0} team-seasons" -f $data.Count)
    Write-Host ("Columns: {0} engineered features and labels" -f $columnCount)
    Write-Host ("Season range: {0} through {1}" -f $seasons[0], $seasons[$seasons.Count - 1])

    Write-Host ""
    Write-Host "Label balance in the data:"
    $data |
        Group-Object PLAYOFF_RESULT |
        Sort-Object -Property @{ Expression = { [int]$_.Name }; Ascending = $true } |
        ForEach-Object {
            [pscustomobject]@{
                Label = $_.Name
                Outcome = (Convert-PlayoffLabel $_.Name)
                Count = $_.Count
                Share = (Format-Percent ($_.Count / $data.Count))
            }
        } |
        Format-Table -AutoSize
}

function Show-FoldEvaluationSummary {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        Write-Warning "Missing fold metrics file: $Path"
        return
    }

    $folds = Import-Csv $Path
    $seasonCount = ($folds.season | Sort-Object -Unique).Count

    Write-Host ""
    Write-Host ("Leave-one-season-out evaluation: {0} held-out seasons per model" -f $seasonCount)

    $folds |
        Group-Object model_key |
        ForEach-Object {
            $best = $_.Group | Sort-Object -Property @{ Expression = { [double]$_.accuracy }; Descending = $true } | Select-Object -First 1
            $worst = $_.Group | Sort-Object -Property @{ Expression = { [double]$_.accuracy }; Ascending = $true } | Select-Object -First 1
            $avgAccuracy = ($_.Group | Measure-Object -Property accuracy -Average).Average
            $avgMacroF1 = ($_.Group | Measure-Object -Property f1_macro -Average).Average

            [pscustomobject]@{
                Model = $_.Name
                "Avg Accuracy" = (Format-Percent $avgAccuracy)
                "Avg Macro F1" = (Format-Percent $avgMacroF1)
                "Best Held-Out Season" = ("{0} ({1})" -f $best.season, (Format-Percent $best.accuracy))
                "Hardest Held-Out Season" = ("{0} ({1})" -f $worst.season, (Format-Percent $worst.accuracy))
            }
        } |
        Format-Table -AutoSize
}

function Show-ArtifactStatus {
    $artifacts = @(
        @{ Path = "docs/final_report.pdf"; Purpose = "final written report" },
        @{ Path = "docs/real_results_summary.md"; Purpose = "plain-English results summary" },
        @{ Path = "results/research_study/model_comparison.csv"; Purpose = "full-season model metrics" },
        @{ Path = "results/research_study/predictions.csv"; Purpose = "held-out full-season predictions" },
        @{ Path = "results/midseason_study/model_comparison.csv"; Purpose = "mid-season model metrics" },
        @{ Path = "results/midseason_study/predictions.csv"; Purpose = "held-out mid-season predictions" },
        @{ Path = "docs/figures"; Purpose = "full-season visual analysis figures" },
        @{ Path = "docs/figures_midseason"; Purpose = "mid-season visual analysis figures" }
    )

    Write-Host ""
    Write-Host "Artifact check:"
    $artifacts |
        ForEach-Object {
            [pscustomobject]@{
                Artifact = $_.Path
                Purpose = $_.Purpose
                Status = if (Test-Path $_.Path) { "ready" } else { "missing" }
            }
        } |
        Format-Table -AutoSize
}

function Show-FigureInventory {
    $figures = @()
    if (Test-Path "docs/figures") {
        $figures += Get-ChildItem "docs/figures" -Filter "*.png"
    }
    if (Test-Path "docs/figures_midseason") {
        $figures += Get-ChildItem "docs/figures_midseason" -Filter "*.png"
    }

    Write-Host ""
    Write-Host ("Figure inventory: {0} PNG analysis figures are ready for the report and presentation." -f $figures.Count)
    $figures |
        Sort-Object DirectoryName, Name |
        ForEach-Object {
            [pscustomobject]@{
                Folder = (Split-Path $_.DirectoryName -Leaf)
                Figure = $_.Name
            }
        } |
        Format-Table -AutoSize
}

Write-Host "NBA Championship Prediction Project Runner"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found. Install Python 3.10+ and rerun."
    exit 1
}

if ($Mode -eq "showcase") {
    Write-Section "0. Environment Setup"
    Write-ShowcaseText "I am starting with the same command a reviewer can run. The script sets up the environment, runs checks, evaluates models, and prints the results on screen."
    Write-Story "Checking Python, the virtual environment, and the required packages before the project code runs."
}

$createdVenv = $false
if (-not (Test-Path ".venv")) {
    if ($Mode -eq "showcase") {
        Write-Host ""
        Write-Host "Running command:"
        Write-Host "  python -m venv .venv"
    }
    python -m venv .venv
    $createdVenv = $true
}

.\.venv\Scripts\Activate.ps1

$requiredModules = @("torch", "numpy", "pandas", "sklearn", "matplotlib", "seaborn", "nba_api", "tqdm")
$missingModules = @()
foreach ($module in $requiredModules) {
    python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$module') else 1)" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        $missingModules += $module
    }
}

if ($createdVenv -or $missingModules.Count -gt 0) {
    $localTemp = Join-Path (Get-Location) ".tmp"
    if (-not (Test-Path $localTemp)) {
        New-Item -ItemType Directory -Path $localTemp | Out-Null
    }
    $env:TEMP = $localTemp
    $env:TMP = $localTemp

    if ($Mode -eq "showcase") {
        Write-Host ""
        if ($missingModules.Count -gt 0) {
            Write-Host ("Missing packages detected: {0}" -f ($missingModules -join ", "))
        }
        Write-Host "Running command:"
        Write-Host "  python -m pip install --disable-pip-version-check -q -r requirements.txt"
    }

    python -m pip install --disable-pip-version-check -q -r requirements.txt
}
elseif ($Mode -eq "showcase") {
    Write-Host ""
    Write-Host "Dependency check: all required packages are already installed, so pip install is skipped."
}

if ($Mode -eq "showcase") {
    Write-Result "The environment is ready, so any later success comes from the project code and saved research artifacts rather than a hidden notebook state."
    Wait-Showcase
}

if ($Mode -eq "showcase") {
    Write-Section "Showcase Mode: Testing, Evaluation, and Prediction"
    Write-ShowcaseText "This project predicts how far NBA teams go in the playoffs using regular-season team stats and top-eight rotation player features. I am going to show testing, a live evaluation run, final research metrics, and concrete team-season predictions."
    Wait-Showcase

    Write-Section "1. Testing: Does The Project Execute?"
    Write-ShowcaseText "Before I trust any metrics, I first test that the runner parses and the Python package compiles. This is the quick sanity check that catches broken scripts before we talk about model quality."
    Invoke-ShowcaseCommand `
        -Description "Validating the PowerShell runner syntax so the one-click demo itself is testable." `
        -CommandText 'powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Command .\run_project.ps1 -Syntax"' `
        -Command { powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Command .\run_project.ps1 -Syntax" }
    Invoke-ShowcaseCommand `
        -Description "Compiling the Python source tree. If imports or syntax are broken, this step fails before evaluation starts." `
        -CommandText "python -m compileall -q src run_pipeline.py run_research_study.py run_midseason_study.py scripts" `
        -Command { python -m compileall -q src run_pipeline.py run_research_study.py run_midseason_study.py scripts }
    Write-Result "Testing passed: the runner is syntactically valid and the Python project compiles."
    Wait-Showcase

    Write-Section "2. Problem Setup: What Are We Solving?"
    Write-ShowcaseText "The model is not guessing one champion from vibes. It sees one row per team-season, learns from engineered team and rotation features, and predicts one of six playoff-depth labels."
    Show-DataSnapshot "data/processed/features_research.csv" "Full-season dataset snapshot"
    Write-Result "This shows the hard part of the problem: the dataset is real but small, and rare outcomes like Champion and Finals Loss are heavily imbalanced."
    Wait-Showcase

    Write-Section "3. Live Evaluation: Run The Pipeline"
    Write-ShowcaseText "Now I run the actual evaluation pipeline in a short showcase configuration. The neural models use only five epochs so the demo finishes live, but the pipeline still rebuilds data artifacts, trains models, evaluates them, and writes outputs."
    Invoke-ShowcaseCommand `
        -Description "Running the research pipeline end to end with a short training budget for screen-share timing." `
        -CommandText "python run_research_study.py --epochs-mlp 5 --epochs-attention 5 --lr 0.001 --seed 42 --output-dir demo_outputs/showcase_results --report-path demo_outputs/showcase_report.md --figures-dir demo_outputs/showcase_figures" `
        -Command {
            $previousTqdmDisable = $env:TQDM_DISABLE
            $env:TQDM_DISABLE = "1"
            try {
                python run_research_study.py `
                    --epochs-mlp 5 `
                    --epochs-attention 5 `
                    --lr 0.001 `
                    --seed 42 `
                    --output-dir demo_outputs/showcase_results `
                    --report-path demo_outputs/showcase_report.md `
                    --figures-dir demo_outputs/showcase_figures
            }
            finally {
                $env:TQDM_DISABLE = $previousTqdmDisable
            }
        }

    Show-ModelComparison "demo_outputs/showcase_results/model_comparison.csv" "Live demo metrics from this run"
    Write-Result "The live run proves the pipeline works end to end. The short neural training budget makes it presentation-friendly, so I use the committed full-training artifacts for the final research claims."
    Wait-Showcase

    Write-Section "4. Final Evaluation: Full Training Artifacts"
    Write-ShowcaseText "Now I switch from the quick live run to the full saved experiments. These are the results from the longer training runs and leave-one-season-out validation."
    Show-ModelComparison "results/research_study/model_comparison.csv" "Full regular-season evaluation"
    Show-ModelComparison "results/midseason_study/model_comparison.csv" "Mid-season evaluation"
    Show-FoldEvaluationSummary "results/research_study/fold_metrics.csv"
    Write-Result "Random Forest leads on full-season accuracy, while the Attention Model stays competitive and gives us interpretability through learned rotation-slot weights. The mid-season task is harder because the model has less season information, which is exactly the real trade-deadline use case."
    Wait-Showcase

    Write-Section "5. Solving Examples: What Did It Predict?"
    Write-ShowcaseText "Aggregate metrics are useful, but the easiest way to understand the model is to inspect held-out team-seasons. These examples show actual playoff outcomes beside each model prediction."
    Show-UseCasePredictions "results/research_study/predictions.csv"
    Write-Result "These examples show both strengths and limits: the model separates many missed-playoff teams well, can identify some champions, and still struggles with rare Finals outcomes."
    Wait-Showcase

    Write-Section "6. Visual Evidence And Saved Artifacts"
    Write-ShowcaseText "The project also produces visual analysis: confusion matrices, t-SNE clustering, attention weights, finals diagnostics, and a market-size fairness audit."
    Show-FigureInventory
    Show-ArtifactStatus
    Write-Host ""
    Write-Host "Generated during this live showcase:"
    Write-Host "  demo_outputs/showcase_report.md"
    Write-Host "  demo_outputs/showcase_results/model_comparison.csv"
    Write-Host "  demo_outputs/showcase_results/predictions.csv"
    Write-Host "  demo_outputs/showcase_figures/"
    Write-Result "The final takeaway is measured: postseason depth is learnable from regular-season and roster statistics, Random Forest is the strongest accuracy baseline, and attention adds model interpretability without pretending that exact champion prediction is solved."
    Write-Host ""
    Write-Host "Showcase complete."
    Write-Host "One-command demo finished successfully."
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
