"""Thin wrapper for the mid-season championship prediction experiment."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nba_championship_prediction.data_pipeline import (
    PROCESSED_MIDSEASON_FEATURES_FILE,
    build_midseason_dataset,
)
from nba_championship_prediction.research_study import StudyReportContext, parse_args, run_study


def main() -> None:
    args = parse_args(
        description="Run the mid-season NBA championship prediction study.",
        default_output_dir="results/midseason_study",
        default_report_path="docs/midseason_report.md",
        default_figures_dir="docs/figures_midseason",
        include_force_refresh_midseason=True,
    )

    def dataset_builder(seasons: list[str], processed_file: Path):
        return build_midseason_dataset(
            seasons=seasons,
            processed_file=processed_file,
            force_refresh=args.force_refresh_midseason,
        )

    context = StudyReportContext(
        title="Mid-Season NBA Championship Prediction from Team and Rotation Statistics",
        setting_name="Mid-season prediction using stats available around the All-Star break",
        feature_timeframe="All-Star-break proxy / mid-season",
        data_source_description=(
            "`nba_api` endpoints `LeagueDashTeamStats` for base team stats, "
            "`LeagueDashTeamStats` with `MeasureType=Advanced` for advanced team stats, "
            "and `LeagueDashPlayerStats`, all filtered with `DateTo`/`date_to_nullable` cutoffs"
        ),
        feature_intro="The mid-season feature matrix matches the full-season schema exactly and contains",
        main_result_note=(
            "The table reports out-of-fold metrics across all team-seasons plus fold-level mean and "
            "standard deviation for accuracy. This setting is intentionally harder than the full-season "
            "experiment because every feature is limited to information available around the mid-season cutoff."
        ),
        limitations=[
            "The cutoff dictionary approximates each season's All-Star Sunday or a nearby mid-February date, so it should be interpreted as an All-Star-break proxy rather than an exact betting-market timestamp.",
            "The 2020-21 season uses March 7, 2021 because the NBA All-Star Game moved later on the shortened COVID calendar.",
            "Advanced mid-season columns are mapped from OFF_RATING, DEF_RATING, NET_RATING, and PACE to the full-season E_* schema; if an endpoint omits one, the pipeline fills that feature with 0.0.",
            "The label remains the final postseason result, so trades, injuries, and form changes after the cutoff can legitimately change the outcome.",
            "The champion and finals-loss classes remain very sparse for a six-class problem.",
        ],
        data_notes=[
            "Features use only regular-season statistics through season-specific cutoffs in `MIDSEASON_CUTOFF_DATES`.",
            "Labels still use the final playoff outcome: 0 = Missed Playoffs through 5 = Champion.",
            "The processed artifact is `data/processed/features_midseason.csv`.",
        ],
        figure_prefix="midseason",
        command_name="run_midseason_study.py",
    )

    run_study(
        args=args,
        dataset_builder=dataset_builder,
        processed_file=PROCESSED_MIDSEASON_FEATURES_FILE,
        report_context=context,
    )


if __name__ == "__main__":
    main()
