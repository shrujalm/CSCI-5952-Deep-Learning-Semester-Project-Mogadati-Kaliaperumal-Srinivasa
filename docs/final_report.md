# Title

Predicting NBA Postseason Depth from Regular-Season Team and Rotation Statistics Using Deep Learning

Pranav Kumar Kaliaperumal, Shrujal Mogadati, and Disha Srinivasa
Department of Computer Science, University of Colorado Denver, Denver, CO 80217 USA

*Keywords*: NBA, sports analytics, deep learning, attention mechanism, player embeddings, postseason prediction, class imbalance, machine learning

# Problem Statement

We study whether we can forecast an NBA team's eventual postseason depth from regular-season team performance, roster context, and rotation-level player statistics. Our project frames the task as a six-class classification problem where each team-season receives one final playoff-depth label: Missed Playoffs, First Round Exit, Second Round Exit, Conference Finals, Finals Loss, or Champion. We use this multi-class formulation because it preserves the difference between a lottery team, a first-round team, a serious contender, and a champion instead of collapsing the season into a simpler champion-versus-non-champion task.

We evaluate the problem across 21 NBA seasons, from 2003-04 through 2023-24, with 629 team-seasons. Each row represents one team in one season, and each label records the deepest playoff round that team reached. We make the task intentionally realistic by validating with leave-one-season-out cross-validation: for each fold, we train on all seasons except one and test on the held-out season. This design asks whether our model generalizes to a future season rather than memorizing patterns from randomly mixed rows.

We face three central difficulties. First, regular-season performance does not fully determine playoff success because matchups, injuries, coaching adjustments, trades, and late-season form can change the playoff path. Second, the dataset is small for deep learning because we only get 30 team examples per season and only one champion per season. Third, the labels are highly imbalanced: 293 team-seasons miss the playoffs, while only 21 team-seasons win the championship and 21 lose in the Finals.

# Motivation

We chose this problem because playoff-depth forecasting sits at the intersection of basketball strategy and machine learning. Teams, analysts, broadcasters, and fans constantly ask whether a team is merely good in the regular season or built for the playoffs. A useful model can support trade-deadline planning, roster evaluation, media analysis, fan engagement, ticketing decisions, and sponsorship planning [1]. We do not treat our model as a betting system or a replacement for expert scouting; we treat it as a decision-support tool that clarifies which statistical signals correlate with postseason depth.

Our project also tests a deeper modeling question. Most NBA prediction work focuses on individual game outcomes or binary playoff qualification, and many strong systems rely on traditional tabular models [2]-[5]. We ask whether a roster-aware neural model can add value when we represent the top eight rotation players explicitly. Basketball is not only a team-average sport: two teams can have similar net ratings while differing sharply in star power, shooting depth, defensive balance, and bench reliability. We therefore build an attention model that learns how much influence to assign to each rotation slot when it forms a team-level postseason prediction.

We also include a mid-season extension because many real decisions happen before the full regular season ends. In that setting, we use statistics available around season-specific All-Star-break proxy dates while keeping the final postseason outcome as the label. This experiment asks whether our workflow can forecast playoff depth when teams still have time to trade, rest players, adjust rotations, or change strategic direction.

We also treat ethics as part of the motivation rather than an afterthought. Our project uses performance statistics instead of player demographics, race, nationality, salary, endorsement value, or media popularity, but team-level sports data can still reflect structural advantages such as market size, organizational resources, media exposure, and free-agency appeal. We therefore include a market-size fairness audit and frame our predictions as decision-support signals, not as final judgments about players, teams, fan bases, cities, or organizational worth.

# Related Works

Sports prediction research often follows two paths: game-level outcome prediction and season-level team evaluation. Khanmohammadi et al. [1] proposed MambaNet, a hybrid neural network that uses team and player time-series to predict NBA playoff games and reports AUC values from 0.72 to 0.82. Zhao et al. [2] used a fused graph convolutional network plus random forest model for basketball game outcome prediction, which motivates graph and team-interaction structure even though our project uses season-level team rows. Ouyang et al. [4] combined XGBoost with SHAP explanations for NBA game outcomes, and Rios et al. [5] applied long-sequence LSTMs to NBA game prediction. These studies show that machine learning can model basketball outcomes, but they focus mainly on games rather than season-level playoff depth.

Representation learning provides another foundation for our project. Guan et al. [3] introduced NBA2Vec, which learns dense player representations from play outcomes without relying only on hand-crafted aggregate measures. Ibrahim et al. [6] and Teno et al. [7] studied season-level basketball prediction with machine learning, showing that championship and playoff-stage forecasting is feasible but difficult; Teno et al. specifically evaluate NBA game outcomes and playoff/championship stages across 10 seasons. Bunker and Thabtah [8] surveyed sport result prediction and explain why sports forecasting matters as a practical decision-support problem. Yeung [9] found that random forests performed well for NBA playoff qualification, and Perricone et al. [10] showed that NBA API data can support competent basketball outcome models.

We also ground our work in class-imbalance and interpretability research. Chawla et al. [12] introduced SMOTE, Elreedy and Atiya [13] analyzed SMOTE variants, and He et al. [14] proposed ADASYN. We do not use synthetic oversampling in our main experiments because leave-one-season-out validation makes cross-season sample synthesis risky, but these methods clarify why our rare champion and Finals Loss classes are hard. Vaswani et al. [15] introduced attention mechanisms, which inspire our roster-weighting architecture. Explainable sports analytics work, including Ouyang et al. [4] and Wang et al. [11], motivates our attention diagnostics and our caution that learned weights support interpretation rather than causal claims.

Our project fills a gap between those threads. We use real NBA data, preserve six playoff-depth labels, compare classical and neural models, validate by season, inspect attention-based roster weights, audit market-size error, and evaluate both full-season and mid-season forecasting.

# Methods

## Data

We collect NBA data through the `nba_api` Python library [16], which exposes NBA.com statistics endpoints. For each season from 2003-04 through 2023-24, we assemble team box-score data, advanced team efficiency metrics, and player statistics. Our full-season processed dataset lives at `data/processed/features_research.csv`, and our mid-season processed dataset lives at `data/processed/features_midseason.csv`. Both files contain 629 rows and 112 columns.

We use one team-season as the unit of analysis. We keep metadata columns for `SEASON`, `TEAM`, `TEAM_ID`, `PLAYOFF_RESULT`, and `PLAYOFF_LABEL`, then append team, context, and player features. We choose 2003-04 as the starting point because it gives us a modern multi-season sample with consistent team and player statistics. We stop at 2023-24 because it is the most recent completed season in our committed artifacts, which keeps every label final and uncontested.

For full-season forecasting, we use complete regular-season statistics. For mid-season forecasting, we use the same schema but filter team and player statistics to season-specific All-Star-break proxy cutoffs through the pipeline described in `docs/midseason_report.md`. We keep the target label unchanged in both settings so that the mid-season task asks a harder and more practical question: how much can we infer about final playoff depth before the regular season finishes?

## Labels

We define six ordered playoff-depth labels:

| Class | Outcome | Count | Share |
| ---: | --- | ---: | ---: |
| 0 | Missed Playoffs | 293 | 46.6% |
| 1 | First Round Exit | 168 | 26.7% |
| 2 | Second Round Exit | 84 | 13.4% |
| 3 | Conference Finals | 42 | 6.7% |
| 4 | Finals Loss | 21 | 3.3% |
| 5 | Champion | 21 | 3.3% |

We preserve the ordered playoff-depth scale because it carries more basketball meaning than a binary target. A second-round team and a champion are both playoff teams, but they imply different roster quality, strategic urgency, and championship probability. We also report macro F1 because accuracy alone can hide poor performance on rare labels such as Finals Loss and Champion.

![Class distribution imbalance](figures/class_distribution_imbalance.png)

Figure 1. We use this figure to show the severe imbalance in our six playoff-depth labels, with missed-playoff teams dominating the dataset and champions appearing only once per season. We generated it from the real processed feature artifacts and label counts summarized in `docs/real_results_summary.md`.

## Features

We build three feature groups. First, we use team-level regular-season performance variables such as wins, losses, win percentage, points, rebounds, assists, steals, blocks, turnovers, field-goal percentage, three-point percentage, free-throw percentage, plus/minus, offensive rating, defensive rating, net rating, and pace. These features capture team strength and playing style.

Second, we add contextual variables. We include a conference-strength proxy so the model can account for season-specific Eastern and Western Conference context. We also include roster continuity because a stable rotation may make regular-season performance more predictive than a team assembled late through trades or disrupted by injuries.

Third, we represent the top eight rotation players by minutes played. For slots P1 through P8, we include production and efficiency variables such as points, rebounds, assists, steals, blocks, minutes, shooting efficiency, and an approximate PER-style impact summary. This fixed roster representation lets every model see not just how the team performed overall, but how production was distributed across the rotation.

We standardize numeric features inside each training fold so that held-out seasons do not leak into preprocessing. The full-season and mid-season datasets keep identical feature schemas, which lets us compare performance changes directly.

## Models

We compare four models:

| Model | Purpose in our project |
| --- | --- |
| Logistic Regression | We use it as a simple linear baseline with interpretable class boundaries. |
| Random Forest | We use it as a strong non-linear tabular baseline that handles feature interactions well. |
| MLP Baseline | We use it as a neural baseline over the flattened feature vector. |
| Attention Model | We use it as our roster-aware neural model with learned weights over top-eight player slots. |

The attention model separates team/context features from player-slot features. It projects each player slot into a learned representation, computes attention weights across the eight slots, pools the weighted roster representation, and concatenates that representation with the team/context vector before classification. This architecture lets our project ask whether explicit roster structure improves forecasting and gives us an interpretable artifact: learned attention weights by rotation slot.

We train neural models with Adam at learning rate 0.001. The committed full-season and mid-season runs use 150 epochs for the MLP baseline and 200 epochs for the attention model, with random seed 42. We keep hyperparameter search lightweight because the dataset is small and season-level overfitting is a serious risk.

## Validation And Evaluation

We use leave-one-season-out cross-validation for both settings. In each fold, we hold out one NBA season, train on all other seasons, fit preprocessing only on the training fold, and evaluate on the held-out season. We then aggregate out-of-fold predictions across all 629 team-seasons.

We report exact accuracy, macro F1, and top-2 accuracy. Exact accuracy measures whether the model predicts the precise playoff-depth class. Macro F1 gives equal weight to every class, so it exposes rare-class failures. Top-2 accuracy measures whether the true label appears in the model's two most likely classes, which matters because adjacent playoff-depth classes can be difficult to separate and still useful for tier-based decision-making.

![Leave-one-season accuracy timeline](figures/leave_one_season_accuracy_timeline.png)

Figure 2. We use this figure to track fold accuracy across held-out seasons, showing that model performance changes substantially from year to year. We generated it from the real fold-level metrics in `results/research_study/fold_metrics.csv`.

![Fold metric variability boxplot](figures/fold_metric_variability_boxplot.png)

Figure 3. We use this figure to summarize season-to-season variability in validation metrics, making the instability of a small seasonal dataset visible. We generated it from the real leave-one-season-out fold metrics saved under `results/research_study/`.

## Results

Our full-season experiment shows that the task is learnable but not solved. Random Forest achieves the best exact accuracy at 68.0% and the best top-2 accuracy at 84.3%. The MLP baseline achieves the best macro F1 at 40.8%, which means it handles the minority classes slightly better overall. The attention model reaches 65.3% accuracy, 39.3% macro F1, and 82.5% top-2 accuracy while also producing attention-weight diagnostics.

| Model | Accuracy | Accuracy Mean +/- SD | Macro F1 | Top-2 Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 60.9% | 60.9% +/- 7.6% | 38.0% | 81.4% |
| Random Forest | 68.0% | 68.0% +/- 6.1% | 36.2% | 84.3% |
| MLP Baseline | 62.8% | 62.8% +/- 6.6% | 40.8% | 81.7% |
| Attention Model | 65.3% | 65.4% +/- 7.6% | 39.3% | 82.5% |

![Full-season Random Forest and Attention results](figures/full_season_results_random_forest_attention.png)

Figure 4. We use this figure to compare the strongest full-season accuracy model, Random Forest, with our interpretable Attention Model across the main metrics. We generated it from `results/research_study/model_comparison.csv`, which stores the real out-of-fold results.

![Research confusion matrices](figures/research_confusion_matrices.png)

Figure 5. We use this figure to show confusion matrices for the full-season models and reveal that errors concentrate among neighboring playoff-depth classes. We generated it from the real out-of-fold predictions in `results/research_study/predictions.csv`.

The full-season attention model performs best on Missed Playoffs, with precision 0.908, recall 0.904, and F1 0.906. It reaches F1 0.616 on First Round Exit and 0.358 on Second Round Exit, but it struggles on the rarest labels. In this run it scores F1 0.000 on Finals Loss and F1 0.261 on Champion, which confirms that a single finalist and champion per season create a difficult rare-class problem.

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Missed Playoffs | 0.908 | 0.904 | 0.906 | 293 |
| First Round Exit | 0.631 | 0.601 | 0.616 | 168 |
| Second Round Exit | 0.372 | 0.345 | 0.358 | 84 |
| Conference Finals | 0.204 | 0.238 | 0.220 | 42 |
| Finals Loss | 0.000 | 0.000 | 0.000 | 21 |
| Champion | 0.240 | 0.286 | 0.261 | 21 |

![Per-class F1 attention vs baseline](figures/per_class_f1_attention_vs_baseline.png)

Figure 6. We use this figure to compare per-class F1 for the attention model against the best baseline by macro F1, making the rare-class tradeoffs easier to inspect. We generated it from `results/research_study/attention_classification_report.csv` and `results/research_study/best_baseline_classification_report.csv`.

![Ordered error by class heatmap](figures/ordered_error_by_class_heatmap.png)

Figure 7. We use this figure to visualize how far predictions move across the ordered playoff-depth scale rather than treating every mistake as equally distant. We generated it from the real full-season prediction artifacts and the ordered label definitions.

We also evaluate the mid-season setting. All models lose information because we restrict features to All-Star-break proxy cutoffs, but the results remain meaningful. Random Forest and the Attention Model tie for best exact accuracy at 60.4%. The Attention Model achieves the best macro F1 at 37.1%, and Random Forest achieves the best top-2 accuracy at 81.9%.

| Model | Accuracy | Accuracy Mean +/- SD | Macro F1 | Top-2 Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 54.8% | 54.8% +/- 7.7% | 32.4% | 74.1% |
| Random Forest | 60.4% | 60.4% +/- 4.2% | 28.6% | 81.9% |
| MLP Baseline | 57.9% | 57.9% +/- 8.1% | 32.6% | 77.6% |
| Attention Model | 60.4% | 60.4% +/- 9.4% | 37.1% | 80.1% |

![Mid-season confusion matrices](figures_midseason/midseason_confusion_matrices.png)

Figure 8. We use this figure to show the mid-season confusion matrices and illustrate how prediction errors increase when we remove late-season information. We generated it from `results/midseason_study/predictions.csv`, which contains real out-of-fold predictions for all 629 team-seasons.

![Full vs mid-season metric drop](figures/full_vs_midseason_metric_drop.png)

Figure 9. We use this figure to compare full-season and mid-season metric changes, showing that every model declines when we limit the feature window. We generated it from the real full-season and mid-season `model_comparison.csv` files.

The attention model has the smallest macro-F1 drop from full-season to mid-season, falling only 2.3 percentage points from 39.3% to 37.1%. Random Forest loses 7.7 macro-F1 points, Logistic Regression loses 5.7 points, and the MLP baseline loses 8.2 points. We interpret this as evidence that explicit roster structure remains useful when team-level aggregates are noisier earlier in the season.

## Interpretability

We use attention weights to inspect how the model uses top-eight rotation slots. These weights do not prove causal player value, but they show which roster slots the model emphasizes when it builds the pooled roster representation. In our runs, the first few rotation slots generally receive higher attention, which matches the basketball intuition that high-end talent matters heavily in the playoffs. Later slots still receive nonzero weight, so our model also recognizes that depth and balance contribute to playoff outcomes.

![Research attention weights](figures/research_attention_weights.png)

Figure 10. We use this figure to show the full-season attention model's learned average weights across the top-eight rotation slots. We generated it from the trained attention model diagnostics created by the real full-season study pipeline.

![Mid-season attention weights](figures_midseason/midseason_attention_weights.png)

Figure 11. We use this figure to show the same attention-weight diagnostic for the mid-season model, allowing us to compare roster emphasis before the regular season ends. We generated it from the real mid-season attention artifacts in `results/midseason_study/` and `docs/figures_midseason/`.

We also visualize team-season embeddings with t-SNE. These plots show that missed-playoff teams separate more clearly than the deepest playoff classes. Champions, Finals teams, and conference finalists overlap heavily, which explains why exact rare-class prediction remains difficult.

![Research t-SNE projection](figures/research_tsne.png)

Figure 12. We use this figure to project full-season team-season feature representations into two dimensions and color them by playoff-depth label. We generated it from the real full-season feature matrix and model visualization pipeline.

![Research t-SNE cluster analysis](figures/research_tsne_cluster_analysis.png)

Figure 13. We use this figure to add cluster-oriented analysis to the full-season t-SNE view, highlighting where playoff-depth groups separate and where they overlap. We generated it from the same real full-season feature artifacts used for the t-SNE diagnostic.

![Mid-season t-SNE projection](figures_midseason/midseason_tsne.png)

Figure 14. We use this figure to project mid-season team-season features and show that partial-season data preserves some structure while increasing overlap among playoff teams. We generated it from `data/processed/features_midseason.csv` and the mid-season visualization pipeline.

## Finals And Champion Diagnostics

Because the Finals Loss and Champion classes have only 21 examples each, we add dedicated diagnostics for those labels. These figures help us inspect whether models recognize the deepest playoff teams or merely classify most elite teams into nearby classes. We treat these plots as error-analysis artifacts rather than claims that the model can reliably identify champions.

![Finals team prediction heatmap](figures/finals_team_prediction_heatmap.png)

Figure 15. We use this figure to show how models predicted teams that reached the Finals, making champion and finalist confusion visible at the team level. We generated it from the real full-season out-of-fold predictions and final playoff labels.

![Finals case study predictions](figures/finals_case_study_predictions.png)

Figure 16. We use this figure to present specific Finals-team prediction cases so we can inspect when the models placed elite teams in the correct or adjacent outcome classes. We generated it from the real prediction rows in `results/research_study/predictions.csv`.

![Finals detection by model](figures/finals_detection_by_model.png)

Figure 17. We use this figure to compare how well each model detects the deepest playoff outcomes, emphasizing the gap between overall accuracy and rare-class recognition. We generated it from the real full-season classification outputs and label-filtered prediction artifacts.

## Fairness Audit

We include a coarse market-size fairness audit using the attention model's predictions. We group teams into large-market and small-market buckets and compare actual mean label, predicted mean label, and average absolute error. We do not claim this audit proves fairness because market size is only one proxy and playoff success itself is not evenly distributed across markets. We use it as a sanity check and as a reminder that sports models can absorb structural patterns from the league.

For the full-season setting, large-market teams have average actual label 1.257, average predicted label 1.305, and average absolute error 0.533. Small-market teams have average actual label 0.924, average predicted label 0.988, and average absolute error 0.475. The full-season error gap is 0.058. For the mid-season setting, the large-market error is 0.710, the small-market error is 0.556, and the error gap widens to 0.153.

![Market-size fairness audit](figures/market_size_fairness_audit.png)

Figure 18. We use this figure to summarize the market-size audit by comparing prediction error across large-market and small-market teams. We generated it from the real fairness JSON artifacts in `results/research_study/market_fairness.json` and `results/midseason_study/market_fairness.json`.

## Generated Artifacts

Our project generates and commits reproducible artifacts so reviewers can inspect the results without rerunning the full pipeline. The primary full-season artifacts are `results/research_study/model_comparison.csv`, `summary_metrics.json`, `fold_metrics.csv`, `predictions.csv`, `attention_classification_report.csv`, `best_baseline_classification_report.csv`, and `market_fairness.json`. The parallel mid-season artifacts live in `results/midseason_study/` with the same filenames.

The full-season figures live in `docs/figures/`, and the mid-season figures live in `docs/figures_midseason/`. We can regenerate the additional presentation figures with `.\.venv\Scripts\python.exe scripts\generate_additional_figures.py`. We can reproduce the full research run with `python run_research_study.py --epochs-mlp 150 --epochs-attention 200 --lr 0.001 --seed 42`, and we can reproduce the mid-season run with `python run_midseason_study.py --epochs-mlp 150 --epochs-attention 200 --lr 0.001 --seed 42`.

# Limitations

Our project has several limitations. First, the dataset remains small for a six-class deep learning task because each season contributes only one champion and one Finals loser. Even across 21 seasons, the rarest classes have only 21 examples each, so macro F1 and rare-class diagnostics matter more than headline accuracy.

Second, our features summarize box-score production, advanced efficiency, roster continuity, conference context, and top-eight rotation players, but they do not include play-by-play, tracking data, lineup combinations, salary, injuries, rest, playoff matchup paths, coaching adjustments, or late-series tactical changes. Those missing signals likely explain many errors among champions, finalists, and conference finalists. Our mid-season setting also cannot know later trades, injuries, rotation changes, or late-season form.

Third, our attention weights are interpretable but not causal. They tell us which rotation slots the model used, not which players caused wins or playoff advancement. Analysts should therefore treat the attention plots as model diagnostics, not as player rankings.

Fourth, our market-size audit is intentionally narrow. We do not encode player demographics, race, nationality, salary, endorsement value, or media popularity, but team-level sports data can still reflect structural inequalities. We audit large-market and small-market errors as a basic check, but a full fairness study would need richer variables, stakeholder review, and clearer definitions of harm.

Fifth, we keep hyperparameter tuning modest to protect reproducibility and reduce overfitting. A larger production study could explore calibrated probabilities, temporal models, matchup-aware playoff simulations, richer player embeddings, imbalance-aware losses, and season-aware oversampling methods inspired by SMOTE [12] and ADASYN [14]. Our strongest conclusion is therefore measured: regular-season and rotation statistics contain real playoff-depth signal, but they do not reliably identify NBA champions as a distinct class.

## Acknowledgment

We thank the course instructors of Deep Learning at the University of Colorado Denver for guidance throughout our project. We also acknowledge the `nba_api` project maintainers for providing open access to NBA statistics.

## References

[1] R. Khanmohammadi, S. Saba-Sadiya, S. Esfandiarpour, T. Alhanai, and M. M. Ghassemi, "MambaNet: A hybrid neural network for predicting the NBA playoffs," SN Comput. Sci., vol. 5, no. 5, 2024, doi: 10.1007/s42979-024-02977-0.

[2] K. Zhao, C. Du, and G. Tan, "Enhancing basketball game outcome prediction through fused graph convolutional networks and random forest algorithm," Entropy, vol. 25, no. 5, 2023, doi: 10.3390/e25050765.

[3] W. Guan, N. Javed, and P. Lu, "NBA2Vec: Dense feature representations of NBA players," arXiv:2302.13386, 2023.

[4] Y. Ouyang et al., "Integration of machine learning XGBoost and SHAP models for NBA game outcome prediction," PLoS ONE, vol. 19, no. 7, e0307478, 2024, doi: 10.1371/journal.pone.0307478.

[5] C. Rios, L. Han, A. Baimagambetov, and N. Polatidis, "Long-sequence LSTM modeling for NBA game outcome prediction using a novel multi-season dataset," arXiv:2512.08591, 2025, doi: 10.48550/arXiv.2512.08591.

[6] S. Z. Ibrahim, A. M. Reza, L. W. Kean, N. A. Ab. Aziz, and S. N. M. Sayed Ismail, "Machine learning insights into basketball championship predictions: An analytical comparison," in Lecture Notes in Bioengineering, Springer, 2024, pp. 275-285, doi: 10.1007/978-981-97-3741-3_26.

[7] G. D. S. Teno, C. Wang, N. Carlsson, and P. Lambrix, "Predicting season outcomes for the NBA," in Machine Learning and Data Mining for Sports Analytics, Springer, 2022, pp. 129-142, doi: 10.1007/978-3-031-02044-5_11.

[8] R. P. Bunker and F. Thabtah, "A machine learning framework for sport result prediction," Appl. Comput. Inform., vol. 15, no. 1, pp. 27-33, 2019, doi: 10.1016/j.aci.2017.09.005.

[9] M. Yeung, "Multiple machine learning algorithms-based NBA team playoffs prediction," ITM Web Conf., vol. 70, 04024, 2025, doi: 10.1051/itmconf/20257004024.

[10] J. Perricone, S. Shaw, and J. Swiechowicz, "Predicting results for professional basketball using NBA API data," Stanford Univ., Stanford, CA, USA, Tech. Rep. CS229, 2016.

[11] Y. Wang, W. Liu, and X. Liu, "Explainable AI techniques with application to NBA gameplay prediction," Neurocomputing, vol. 483, pp. 59-71, 2022, doi: 10.1016/j.neucom.2022.01.098.

[12] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," J. Artif. Intell. Res., vol. 16, pp. 321-357, 2002, doi: 10.1613/jair.953.

[13] D. Elreedy and A. F. Atiya, "A comprehensive analysis of synthetic minority oversampling technique (SMOTE) for handling class imbalance," Inf. Sci., vol. 505, pp. 32-64, 2019, doi: 10.1016/j.ins.2019.07.070.

[14] H. He, Y. Bai, E. A. Garcia, and S. Li, "ADASYN: Adaptive synthetic sampling approach for imbalanced learning," in Proc. IEEE Int. Joint Conf. Neural Networks, 2008, pp. 1322-1328, doi: 10.1109/IJCNN.2008.4633969.

[15] A. Vaswani et al., "Attention is all you need," in Proc. 31st Conf. Neural Inf. Process. Syst., 2017, pp. 5998-6008.

[16] P. Mokha, "nba_api: An API client package to access NBA.com APIs," GitHub repository, 2018. [Online]. Available: https://github.com/swar/nba_api
