# Predicting NBA Postseason Depth from Regular-Season Team and Rotation Statistics Using Deep Learning

Pranav Kumar Kaliaperumal, Shrujal Mogadati, and Disha Srinivasa
Department of Computer Science, University of Colorado Denver, Denver, CO 80217 USA

## Abstract

We investigate whether deep learning models can forecast a team's eventual NBA playoff depth from regular-season team efficiency, contextual roster features, and rotation-level player statistics. Using data from the 2003-04 through 2023-24 seasons (629 team-seasons), we compare four models under leave-one-season-out cross-validation: logistic regression, random forest, a multilayer perceptron baseline, and an attention-based roster model that learns player-importance weights. In the full-season setting, random forest achieves the best overall accuracy at 68.0% and top-2 accuracy at 84.3%, while the MLP baseline achieves the highest macro F1 at 40.8%. Our attention model reaches 65.3% accuracy and 39.3% macro F1 while providing interpretable player-importance diagnostics. We also evaluate a harder mid-season forecasting setting that uses only statistics available around the All-Star break; all models decline in performance, yet the attention model exhibits the smallest macro-F1 drop. These results show that postseason forecasting is learnable but difficult, with severe class imbalance representing the primary barrier to robust champion prediction. We include a market-size fairness audit and discuss ethical considerations regarding equitable prediction across large-market and small-market franchises.

*Keywords*-NBA, sports analytics, deep learning, attention mechanism, player embeddings, postseason prediction, class imbalance, machine learning

## I. Introduction

Every NBA season, thirty teams compete for the championship, yet only a handful are genuine contenders. Front offices, analysts, and fans spend considerable effort debating which teams will advance deepest into the playoffs. We address a concrete forecasting question: given how a team performs during the regular season, can we predict how far that team will advance in the postseason?

### A. Problem Statement

We frame this as a six-class classification task. Each team-season receives one of six labels based on its final playoff outcome: Missed Playoffs, First Round Exit, Second Round Exit, Conference Finals, Finals Loss, or Champion. This multi-class formulation preserves playoff depth rather than collapsing the task into a binary champion-versus-non-champion decision. While the multi-class approach is more informative for analysts who want to distinguish pretenders from serious contenders, it is also substantially harder because the deepest outcomes occur only once or twice per season.

The forecasting problem is challenging for several reasons. First, the NBA regular season consists of eighty-two games, but the playoffs are four best-of-seven rounds with matchup-specific adjustments and much shorter rotations. A team's regular-season dominance does not guarantee postseason success because playoff basketball involves more focused scouting, more intense defense, and opponent-specific strategies that regular-season statistics may not capture. Second, injuries, trades, and late-season form changes can dramatically alter a team's playoff trajectory after the regular season concludes. Third, the class distribution is highly imbalanced: roughly half of all team-seasons miss the playoffs, while only one team per season wins the championship.

### B. Motivation and Application Value

Predicting which teams will contend for the title is one of the most discussed and practically consequential topics in basketball. Playoff-depth forecasts can support front-office planning, trade-deadline decisions, roster evaluation, media analysis, fan engagement, and long-horizon business decisions such as ticket pricing, broadcast planning, and sponsorship valuation [1]. A team that can estimate whether it is most likely a first-round exit, conference-finals contender, or true championship threat faces different strategic choices: it may trade future draft capital for immediate help, preserve flexibility, prioritize player development, or avoid overreacting to a misleading regular-season record.

The applied importance is especially strong because playoff success is not a simple extension of regular-season wins. Postseason basketball compresses rotations, magnifies star creation, rewards matchup-specific adaptability, and punishes weaknesses that may be hidden across an 82-game schedule. This makes playoff-depth forecasting interesting for both basketball analytics and machine learning. The model must learn from structured team statistics, player production, and contextual signals while recognizing that the final outcome can hinge on small samples, injuries, seeding paths, and opponent matchups. A useful model therefore should not only chase exact champion prediction; it should help identify plausible playoff tiers and communicate uncertainty when several outcomes remain close.

From a machine learning perspective, most existing work on NBA prediction focuses on individual game outcomes and relies on traditional methods such as random forests and XGBoost [2]-[5]. Fewer studies apply deep learning to season-level playoff depth, and fewer still combine team aggregates with explicit roster structure. This is the gap we aim to fill. We hypothesize that championship outcomes are not explained by team aggregates alone. Two strong regular-season teams may look similar in summary metrics while differing meaningfully in how production is distributed across stars, secondary creators, shooters, and rotation depth. Our attention-based model tests whether a learned weighting over the top eight rotation players adds predictive signal and interpretability beyond standard tabular baselines.

### C. Ethical Considerations

We design our model to use performance-based features such as team statistics, records, ratings, and rotation-player box-score summaries. We do not encode player demographics, nationality, race, socioeconomic background, salary, endorsement value, or media popularity. Even so, sports data can still reflect structural inequities. Team market size, organizational resources, national coverage, free-agency appeal, and historical franchise prestige may correlate with both roster strength and public perception. For that reason, we include a coarse market-size fairness audit using the attention model predictions and report the full-season and mid-season audit values from `results/research_study/market_fairness.json` and `results/midseason_study/market_fairness.json`.

Responsible use is central to this project. The model should be treated as a decision-support and analysis tool, not as an authoritative ranking of teams, players, coaches, or markets. Forecasts can influence narratives around athletes and franchises, so users should avoid interpreting predictions as judgments about player worth, team culture, fan bases, or city-level merit. Attention weights should also be interpreted carefully: they indicate which top-eight rotation slots the model used for prediction, not which individual players causally produced wins or losses.

Uncertainty and class imbalance require explicit caution. Champion and Finals Loss each appear only once per season, producing severe minority-class scarcity. This means exact champion predictions are fragile, and strong accuracy on common classes can coexist with weak rare-class recognition. We therefore report macro F1 and top-2 accuracy alongside exact accuracy, discuss the rare-class failures directly, and avoid overclaiming that the system can reliably identify champions. Demographic, player, and team-market concerns should be revisited in future work with richer fairness variables and stakeholder review, especially before applying any model to high-stakes roster, employment, betting, or public-reputation decisions.

### D. Contributions

Our project makes the following contributions. First, we build a reproducible end-to-end pipeline that collects real NBA data via the nba_api library, engineers team and player features, and evaluates models under a season-aware validation protocol. Second, we design an attention-based architecture that learns which rotation slots matter most for postseason forecasting, offering interpretable player-importance weights that analysts can inspect. Third, we compare classical machine learning and deep learning models fairly under leave-one-season-out cross-validation, showing that the problem is learnable but that class imbalance remains the main barrier to champion prediction. Fourth, we extend the study to a mid-season forecasting setting and quantify how much performance drops when only partial-season information is available. Fifth, we include a fairness audit and discuss limitations transparently, setting a realistic benchmark that future architectures must beat consistently.

## II. Related Work

Prior sports analytics research generally follows two complementary paths. One line models games directly, using team box scores, recent performance, player availability, or temporal sequences to predict a single matchup. Another line builds player or team representations from box-score, play-by-play, lineup, or tracking data and then uses machine learning to support scouting, strategy, or season forecasting. Across these studies, classical models such as logistic regression, random forests, gradient boosting, and k-nearest neighbors remain competitive because sports datasets are often structured, noisy, and relatively small. Deep learning methods can add value when the representation contains temporal, graph, or roster structure, but they must be evaluated against strong tabular baselines and interpreted cautiously.

### A. Game-Level Prediction

Khanmohammadi et al. [1] proposed MambaNet, a hybrid neural network combining CNNs and RNNs to predict NBA playoff game outcomes. They achieved AUC scores between 0.72 and 0.82, but their focus was on individual playoff games rather than season-level championship prediction. Their work demonstrates that temporal dependencies in game sequences improve accuracy, yet it does not address how mid-season team snapshots forecast final playoff depth. Zhao et al. [2] used graph convolutional networks to model relationships between NBA teams for game prediction, reaching 71.54% accuracy. Their approach captures team interactions but does not examine player-level features or roster makeup. Ouyang et al. [4] integrated XGBoost and SHAP models for NBA game outcome prediction, demonstrating the value of explainable techniques in sports analytics. Their quantitative analysis showed which box-score features most influence single-game results. Rios et al. [5] applied long-sequence LSTM modeling to NBA game outcome prediction, showing that temporal dependencies improve accuracy beyond what traditional ML achieves. These game-level studies establish that deep learning can model basketball dynamics, but they do not answer the season-level question we pose.

### B. Player and Team Representations

Guan et al. [3] created NBA2Vec, which learns vector representations of NBA players from play-by-play data in a manner analogous to Word2Vec. They showed that these embeddings predict playoff series outcomes, but they did not extend the idea to season-level championship prediction. Their work is foundational for our approach because it demonstrates that player embeddings carry predictive signal. Ibrahim et al. [6] studied machine learning approaches to basketball championship prediction, reinforcing that season-level forecasting is feasible but difficult with limited samples. Teno et al. [7] addressed predicting season outcomes for the NBA using machine learning and data mining techniques, which is closer to our goal but did not use deep learning or player embeddings. They found that ensemble methods outperform single classifiers when forecasting season-level results.

### C. Machine Learning in Sports Forecasting

Bunker and Thabtah [8] surveyed machine learning frameworks for sport result prediction, highlighting that ensemble methods often outperform single classifiers. Their meta-analysis supports our decision to include random forest as a strong baseline. Yeung [9] compared logistic regression, k-nearest neighbors, random forest, and elastic net for NBA playoff qualification, finding that random forest achieved the highest ROC-AUC score of 0.841. Their research demonstrates the power of ensemble learning in sports but limits the task to binary playoff qualification rather than multi-class depth prediction. Perricone et al. [10] used traditional ML algorithms on NBA API data for outcome prediction, establishing that accessible public data supports competent forecasting models. Ni and Lee [11] conducted a comparative study of machine learning models for NCAA tournament games, finding that simpler models sometimes outperform complex ones when data is limited. Wang et al. [12] applied explainable AI techniques to NBA gameplay prediction, demonstrating that SHAP and LIME improve model transparency for coaching staff. Their work supports our emphasis on interpretability. Tsagris et al. [13] showed how half-time statistics can predict NBA game outcomes, reinforcing that intermediate snapshots carry signal that models can exploit.

### D. Class Imbalance and Sampling Techniques

Our dataset suffers from severe class imbalance because only one champion and one finals loser exist per season. Chawla et al. [14] introduced SMOTE, a synthetic minority oversampling technique that creates synthetic examples rather than replicating existing ones, improving classifier performance in ROC space. Their experiments on multiple datasets showed that SMOTE combined with under-sampling outperforms plain under-sampling for minority class recognition. Elreedy and Atiya [15] provided a comprehensive analysis of SMOTE variants for handling class imbalance, identifying conditions under which different oversampling strategies succeed. He et al. [16] proposed ADASYN, an adaptive synthetic sampling approach that focuses on harder minority examples. We do not apply SMOTE in our main experiments because our season-aware validation protocol makes synthetic sample generation across seasons problematic; creating synthetic team-seasons could violate the temporal structure we aim to preserve. However, we cite these works to contextualize the imbalance challenge and to suggest future directions.

### E. Attention Mechanisms

Vaswani et al. [17] introduced the Transformer architecture based entirely on attention mechanisms, eliminating recurrence and enabling parallel training. Their self-attention mechanism allows each element in a sequence to attend to every other element, which we adapt to our roster setting where each player slot attends to the others. Attention has since been applied across domains from natural language processing to computer vision; our work applies a simplified attention pooling to sports rosters, learning which players in a rotation deserve more influence when predicting team success.

### F. Summary of Gap

Existing literature either focuses on game-level prediction rather than season-level depth forecasting, or uses traditional machine learning without exploring deep learning architectures that explicitly model roster structure. Our project fills this gap by combining real multi-season NBA data from 2003-04 through 2023-24 with six-class playoff-depth labels, fixed top-eight rotation-player features, attention-based roster interpretability, and leave-one-season-out validation. The resulting task is more granular than binary playoff qualification and more season-oriented than individual game prediction.

The project's contribution is also unique in its extension beyond a single full-season snapshot. We evaluate both complete regular-season features and a harder mid-season setting built around season-specific All-Star-break proxy dates, while preserving the same final playoff-depth labels. This creates a practical trade-deadline-style forecasting benchmark. The full-season model comparison is stored in `results/research_study/model_comparison.csv`, the mid-season comparison is stored in `results/midseason_study/model_comparison.csv`, and the corresponding attention diagnostics are visualized in `docs/figures/research_attention_weights.png` and `docs/figures_midseason/midseason_attention_weights.png`.

## III. Methodology

### A. Data Collection

We collect data using the nba_api Python library [18], which provides programmatic access to NBA.com statistics. For each season from 2003-04 through 2023-24, we pull three categories of data. Team-level regular-season statistics come from the LeagueDashTeamStats endpoint and include box-score aggregates such as wins, losses, points, rebounds, assists, steals, blocks, and shooting percentages. Advanced team metrics come from TeamEstimatedMetrics and include offensive rating, defensive rating, net rating, and pace. Player statistics come from LeagueDashPlayerStats and include per-game and advanced metrics for every player in the league. We also curate historical playoff labels that record the deepest round each team reached. The unit of analysis is one team-season, yielding 629 total rows (30 teams per season across 21 seasons, minus a few expansions and lockout years).

We choose the 2003-04 season as our starting point because it marks the beginning of the modern analytics era in the NBA, when tracking data and advanced metrics became more widely adopted. The 2023-24 season is the most recent complete season available at the time of our study. We intentionally exclude the current in-progress season to ensure that all labels are final and uncontested.

### B. Label Definition

We define six ordered classes for playoff depth:

- Class 0: Missed Playoffs
- Class 1: First Round Exit
- Class 2: Second Round Exit
- Class 3: Conference Finals
- Class 4: Finals Loss
- Class 5: Champion

These labels intentionally preserve playoff depth. A binary champion-versus-non-champion task would be simpler but far less informative for analysts who want to distinguish contenders from pretenders. The ordered nature of the classes also means that adjacent classes are often more similar than distant classes; a team that reaches the Conference Finals is typically closer in quality to a Finals team than to a lottery team.

### C. Feature Engineering

Our feature matrix combines three groups of predictors.

Team-level performance variables capture overall quality and style. These include win percentage, wins, losses, points, rebounds, assists, steals, blocks, turnovers, field-goal percentage, three-point percentage, free-throw percentage, plus/minus, offensive rating, defensive rating, net rating, and pace. These metrics summarize how a team performed across the entire regular season and provide the strongest baseline signal for playoff success.

Contextual variables add environmental information that may moderate the predictive power of team statistics. Conference strength proxy captures the competitive context of the Eastern or Western Conference in a given season; some seasons feature historically strong conferences where good teams miss the playoffs, while other seasons feature weaker conferences where mediocre teams advance. Roster continuity summarizes how stable a team's rotation remains across the data window; regular-season quality may be more predictive when it comes from a stable roster rather than a team whose personnel shifted substantially due to trades or injuries.

Player-level variables represent the top eight rotation players ranked by minutes played. Each slot P1 through P8 contains points, rebounds, assists, steals, blocks, turnovers, field-goal percentage, three-point percentage, free-throw percentage, minutes, and an approximate PER-style impact score. Using fixed player slots gives our neural models a consistent representation of rotation structure. The first slot corresponds to the highest-minute player, so the representation is reproducible from data rather than relying on manual superstar labels. This design choice means our model does not need to know whether a player is a star or a role player; it learns the importance of each slot from the data.

### D. Model Architecture

We evaluate four models that span linear, nonlinear, and neural paradigms.

Logistic Regression serves as a linear baseline to measure how much signal exists in a simple regularized decision surface. We include L2 regularization and tune the regularization strength via cross-validation.

Random Forest serves as a nonlinear tabular baseline that can capture interactions among team and player features without neural training. We set 200 estimators and use entropy criterion. Ensemble tree methods are known to perform well on structured sports data [8, 9], and random forest in particular has been identified as a top performer for NBA playoff prediction tasks.

MLP Baseline is a feed-forward neural network over the full flattened feature vector. It has two hidden layers with 128 and 64 units, ReLU activation, and dropout regularization at rate 0.3. We train with cross-entropy loss and apply class weights inversely proportional to class frequency to partially mitigate imbalance.

Attention Model is our proposed architecture. It has two main parts. First, each player slot's statistics pass through a shared subnetwork that produces a compact player embedding of dimension 16. These embeddings are combined using attention pooling, which lets the model learn which players matter most. Specifically, we compute attention scores by passing each player embedding through a learned linear layer, apply softmax across the eight slots, and form a weighted sum of embeddings. The combined roster representation is then concatenated with team-level and contextual features and passed through fully connected layers (128 and 64 units with ReLU and dropout) to predict the playoff outcome. The attention mechanism computes a weighted sum of player embeddings where the weights are learned from the embeddings themselves, analogous to single-head self-attention [17].

### E. Training and Evaluation Protocol

We use leave-one-season-out cross-validation. For each of 21 folds, we hold out one full NBA season for testing and train on all remaining seasons. This design is stronger than random row-level splitting because it reduces leakage across seasons and better reflects real forecasting behavior: applying patterns learned from prior seasons to a future season. In financial and sports forecasting, temporal leakage is a serious threat to validity; our protocol eliminates it by construction.

Neural models use the Adam optimizer. The MLP baseline trains for 150 epochs and the attention model trains for 200 epochs, both with learning rate 0.001. We set random seed 42 for reproducibility. All experiments run on CPU within a local workspace. We apply early stopping based on validation loss computed within each training fold to prevent overfitting.

We report three main metrics. Accuracy measures the share of team-seasons whose exact playoff-depth class is predicted correctly. Macro F1 is the unweighted mean F1 across all six classes; this is crucial because rare classes such as Finals Loss and Champion should influence the score rather than being overwhelmed by missed-playoff teams. Top-2 accuracy records whether the correct class appears among the model's two highest-probability classes. This is useful for playoff-depth forecasting because adjacent classes are often difficult to separate, and a model that ranks the true outcome second may still be informative for analysts.

### F. Interpretability

We visualize the learned attention weights to see which rotation slots the model considers most important for championship contention. We also use t-SNE with perplexity 30 to project team-season feature vectors into two dimensions, providing a qualitative view of whether championship teams cluster together in feature space. These visualizations are diagnostic rather than evaluative; they help us understand how the model uses roster information rather than proving causal importance.

## IV. Experiments and Results

All reported result metrics in this section, along with the summary claims repeated in the Abstract and Conclusion, come from the generated artifacts under `results/research_study/` and `results/midseason_study/`. The primary comparison files are `results/research_study/model_comparison.csv` and `results/midseason_study/model_comparison.csv`; fold-level results are in `results/research_study/fold_metrics.csv` and `results/midseason_study/fold_metrics.csv`; per-class reports are in `results/research_study/attention_classification_report.csv`, `results/research_study/best_baseline_classification_report.csv`, `results/midseason_study/attention_classification_report.csv`, and `results/midseason_study/best_baseline_classification_report.csv`; fairness audit values are in `results/research_study/market_fairness.json` and `results/midseason_study/market_fairness.json`.

### A. Dataset Characteristics

Table I shows the class distribution, derived from `data/processed/features_research.csv` and mirrored in `data/processed/features_midseason.csv`. The Missed Playoffs class dominates with 46.6% of team-seasons, while Champion and Finals Loss each have only 3.3%. This distribution explains why accuracy alone is insufficient: a naive classifier that always predicts Missed Playoffs would achieve 46.6% accuracy while failing completely on the outcomes that matter most to analysts and fans.

TABLE I. Class Distribution Across 629 Team-Seasons

| Class | Outcome | Count | Share |
|---|---|---|---|
| 0 | Missed Playoffs | 293 | 46.6% |
| 1 | First Round Exit | 168 | 26.7% |
| 2 | Second Round Exit | 84 | 13.4% |
| 3 | Conference Finals | 42 | 6.7% |
| 4 | Finals Loss | 21 | 3.3% |
| 5 | Champion | 21 | 3.3% |

### B. Full Regular-Season Experiment

Table II presents the main results for the full-season setting. Random Forest achieves the best exact accuracy at 68.0% and the best top-2 accuracy at 84.3%. The MLP baseline achieves the highest macro F1 at 40.8%, suggesting better balance across minority classes. The attention model reaches 65.3% accuracy, 39.3% macro F1, and 82.5% top-2 accuracy, making it competitive while also producing interpretable roster-weight artifacts.

TABLE II. Full-Season Model Comparison (Out-of-Fold)

| Model | Accuracy | Acc. Mean +/- SD | Macro F1 | Top-2 Acc. |
|---|---|---|---|---|
| Logistic Regression | 60.9% | 60.9% +/- 7.6% | 38.0% | 81.4% |
| Random Forest | 68.0% | 68.0% +/- 6.1% | 36.2% | 84.3% |
| MLP Baseline | 62.8% | 62.8% +/- 6.6% | 40.8% | 81.7% |
| Attention Model | 65.3% | 65.4% +/- 7.6% | 39.3% | 82.5% |

The fold-level metrics show that performance varies substantially by season. For Random Forest, fold accuracy ranges from 56.7% in 2012-13 and 2022-23 to 80.0% in 2016-17. For the Attention Model, fold accuracy ranges from 50.0% in 2022-23 to 80.0% in 2016-17. The strongest season for macro F1 is also 2016-17: the MLP baseline reaches 68.2% macro F1 and the Attention Model reaches 68.1% macro F1. This does not mean the task is solved; rather, some seasons exhibit cleaner separation among playoff outcomes than others. Seasons with unusual playoff paths, injuries, trades, or compressed team quality naturally make this type of classifier less stable.

Top-2 accuracy is consistently much higher than exact accuracy across all models. Random Forest averages 84.3% top-2 accuracy, and the attention model averages 82.5%. This indicates that models often place teams near the correct playoff tier even when they miss the exact class. For practical applications such as media analysis or front-office planning, top-2 accuracy may be nearly as valuable as exact accuracy because it correctly identifies the neighborhood of outcomes.

### C. Per-Class Analysis

Table III shows the per-class results for the Attention Model. The Missed Playoffs class is by far the easiest, with precision 0.908 and recall 0.904, because it has the most examples and is often separable by regular-season strength alone. First Round Exit is moderately learnable at precision 0.631 and recall 0.601. Conference Finals, Finals Loss, and Champion remain difficult because the number of examples is tiny and because elite teams can be separated by factors not fully represented in our feature matrix, such as injuries, matchup paths, trade timing, and late-season form.

TABLE III. Attention Model Per-Class Results (Full Season)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Missed Playoffs | 0.908 | 0.904 | 0.906 | 293 |
| First Round Exit | 0.631 | 0.601 | 0.616 | 168 |
| Second Round Exit | 0.372 | 0.345 | 0.358 | 84 |
| Conference Finals | 0.204 | 0.238 | 0.220 | 42 |
| Finals Loss | 0.000 | 0.000 | 0.000 | 21 |
| Champion | 0.240 | 0.286 | 0.261 | 21 |

Notably, the attention model scores zero on Finals Loss in this run, indicating it did not successfully isolate losing finalists as a distinct class. Many such teams likely resemble champions, conference finalists, or other elite playoff teams in regular-season statistics. The macro average F1 of 0.393 reflects the severe difficulty of the rarest classes.

The best baseline by macro F1 is the MLP Baseline. Its per-class results show higher recall on Champion (0.286) and Finals Loss (0.095) than the attention model, though both struggle with the rarest classes. The MLP baseline achieves precision 0.353 on Champion, compared to the attention model's 0.240, suggesting that the simpler neural network finds somewhat better decision boundaries for the champion class.

### D. Mid-Season Forecasting Extension

We also evaluate a harder setting that uses only statistics available around season-specific All-Star-break proxy dates. The label remains the final postseason outcome, so the task asks whether partial-season information can forecast eventual playoff depth. This setting is more realistic for trade-deadline decisions because front offices must evaluate their contender status with incomplete information.

TABLE IV. Mid-Season Model Comparison (Out-of-Fold)

| Model | Accuracy | Acc. Mean +/- SD | Macro F1 | Top-2 Acc. |
|---|---|---|---|---|
| Logistic Regression | 54.8% | 54.8% +/- 7.7% | 32.4% | 74.1% |
| Random Forest | 60.4% | 60.4% +/- 4.2% | 28.6% | 81.9% |
| MLP Baseline | 57.9% | 57.9% +/- 8.1% | 32.6% | 77.6% |
| Attention Model | 60.4% | 60.4% +/- 9.4% | 37.1% | 80.1% |

All models lose accuracy in the mid-season setting, which is expected because features contain less information. Random Forest and the Attention Model tie for best exact accuracy at 60.4%. The Attention Model achieves the best macro F1 at 37.1% and has the smallest macro-F1 drop from full-season to mid-season (only 2.2 percentage points), suggesting its roster-aware representation retains more signal when team aggregates are noisier. Random Forest remains strongest on top-2 accuracy at 81.9%.

The mid-season per-class results show that all models struggle more with every class. The attention model's Champion precision drops from 0.240 in the full-season setting to 0.185 in the mid-season setting, while its recall drops from 0.286 to 0.238. This confirms that champion prediction requires as much regular-season data as possible.

### E. Comparison Across Settings

Table V summarizes the performance change from full-season to mid-season. All four models lose exact accuracy and macro F1. The Attention Model has the smallest macro-F1 decline (2.2 points), while Random Forest suffers the largest macro-F1 decline (7.7 points). This pattern suggests that the attention mechanism's explicit roster modeling provides some robustness when aggregate team statistics are noisier, because player-level patterns may stabilize earlier in the season than team-level summaries.

TABLE V. Full-Season vs. Mid-Season Performance Change

| Model | Acc. Change | Macro F1 Change | Top-2 Change |
|---|---|---|---|
| Logistic Regression | -6.0% | -5.7% | -7.3% |
| Random Forest | -7.6% | -7.7% | -2.4% |
| MLP Baseline | -4.9% | -8.2% | -4.1% |
| Attention Model | -4.9% | -2.2% | -2.4% |

### F. Visual Diagnostics

The confusion matrices for all models reveal that mistakes often occur between adjacent playoff-depth tiers, as shown in `docs/figures/research_confusion_matrices.png` and `docs/figures_midseason/midseason_confusion_matrices.png`. Models rarely confuse Missed Playoffs with Champion, but they frequently confuse Conference Finals with Second Round Exit or Finals Loss. This pattern is consistent with the intuition that teams near the top are statistically similar in regular-season metrics and are separated by factors such as playoff experience, clutch performance, and matchup luck that our features do not capture.

The attention weight profiles in `docs/figures/research_attention_weights.png` and `docs/figures_midseason/midseason_attention_weights.png` show that the model does not weight every roster slot equally. The first two to three slots typically receive higher weights, consistent with the intuition that high-end talent matters disproportionately for playoff success. Later slots still contribute non-negligible weight, suggesting depth and balance also play a role. This pattern supports our original hypothesis that roster structure matters beyond team aggregates.

The t-SNE projections in `docs/figures/research_tsne.png` and `docs/figures_midseason/midseason_tsne.png` show partial clustering: missed-playoff teams tend to separate from playoff teams, but the deepest playoff classes overlap substantially. This visualization confirms that the feature space has structure but that the rarest classes are not linearly separable, which explains why all models struggle with Champion and Finals Loss.

## V. Discussion

### A. Interpretability and Player Importance

The attention model provides learned roster weights across the top eight player slots. These weights offer a direct diagnostic: the model can assign different influence to different rotation positions rather than treating every player slot as equally important. While attention weights are not a perfect causal explanation, they do reveal how the roster representation informs predictions. In our runs, the attention model's predictive gain over simpler baselines is modest, but the interpretability artifact helps explain how roster information is being used. This aligns with findings in explainable sports analytics that model transparency matters for stakeholder trust [4, 12].

For front offices, these weights could inform roster construction questions. If the model consistently assigns the highest weights to the first two rotation slots, this supports the conventional wisdom that acquiring star talent is the most reliable path to contention. If later slots receive substantial weight, this suggests that depth upgrades may also move the needle. We caution that these interpretations are post-hoc and correlational; the attention weights tell us which slots the model uses, not necessarily which slots causally determine success.

### B. Fairness and Market-Size Audit

We performed a coarse market-size audit using the attention model's predictions. We grouped teams into large-market and small-market buckets based on media market size and compared average actual labels, predicted labels, and absolute errors.

In the full-season setting, the large-market average actual label is 1.257 while the small-market average actual label is 0.924. The large-market average predicted label is 1.305 while the small-market average predicted label is 0.988. The large-market average absolute error is 0.533, while the small-market average absolute error is 0.475, yielding an error gap of 0.058. In the mid-season setting, the gap widens: large-market error is 0.710 and small-market error is 0.556, for a gap of 0.153.

The predictions track the direction of actual group means: large-market teams have a higher average actual label and a higher average predicted label. We do not claim this establishes causal fairness; market size is a coarse proxy, and playoff success itself is not distributed evenly across markets. However, the full-season gap of 0.058 is small enough that the model does not appear to grossly favor large-market teams. The widening gap in the mid-season setting suggests that when data is sparser, structural advantages may be harder to disentangle from performance signals. Future work should conduct more comprehensive fairness audits using demographic and economic variables beyond market size.

### C. Limitations

Our project has several limitations that we address transparently. First, the class distribution is highly imbalanced, with only one champion and one finals loser per season. Even with 21 seasons, the rarest classes have only 21 examples each. Techniques such as SMOTE [14] or ADASYN [16] could be explored in future work, though temporal structure complicates their application. Second, the sample size is modest for a six-class forecasting problem. Third, the primary framing uses full regular-season statistics, so results should be interpreted as season-level forecasting rather than a strict mid-season prediction task. Fourth, contextual features such as roster continuity and conference strength are useful but still relatively simple compared with a production sports analytics system. Fifth, hyperparameter search was intentionally lightweight to preserve reproducibility and avoid overfitting the small dataset. Sixth, the fixed top-eight rotation representation may miss information from injuries, late-season rotation changes, and bench contributors outside the selected slots. Seventh, the player-level features are box-score and efficiency summaries, not play-by-play, lineup, matchup, or tracking data. Eighth, the market-size audit is a sanity check, not a complete fairness framework. It does not establish causal fairness or account for all structural differences among teams.

### D. Threats to Validity

External validity is limited because the NBA is a single league with unique rules, schedule structures, and playoff formats. Our findings may not transfer directly to other sports or leagues. Internal validity is strengthened by the season-aware validation protocol but weakened by the small sample size per class. Construct validity is supported by using standard basketball metrics that analysts and coaches recognize, but advanced tracking data might capture player value more accurately than box-score approximations. Statistical conclusion validity is adequate for the dominant classes but weak for Champion and Finals Loss due to sample size.

### E. Future Work

Several directions could extend our project. First, incorporating play-by-play or tracking data would provide richer player representations than box-score summaries alone. Second, modeling playoff matchups explicitly rather than treating each team independently could improve predictions because playoff paths depend on opponent seeding. Third, applying temporal models such as LSTMs or Transformers to within-season game sequences could capture momentum and form changes that our aggregate features miss. Fourth, exploring advanced imbalance techniques tailored to temporal data could improve rare-class performance. Fifth, a more comprehensive fairness framework using player demographics and team payroll data would address equity more rigorously.

## VI. Conclusion

We presented a reproducible deep learning study on forecasting NBA postseason depth from regular-season team and rotation statistics. Our attention-based model learns which rotation players matter most and combines roster information with team and contextual features. Under leave-one-season-out cross-validation on 21 seasons, we found that random forest delivers the strongest exact accuracy, the MLP baseline achieves the best macro F1, and the attention model remains competitive while adding interpretable player-importance weights. Top-2 accuracy exceeds 81% for every model, indicating that models often rank plausible postseason outcomes well even when they miss the exact class.

The mid-season extension confirmed that forecasting becomes harder when features are limited to partial-season information. Even there, the attention model tied for best exact accuracy and achieved the best macro F1 among tested models, with the smallest performance drop from full-season to mid-season.

The central empirical lesson is that regular-season team quality is predictive, but not enough to reliably identify the NBA champion as a distinct class. Championship outcomes are influenced by a small number of games, matchup paths, injuries, and player availability that regular-season statistics cannot fully capture. Our project contributes a reproducible benchmark, an interpretable attention architecture for roster modeling, and a transparent discussion of limitations that future work can build upon.

## ACKNOWLEDGMENT

We thank the course instructors of Deep Learning at the University of Colorado Denver for guidance throughout this project. We also acknowledge the nba_api project maintainers for providing open access to NBA statistics.

## REFERENCES

[1] R. Khanmohammadi, S. Saba-Sadiya, S. Esfandiarpour, T. Alhanai, and M. M. Ghassemi, "MambaNet: A hybrid neural network for predicting the NBA playoffs," SN Comput. Sci., vol. 5, no. 5, 2024.

[2] K. Zhao, C. Du, and G. Tan, "Enhancing basketball game outcome prediction through fused graph convolutional networks and random forest algorithm," Entropy, vol. 25, no. 5, 2023.

[3] W. Guan, N. Javed, and P. Lu, "NBA2Vec: Dense feature representations of NBA players," arXiv:2302.13386, 2023.

[4] Y. Ouyang et al., "Integration of machine learning XGBoost and SHAP models for NBA game outcome prediction," PLoS ONE, vol. 19, no. 7, e0307478, 2024.

[5] C. Rios, L. Han, A. Baimagambetov, and N. Polatidis, "Long sequence LSTM modeling for NBA game outcome prediction," arXiv:2512.08591, 2025.

[6] S. Z. Ibrahim et al., "Machine learning insights into basketball championship predictions: An analytical comparison," in Proc. ICITS, Springer, 2024.

[7] G. D. S. Teno, C. Wang, N. Carlsson, and P. Lambrix, "Predicting season outcomes for the NBA," in Machine Learning and Data Mining for Sports Analytics, ser. LNCS, vol. 1571, Springer, 2022, pp. 129-142.

[8] R. P. Bunker and F. Thabtah, "A machine learning framework for sport result prediction," Appl. Comput. Inform., vol. 15, no. 1, pp. 27-33, 2019.

[9] M. Yeung, "Multiple machine learning algorithms-based NBA team playoffs prediction," ITM Web Conf., vol. 70, 04024, 2025.

[10] J. Perricone, S. Shaw, and J. Swiechowicz, "Predicting results for professional basketball using NBA API data," Stanford Univ., Stanford, CA, USA, Tech. Rep. CS229, 2016.

[11] Y. Ni and S. Lee, "A comparative study of machine learning models for NCAA men's basketball tournament games outcome prediction," J. Prediction Markets, vol. 17, no. 2, pp. 3-34, 2023.

[12] Y. Wang, W. Liu, and X. Liu, "Explainable AI techniques with application to NBA gameplay prediction," Neurocomputing, vol. 483, pp. 59-71, 2022.

[13] M. Tsagris, C. Adam, and P. Pantatosakis, "On predicting an NBA game outcome from half-time statistics," Discov. Artif. Intell., vol. 4, 111, 2024.

[14] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," J. Artif. Intell. Res., vol. 16, pp. 321-357, 2002.

[15] D. Elreedy and A. F. Atiya, "A comprehensive analysis of synthetic minority oversampling technique (SMOTE) for handling class imbalance," Inf. Sci., vol. 505, pp. 32-64, 2019.

[16] H. He, Y. Bai, E. A. Garcia, and S. Li, "ADASYN: Adaptive synthetic sampling approach for imbalanced learning," in Proc. IEEE Int. Joint Conf. Neural Networks, 2008, pp. 1322-1328.

[17] A. Vaswani et al., "Attention is all you need," in Proc. 31st Conf. Neural Inf. Process. Syst., 2017, pp. 5998-6008.

[18] P. Mokha, "nba_api: An API client package to access NBA.com API," GitHub repository, 2018. [Online]. Available: https://github.com/swar/nba_api

[19] Z. Zhu, "High dimensional sports statistics and machine learning in NBA," Adv. Eng. Innov., vol. 11, pp. 78-94, 2024.

[20] M. Pietraszewski et al., "The role of artificial intelligence in sports analytics: A systematic review and meta-analysis of performance trends," Appl. Sci., vol. 15, 7254, 2025.

[21] N. Paine, "How our NBA predictions work," FiveThirtyEight, 2018. [Online]. Available: https://fivethirtyeight.com/methodology/how-our-nba-predictions-work/

[22] M. J. Dixon and S. G. Coles, "Modelling association football scores and inefficiencies in the football betting market," J. R. Stat. Soc. Ser. C Appl. Stat., vol. 46, pp. 265-280, 1997.

[23] J. G. Claudino et al., "Current approaches to the use of artificial intelligence for injury risk assessment and performance prediction in team sports: A systematic review," Sports Med. Open, vol. 5, 28, 2019.

[24] M. Naughton, P. M. Salmon, H. R. Compton, and S. McLean, "Challenges and opportunities of artificial intelligence implementation within sports science and sports medicine teams," Front. Sports Act. Living, vol. 6, 2024.

[25] T. Xu and S. Baghaei, "Reshaping the future of sports with artificial intelligence: Challenges and opportunities in performance enhancement, fan engagement, and strategic decision-making," Eng. Appl. Artif. Intell., vol. 142, 109912, 2025.
