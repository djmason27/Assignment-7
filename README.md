# NBA SciPy Data Analysis

## Purpose

This DSCI-D 211 project uses SciPy to find and analyze stories in NBA player statistics. It filters the supplied dataset to NBA regular-season records, identifies the player with the most recorded seasons, studies that player's three-point accuracy, models the trend with linear regression and integration, estimates two missing seasons with interpolation, and compares Field Goals Made with Field Goals Attempted using descriptive statistics and t-tests.

## Dataset

The repository uses `NBA_Player_Stats.tsv`, the file supplied with the course module. The assignment description also refers to `NBA_Data.csv`, so the program's loader supports either TSV or CSV based on the filename extension.

## Class design

The `NBAAnalysis` class keeps the complete workflow in one reusable object. Each method completes one stage of the analysis, and `run()` performs those stages in the correct order.

### Attributes

- `data_path`: Location of the CSV or TSV dataset.
- `output_directory`: Folder where CSV results and the graph are saved.
- `data`: Original data loaded from the file.
- `regular_season_data`: Data remaining after filtering for NBA regular-season rows.
- `longest_career_player`: Player with the most unique recorded seasons.
- `player_data`: Season records and calculated three-point accuracy for that player.
- `regression_result`: SciPy `linregress` output containing the slope, intercept, correlation, p-value, and standard errors.
- `fit_function`: Function representing the fitted regression line.

### Methods

- `load_data()`: Reads CSV or TSV data and verifies required columns.
- `filter_regular_season()`: Keeps NBA regular-season rows and converts shooting columns to numeric values.
- `_season_start()`: Converts a season label into its starting year.
- `find_longest_career_player()`: Counts unique seasons and calculates three-point accuracy as `3PM / 3PA`.
- `perform_regression()`: Uses `scipy.stats.linregress` to create the line of best fit.
- `calculate_accuracy_averages()`: Integrates the fitted line with `scipy.integrate.quad` and calculates actual averages.
- `interpolate_missing_seasons()`: Uses `scipy.interpolate.interp1d` to estimate 2002-2003 and 2015-2016.
- `_describe_column()`: Calculates mean, sample variance, bias-corrected skew, and Fisher kurtosis.
- `calculate_shooting_statistics()`: Applies the descriptive calculations to FGM and FGA.
- `perform_t_tests()`: Runs a paired t-test, a Welch independent t-test, and one-sample tests against zero.
- `create_regression_plot()`: Saves the observed accuracy, fit line, and interpolated estimates.
- `save_outputs()`: Saves detailed result tables as CSV files.
- `run()`: Executes the complete workflow.

## Results

The filtered dataset contains **4,658 NBA regular-season records**. **Vince Carter** has the most recorded seasons with **19**.

### Three-point analysis

| Measurement | Result |
|---|---:|
| Regression slope | -0.002517 accuracy per year |
| Regression R-squared | 0.233861 |
| Regression p-value | 0.035930 |
| Integrated fitted average accuracy | 37.0083% |
| Actual mean of seasonal accuracies | 36.9950% |
| Career weighted accuracy | 37.3492% |
| Average three-pointers made per recorded season | 114.05 |

The two interpolated values are **38.4649% for 2002-2003** and **33.7896% for 2015-2016**.

### FGM and FGA descriptive statistics

| Column | Mean | Sample variance | Skew | Excess kurtosis |
|---|---:|---:|---:|---:|
| FGM | 311.4841 | 25,944.0845 | 0.7670 | 0.1497 |
| FGA | 679.8620 | 118,066.7512 | 0.7331 | 0.1205 |

### T-tests

| Test | t-statistic | p-value |
|---|---:|---:|
| Paired FGM vs FGA | -133.007 | < 1e-300 |
| Welch independent FGM vs FGA | -66.2515 | < 1e-300 |
| One-sample FGM vs 0 | 131.983 | < 1e-300 |
| One-sample FGA vs 0 | 135.038 | < 1e-300 |

## Interpretation

Vince Carter's fitted accuracy decreases by about 0.252 percentage points per year. The R-squared value of 0.2339 means that a straight line explains only about 23.4% of the season-to-season variation, so the line is useful as a broad trend rather than a precise predictor. Integrating the fit line produces 37.0083%, only 0.0133 percentage points above the actual unweighted mean. The career weighted result is slightly higher because seasons with more attempts have more influence.

The prompt asks how fitted accuracy compares with the actual average number of three-pointers made. These measurements use different units, so they cannot be compared directly. The fitted value is a percentage, while 114.05 is a count per recorded season. The actual seasonal accuracy is the appropriate comparison for the integrated value.

FGA has a mean about 2.18 times the FGM mean and variance about 4.55 times as large. Both distributions are moderately right-skewed and have slightly positive excess kurtosis. The paired test has a much larger absolute t-statistic than the independent test because every player's attempts and makes come from the same season record. The one-sample tests only confirm that both column means are greater than zero; they do not directly compare makes with attempts.

## T-test assumption

The phrase "regular t-test on the FGM and FGA columns individually" does not state a comparison mean. This project interprets it as a one-sample t-test against zero and states that assumption openly. It also reports a Welch independent two-sample test so the independent and paired versions can be compared.

## How to run

1. Download or clone this repository.
2. Open a terminal in the project folder.
3. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the project:

   ```bash
   python scipy_nba_analysis.py
   ```

5. Review the terminal output and the generated files in the `results` folder.

## Output files

- `results/player_season_accuracy.csv`
- `results/interpolated_missing_seasons.csv`
- `results/fgm_fga_descriptive_statistics.csv`
- `results/t_test_results.csv`
- `results/three_point_regression.png`

## Limitations

- The supplied file contains recorded seasons rather than a perfect historical career record. Missing rows affect the season count and averages.
- Linear regression assumes a straight trend even though shooting performance changes nonlinearly.
- Interpolation estimates missing values from nearby seasons and does not reproduce the player's actual historical statistics.
- An accuracy percentage cannot be directly compared with a count of made three-pointers. The program therefore reports fitted accuracy, actual season-average accuracy, career weighted accuracy, and average made threes separately.
- The one-sample t-tests use zero as the null mean because the assignment does not specify another benchmark.

## AI-use disclosure

Generative AI helped organize, code, test, and document this project. Details are provided in `AI_CHAT_LOG.md`.
