"""SciPy analysis of NBA regular-season shooting data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate, interpolate, stats


class NBAAnalysis:
    """Load, validate, analyze, and report NBA regular-season data."""

    REQUIRED_COLUMNS = {
        "League",
        "Season",
        "Stage",
        "Player",
        "FGM",
        "FGA",
        "3PM",
        "3PA",
    }

    def __init__(self, data_path, output_directory="results"):
        """Store file locations and initialize analysis attributes."""
        self.data_path = Path(data_path)
        self.output_directory = Path(output_directory)
        self.data = None
        self.regular_season_data = None
        self.longest_career_player = None
        self.player_data = None
        self.regression_result = None
        self.fit_function = None

    def load_data(self):
        """Read a CSV or TSV file and verify its required columns."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        separator = "\t" if self.data_path.suffix.lower() == ".tsv" else ","
        self.data = pd.read_csv(self.data_path, sep=separator)

        missing = self.REQUIRED_COLUMNS.difference(self.data.columns)
        if missing:
            raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
        if self.data.empty:
            raise ValueError("The dataset contains no rows.")
        return self.data

    def filter_regular_season(self):
        """Keep NBA regular-season rows and convert numeric columns safely."""
        if self.data is None:
            raise RuntimeError("Call load_data() before filtering.")

        league = self.data["League"].astype(str).str.strip().str.upper()
        stage = (
            self.data["Stage"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )
        filtered = self.data[(league == "NBA") & (stage == "regular_season")].copy()
        if filtered.empty:
            raise ValueError("No NBA regular-season rows were found.")

        for column in ["FGM", "FGA", "3PM", "3PA"]:
            filtered[column] = pd.to_numeric(filtered[column], errors="coerce")
        filtered = filtered.dropna(subset=["Player", "Season", "FGM", "FGA", "3PM", "3PA"])
        self.regular_season_data = filtered
        return filtered

    @staticmethod
    def _season_start(season):
        """Convert a label such as '1999 - 2000' to the integer 1999."""
        try:
            return int(str(season).split("-")[0].strip())
        except (ValueError, IndexError) as error:
            raise ValueError(f"Unrecognized season label: {season}") from error

    def find_longest_career_player(self):
        """Find the player with the greatest number of unique regular seasons."""
        if self.regular_season_data is None:
            raise RuntimeError("Call filter_regular_season() first.")

        season_counts = (
            self.regular_season_data.groupby("Player")["Season"]
            .nunique()
            .sort_values(ascending=False)
        )
        if season_counts.empty:
            raise ValueError("No players were available after filtering.")

        self.longest_career_player = season_counts.index[0]
        player = self.regular_season_data[
            self.regular_season_data["Player"] == self.longest_career_player
        ].copy()
        player["Season_Start"] = player["Season"].map(self._season_start)
        player["Three_Point_Accuracy"] = np.where(
            player["3PA"] > 0, player["3PM"] / player["3PA"], np.nan
        )
        player = player.dropna(subset=["Three_Point_Accuracy"]).sort_values("Season_Start")
        self.player_data = player
        return self.longest_career_player, int(season_counts.iloc[0]), player

    def perform_regression(self):
        """Fit a SciPy linear regression to three-point accuracy by season."""
        if self.player_data is None:
            raise RuntimeError("Call find_longest_career_player() first.")

        years = self.player_data["Season_Start"].to_numpy(dtype=float)
        accuracy = self.player_data["Three_Point_Accuracy"].to_numpy(dtype=float)
        if len(years) < 2:
            raise ValueError("At least two seasons are needed for regression.")

        self.regression_result = stats.linregress(years, accuracy)
        slope = self.regression_result.slope
        intercept = self.regression_result.intercept
        self.fit_function = lambda year: slope * year + intercept
        return self.regression_result

    def calculate_accuracy_averages(self):
        """Integrate the fitted line and calculate actual shooting averages."""
        if self.fit_function is None:
            raise RuntimeError("Call perform_regression() first.")

        first_year = float(self.player_data["Season_Start"].min())
        last_year = float(self.player_data["Season_Start"].max())
        year_span = last_year - first_year
        if year_span <= 0:
            raise ValueError("The played-season range must be greater than zero.")

        area, integration_error = integrate.quad(self.fit_function, first_year, last_year)
        fitted_average_accuracy = area / year_span
        actual_mean_accuracy = self.player_data["Three_Point_Accuracy"].mean()
        career_weighted_accuracy = self.player_data["3PM"].sum() / self.player_data["3PA"].sum()
        average_three_pointers_made = self.player_data["3PM"].mean()

        return {
            "first_year": int(first_year),
            "last_year": int(last_year),
            "year_span": int(year_span),
            "integrated_area": area,
            "integration_error": integration_error,
            "fitted_average_accuracy": fitted_average_accuracy,
            "actual_mean_accuracy": actual_mean_accuracy,
            "career_weighted_accuracy": career_weighted_accuracy,
            "average_three_pointers_made": average_three_pointers_made,
        }

    def interpolate_missing_seasons(self, missing_years=(2002, 2015)):
        """Use linear interpolation to estimate missing accuracy values."""
        if self.player_data is None:
            raise RuntimeError("Player data is not available.")

        years = self.player_data["Season_Start"].to_numpy(dtype=float)
        accuracy = self.player_data["Three_Point_Accuracy"].to_numpy(dtype=float)
        interpolation_function = interpolate.interp1d(
            years, accuracy, kind="linear", bounds_error=True
        )

        estimates = []
        for year in missing_years:
            if year < years.min() or year > years.max():
                raise ValueError(f"Cannot interpolate {year}; it is outside the data range.")
            estimates.append(
                {
                    "Season": f"{year} - {year + 1}",
                    "Season_Start": year,
                    "Estimated_Three_Point_Accuracy": float(interpolation_function(year)),
                }
            )
        return pd.DataFrame(estimates)

    @staticmethod
    def _describe_column(values):
        """Return SciPy mean, variance, skew, and kurtosis for a numeric array."""
        clean = np.asarray(values, dtype=float)
        clean = clean[np.isfinite(clean)]
        if clean.size < 2:
            raise ValueError("At least two finite values are required for statistics.")
        return {
            "Mean": float(np.mean(clean)),
            "Variance": float(np.var(clean, ddof=1)),
            "Skew": float(stats.skew(clean, bias=False)),
            "Kurtosis": float(stats.kurtosis(clean, fisher=True, bias=False)),
        }

    def calculate_shooting_statistics(self):
        """Calculate descriptive statistics for FGM and FGA."""
        if self.regular_season_data is None:
            raise RuntimeError("Regular-season data is not available.")
        return pd.DataFrame(
            {
                "FGM": self._describe_column(self.regular_season_data["FGM"]),
                "FGA": self._describe_column(self.regular_season_data["FGA"]),
            }
        ).T

    def perform_t_tests(self):
        """Run paired, independent, and one-sample t-tests on FGM and FGA."""
        if self.regular_season_data is None:
            raise RuntimeError("Regular-season data is not available.")

        paired = self.regular_season_data[["FGM", "FGA"]].dropna()
        fgm = paired["FGM"].to_numpy(dtype=float)
        fga = paired["FGA"].to_numpy(dtype=float)

        paired_result = stats.ttest_rel(fgm, fga)
        independent_result = stats.ttest_ind(fgm, fga, equal_var=False)
        fgm_one_sample = stats.ttest_1samp(fgm, popmean=0)
        fga_one_sample = stats.ttest_1samp(fga, popmean=0)

        return pd.DataFrame(
            [
                ["Paired FGM vs FGA", paired_result.statistic, paired_result.pvalue, "Related samples"],
                ["Independent FGM vs FGA", independent_result.statistic, independent_result.pvalue, "Welch two-sample"],
                ["FGM vs 0", fgm_one_sample.statistic, fgm_one_sample.pvalue, "One-sample"],
                ["FGA vs 0", fga_one_sample.statistic, fga_one_sample.pvalue, "One-sample"],
            ],
            columns=["Test", "Statistic", "P_Value", "Test_Type"],
        )

    def create_regression_plot(self, interpolation_results):
        """Save the observed values, fit line, and interpolated values as a PNG."""
        if self.fit_function is None:
            raise RuntimeError("Regression results are not available.")
        self.output_directory.mkdir(parents=True, exist_ok=True)

        years = self.player_data["Season_Start"].to_numpy(dtype=float)
        accuracy = self.player_data["Three_Point_Accuracy"].to_numpy(dtype=float)
        line_years = np.linspace(years.min(), years.max(), 300)

        plt.figure(figsize=(10, 6))
        plt.scatter(years, accuracy * 100, color="#1f77b4", label="Observed accuracy", zorder=3)
        plt.plot(
            line_years,
            self.fit_function(line_years) * 100,
            color="#d62728",
            linewidth=2,
            label="Linear fit",
        )
        plt.scatter(
            interpolation_results["Season_Start"],
            interpolation_results["Estimated_Three_Point_Accuracy"] * 100,
            color="#2ca02c",
            marker="X",
            s=110,
            label="Interpolated missing seasons",
            zorder=4,
        )
        plt.title(f"{self.longest_career_player} Three-Point Accuracy by Season")
        plt.xlabel("Season start year")
        plt.ylabel("Three-point accuracy (%)")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plot_path = self.output_directory / "three_point_regression.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        return plot_path

    def save_outputs(self, interpolation_results, descriptive_statistics, t_tests):
        """Save all tabular analysis results to CSV files."""
        self.output_directory.mkdir(parents=True, exist_ok=True)

        season_accuracy = self.player_data[
            ["Season", "Season_Start", "3PM", "3PA", "Three_Point_Accuracy"]
        ].copy()
        season_accuracy["Three_Point_Accuracy_Percent"] = (
            season_accuracy["Three_Point_Accuracy"] * 100
        )
        season_accuracy.to_csv(self.output_directory / "player_season_accuracy.csv", index=False)
        interpolation_results.to_csv(
            self.output_directory / "interpolated_missing_seasons.csv", index=False
        )
        descriptive_statistics.to_csv(
            self.output_directory / "fgm_fga_descriptive_statistics.csv"
        )
        t_tests.to_csv(self.output_directory / "t_test_results.csv", index=False)

    def run(self):
        """Execute the complete assignment workflow and return a result summary."""
        self.load_data()
        self.filter_regular_season()
        player, season_count, _ = self.find_longest_career_player()
        regression = self.perform_regression()
        averages = self.calculate_accuracy_averages()
        interpolation_results = self.interpolate_missing_seasons()
        descriptive_statistics = self.calculate_shooting_statistics()
        t_tests = self.perform_t_tests()
        self.save_outputs(interpolation_results, descriptive_statistics, t_tests)
        plot_path = self.create_regression_plot(interpolation_results)

        return {
            "player": player,
            "season_count": season_count,
            "row_count": len(self.regular_season_data),
            "regression": regression,
            "averages": averages,
            "interpolation": interpolation_results,
            "descriptive_statistics": descriptive_statistics,
            "t_tests": t_tests,
            "plot_path": plot_path,
        }


def print_results(results):
    """Print a readable summary of the completed analysis."""
    regression = results["regression"]
    averages = results["averages"]

    print(f"Regular-season rows: {results['row_count']}")
    print(f"Longest recorded career: {results['player']} ({results['season_count']} seasons)")
    print(
        "Regression: accuracy = "
        f"{regression.slope:.8f} * year + {regression.intercept:.6f}"
    )
    print(f"R-squared: {regression.rvalue ** 2:.6f}")
    print(f"Regression p-value: {regression.pvalue:.6g}")
    print(f"Fitted average accuracy: {averages['fitted_average_accuracy']:.4%}")
    print(f"Actual season-average accuracy: {averages['actual_mean_accuracy']:.4%}")
    print(f"Career weighted accuracy: {averages['career_weighted_accuracy']:.4%}")
    print(f"Average three-pointers made per recorded season: {averages['average_three_pointers_made']:.2f}")
    print("\nInterpolated missing seasons:")
    print(results["interpolation"].to_string(index=False))
    print("\nFGM and FGA descriptive statistics:")
    print(results["descriptive_statistics"].to_string(float_format="{:.6f}".format))
    print("\nT-tests:")
    print(results["t_tests"].to_string(index=False, float_format="{:.6g}".format))
    print(f"\nResults saved in: {results['plot_path'].parent}")


def main():
    """Run the analysis while providing useful errors for common failures."""
    try:
        project = NBAAnalysis("NBA_Player_Stats.tsv")
        results = project.run()
        print_results(results)
    except (FileNotFoundError, ValueError, RuntimeError, pd.errors.ParserError) as error:
        print(f"Analysis could not be completed: {error}")


if __name__ == "__main__":
    main()
