import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
from experiment.settings import settings

class Plotter:

    @classmethod
    def plot_cce_results(self, csv_data):

        # Data loading
        df = pl.read_csv(io.StringIO(csv_data))
        df = df.with_columns(pl.col('period').str.to_datetime())

        # Chart Configuration
        plt.style.use('default') 
        fig, ax1 = plt.subplots(figsize=(14, 8))

        # Define colors and styles
        color_baseline = 'gray'
        color_kept = 'tab:blue'
        color_rejected = 'tab:red'
        color_bar = 'lightgray'

        # 1. Rejection Rate Bars (Background)
        # Approximate width of 10 days for visibility
        ax1.bar(df['period'].to_list(), df['rejection_rate'], color=color_bar, width=10, label='Rejection Rate', alpha=0.6)

        # 2. Scatter plot for drift rates (Malware vs Goodware)
        ax1.scatter(df['period'].to_list(), df['drift_rate_malware'], marker='x', color='tab:red', label='Rate of drifting malware', zorder=5)
        ax1.scatter(df['period'].to_list(), df['drift_rate_goodware'], marker='o', facecolors='none', edgecolors='black', label='Rate of drifting goodware', zorder=5)

        # 3. Line plot for F1 Scores
        ax1.plot(df['period'].to_list(), df['f1_baseline'], linestyle='--', color=color_baseline, label='F1 Baseline (No Rejection)', linewidth=2)
        ax1.plot(df['period'].to_list(), df['f1_kept'], marker='^', markersize=5, color=color_kept, label='F1 Kept (Accepted)', linewidth=2)
        ax1.plot(df['period'].to_list(), df['f1_rejected'], marker='v', markersize=5, color=color_rejected, label='F1 Rejected (Quarantined)', linewidth=2)

        # Formatting
        ax1.set_xlabel('Testing Period (Month)')
        ax1.set_ylabel('Score / Rate')
        ax1.set_ylim(0, 1.05)
        ax1.set_title('TRANSCENDENT Performance Analysis: CCE with Credibility', fontsize=16)

        # Time axis formatting
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)

        # Legend
        handles, labels = ax1.get_legend_handles_labels()
        # Legend positioning bottom left
        ax1.legend(handles, labels, loc='lower left', ncol=2, framealpha=0.9)

        plt.tight_layout()
        plt.grid(True, linestyle=':', alpha=0.6)

        plt.show()

if __name__ == "__main__":
    plotter = Plotter()
    # Example file path using settings
    file_path = settings.results_path / "runs" / f"cce_experiment_results_{settings.plot_experiment_name}.csv"
    
    # Check if file exists before reading (optional but good for testing)
    if file_path.exists():
        with open(file_path, "r") as file:
            csv_data = file.read()
            plotter.plot_cce_results(csv_data)
    else:
        print(f"File not found: {file_path}")