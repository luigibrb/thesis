import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from experiment.settings import settings

class Plotter:

    @classmethod
    def plot_cce_results(self, df):
        # Data loading
        if df.schema['period'] == pl.String:
            df = df.with_columns(pl.col('period').str.to_datetime())
        
        dates = df['period'].to_list()

        # Chart Configuration: 3 rows for F1, Precision, Recall
        plt.style.use('default') 
        fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)

        # Metrics to plot
        metrics = [
            ("f1", "F1 Score"),
            ("prec", "Precision"),
            ("rec", "Recall")
        ]

        # Define colors and styles
        color_baseline = 'gray'
        color_kept = 'tab:blue'
        color_rejected = 'tab:red'
        color_bar = 'lightgray'

        for i, (metric_prefix, metric_label) in enumerate(metrics):
            ax = axes[i]
            
            # --- 1. Context: Rejection Rate Bars (Background) ---
            # Approximate width of 5 days for visibility
            ax.bar(dates, df['rejection_rate'], color=color_bar, width=5, label='Rejection Rate', alpha=0.6)

            # --- 2. Context: Drift Rates (Scatter) ---
            ax.scatter(dates, df['drift_rate_malware'], marker='x', color='tab:red', 
                       label='Rate of drifting malware', zorder=5)
            ax.scatter(dates, df['drift_rate_goodware'], marker='o', facecolors='none', edgecolors='black', 
                       label='Rate of drifting goodware', zorder=5)

            # --- 3. Performance Lines (Metric Specific) ---
            # Baseline
            ax.plot(dates, df[f'{metric_prefix}_baseline'], linestyle='--', color=color_baseline, 
                    label=f'{metric_label} Baseline (No Rejection)', linewidth=2)
            
            # Kept (Accepted)
            ax.plot(dates, df[f'{metric_prefix}_kept'], marker='^', markersize=5, color=color_kept, 
                    label=f'{metric_label} Kept (Accepted)', linewidth=2)
            
            # Rejected (Quarantined)
            ax.plot(dates, df[f'{metric_prefix}_rejected'], marker='v', markersize=5, color=color_rejected, 
                    label=f'{metric_label} Rejected (Quarantined)', linewidth=2)

            # Formatting per subplot
            ax.set_ylabel(f'{metric_label} / Rate')
            ax.set_ylim(0, 1.05)
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # Title only on the top subplot
            if i == 0:
                ax.set_title('TRANSCENDENT Performance Analysis: CCE with Credibility', fontsize=16)

            # --- Legend on ALL subplots ---
            # Placing the legend outside to the right to prevent obscuring data
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, framealpha=0.9)

        # X-axis Formatting (Shared, applied to the last subplot)
        axes[-1].set_xlabel('Testing Period (Week)')
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Rotate dates on the bottom subplot
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_rejection_rate(cls, df, W=4, retrain_threshold=None):
        # Data loading
        if df.schema['period'] == pl.String:
            df = df.with_columns(pl.col('period').str.to_datetime())
            
        # Determine retrain_threshold if not provided
        if retrain_threshold is None:
            if "retrain_threshold" in df.columns:
                # Assume constant threshold for the run, take the first non-null value
                retrain_threshold = df['retrain_threshold'][0]
            else:
                print("Warning: 'retrain_threshold' not found in DataFrame and not provided. Defaulting to 0.10")
                retrain_threshold = 0.10

        # Calculate moving average of rejection rate
        df = df.with_columns(
            pl.col('rejection_rate').rolling_mean(window_size=W).alias('rejection_rate_ma')
        )
        
        dates = df['period'].to_list()
        
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(14, 6))
        
        color_bar = 'lightgray'
        
        # Bars
        ax.bar(dates, df['rejection_rate'], color=color_bar, width=5, label='Rejection Rate', alpha=0.6)
        
        # Moving Average
        ax.plot(dates, df['rejection_rate_ma'], color='tab:purple', linestyle='-', linewidth=2, label=f'Rejection Rate MA (W={W})')
        
        # Threshold Line
        ax.axhline(y=retrain_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({round(retrain_threshold, 4)})')
        
        ax.set_xlabel('Testing Period (Week)')
        ax.set_ylabel('Rejection Rate')
        ax.set_title(f'Rejection Rate Analysis (MA Window = {W})')
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        ax.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    plotter = Plotter()
    # Example file path using settings
    file_path = settings.results_path / "runs" / f"cce_experiment_results_{settings.plot_experiment_name}.csv"
    
    # Check if file exists before reading
    if file_path.exists():
        df = pl.read_csv(file_path)
        plotter.plot_cce_results(df)
        plotter.plot_rejection_rate(df)
    else:
        print(f"File not found: {file_path}")