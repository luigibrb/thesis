import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from experiment.settings import settings

class Plotter:

    @classmethod
    def plot_cce_results(cls, df, ma_window=5):
        # Data loading
        if df.schema['period'] == pl.String:
            df = df.with_columns(pl.col('period').str.to_datetime())
        
        dates = df['period'].to_list()
        
        # --- 1. Dynamic Width Calculation ---
        # Determine the minimum time delta between points to set an appropriate bar width.
        # Matplotlib dates are floats representing days.
        if len(dates) > 1:
            # Calculate differences in days
            diffs = [(dates[i+1] - dates[i]).total_seconds() / (3600 * 24) for i in range(len(dates)-1)]
            min_diff = min(diffs)
            # Set width to 100% of the interval (or slightly less for spacing)
            bar_width = min_diff * 1.0 
        else:
            bar_width = 1.0 # Default fallback

        # Chart Configuration: 3 rows for F1, Precision, Recall
        plt.style.use('default') 
        fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)

        # Metrics to plot
        metrics = [
            ("f1", "F1 Score"),
            ("prec", "Precision"),
            ("rec", "Recall")
        ]
        
        # Calculate Moving Averages for Rejected, Baseline, and Kept Metrics
        df = df.with_columns([
            pl.col(f'{m[0]}_rejected').rolling_mean(window_size=ma_window).alias(f'{m[0]}_rejected_ma')
            for m in metrics
        ] + [
            pl.col(f'{m[0]}_baseline').rolling_mean(window_size=ma_window).alias(f'{m[0]}_baseline_ma')
            for m in metrics
        ] + [
            pl.col(f'{m[0]}_kept').rolling_mean(window_size=ma_window).alias(f'{m[0]}_kept_ma')
            for m in metrics
        ])

        # Define colors and styles
        color_baseline = 'gray'
        color_kept = 'tab:blue'
        color_rejected = 'tab:red'
        color_bar = 'lightgray'

        for i, (metric_prefix, metric_label) in enumerate(metrics):
            ax = axes[i]
            
            # --- 2. Context: Rejection Rate Bars (Background) ---
            # Adapted width to the granularity
            # Made less transparent (alpha 0.5 -> 0.8) as requested
            ax.bar(dates, df['rejection_rate'], color=color_bar, width=bar_width, 
                   label='Rejection Rate', alpha=0.8, align='center')

            # --- 3. Context: Drift Rates (Scatter) ---
            # Reduced size (s) and added transparency (alpha) to handle high density
            ax.scatter(dates, df['drift_rate_malware'], marker='x', color='tab:red', s=15, alpha=0.7,
                       label='Rate of drifting malware', zorder=5)
            ax.scatter(dates, df['drift_rate_goodware'], marker='o', facecolors='none', edgecolors='black', s=15, alpha=0.5,
                       label='Rate of drifting goodware', zorder=5)

            # --- 4. Performance Lines (Metric Specific) ---
            # Removed markers from lines to prevent visual clutter in daily views.
            # Increased zorder to ensure lines appear above bars.
            
            # Baseline - Raw (Semi-transparent)
            ax.plot(dates, df[f'{metric_prefix}_baseline'], linestyle='--', color=color_baseline, 
                    label=f'{metric_label} Baseline (Raw)', linewidth=1.0, alpha=0.3, zorder=6)
            
            # Baseline - MA (Solid/Dashed)
            ax.plot(dates, df[f'{metric_prefix}_baseline_ma'], linestyle='--', color=color_baseline, 
                    label=f'{metric_label} Baseline (MA)', linewidth=2.0, alpha=1.0, zorder=8)
            
            # Kept (Accepted) - Raw (Semi-transparent)
            ax.plot(dates, df[f'{metric_prefix}_kept'], linestyle='-', color=color_kept, 
                    label=f'{metric_label} Kept (Raw)', linewidth=1.0, alpha=0.3, zorder=7)
            
            # Kept (Accepted) - MA (Solid)
            ax.plot(dates, df[f'{metric_prefix}_kept_ma'], linestyle='-', color=color_kept, 
                    label=f'{metric_label} Kept (MA)', linewidth=2.0, alpha=1.0, zorder=8)
            
            # Rejected (Quarantined) - Raw (Semi-transparent)
            ax.plot(dates, df[f'{metric_prefix}_rejected'], linestyle='-', color=color_rejected, 
                    label=f'{metric_label} Rejected (Raw)', linewidth=1.0, alpha=0.2, zorder=6)
            
            # Rejected (Quarantined) - Moving Average (Solid)
            ax.plot(dates, df[f'{metric_prefix}_rejected_ma'], linestyle='-', color=color_rejected, 
                    label=f'{metric_label} Rejected (MA)', linewidth=2, alpha=1.0, zorder=8)

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
        axes[-1].set_xlabel('Testing Period') # Removed specific "Week" or "Day" label to be generic
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # Adjust locator based on data length to prevent label overlapping
        if len(dates) > 50:
             axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
        else:
             axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Rotate dates on the bottom subplot
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_rejection_rate(cls, df, W=4, retrain_tolerance_constant=0.02):
        # Data loading
        if df.schema['period'] == pl.String:
            df = df.with_columns(pl.col('period').str.to_datetime())

        if "retrain_threshold" in df.columns:
            retrain_threshold = df['retrain_threshold'][0] + retrain_tolerance_constant
        else:
            retrain_threshold = 0.10 + retrain_tolerance_constant

        # Calculate moving average
        df = df.with_columns(
            pl.col('rejection_rate').rolling_mean(window_size=W).alias('rejection_rate_ma')
        )
        
        dates = df['period'].to_list()
        
        # --- Dynamic Width Calculation (Same logic as above) ---
        if len(dates) > 1:
            diffs = [(dates[i+1] - dates[i]).total_seconds() / (3600 * 24) for i in range(len(dates)-1)]
            min_diff = min(diffs)
            bar_width = min_diff * 1.0 
        else:
            bar_width = 1.0

        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(14, 6))
        
        color_bar = 'lightgray'
        
        # Bars with dynamic width
        ax.bar(dates, df['rejection_rate'], color=color_bar, width=bar_width, label='Rejection Rate', alpha=0.6, align='center')
        
        # Moving Average
        ax.plot(dates, df['rejection_rate_ma'], color='tab:purple', linestyle='-', linewidth=2, label=f'Rejection Rate MA (W={W})')
        
        # Threshold Line
        ax.axhline(y=retrain_threshold, color='red', linestyle='--', linewidth=2, label=f'Retraining Threshold ({round(retrain_threshold*100, 2)}%)')

        # Find first breach of threshold
        breach = df.filter(pl.col('rejection_rate_ma') >= retrain_threshold).head(1)
        if not breach.is_empty():
            breach_date = breach['period'][0]
            breach_val = breach['rejection_rate_ma'][0]
            ax.scatter([breach_date], [breach_val], color='red', marker='x', s=100, zorder=10, label='First Threshold Breach')
        
        ax.set_xlabel('Testing Period')
        ax.set_ylabel('Rejection Rate')
        ax.set_title(f'Rejection Rate Analysis (MA Window = {W})')
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        if len(dates) > 50:
             ax.xaxis.set_major_locator(mdates.MonthLocator())
        else:
             ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        ax.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    plotter = Plotter()
    file_path = settings.results_path / "runs" / f"cce_experiment_results_{settings.plot_experiment_name}.csv"
    
    if file_path.exists():
        df = pl.read_csv(file_path)
        plotter.plot_cce_results(df)
        plotter.plot_rejection_rate(df)
    else:
        print(f"File not found: {file_path}")