import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from pathlib import Path
from experiment.settings import settings

# Make sure to import your classes
# from dataloader import DataLoader
# from trainer import CrossConformalTrainer

class ExperimentRunner:
    def __init__(self, output_dir=None):
        self.results = []
        self.output_dir = Path(output_dir) if output_dir else settings.results_path
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _compute_metrics(self, y_true, y_pred, mask, set_name):
        """
        Computes standard metrics on a subset defined by the mask.
        If the mask is empty, returns NaN or 0.
        """
        if np.sum(mask) == 0:
            return {
                f"f1_{set_name}": np.nan,
                f"prec_{set_name}": np.nan,
                f"rec_{set_name}": np.nan,
                f"count_{set_name}": 0
            }
        
        y_t = y_true[mask]
        y_p = y_pred[mask]
        
        return {
            f"f1_{set_name}": f1_score(y_t, y_p, average='macro', zero_division=0),
            f"prec_{set_name}": precision_score(y_t, y_p, average='macro', zero_division=0),
            f"rec_{set_name}": recall_score(y_t, y_p, average='macro', zero_division=0),
            f"count_{set_name}": len(y_t)
        }

    def run(self, trainer, test_sets, run_id=None):
        print(f"Starting evaluation on {len(test_sets)} periods...")
        
        for i, batch in enumerate(test_sets):
            period_date = batch['period_start']
            X_test = batch['X_test']
            y_test = batch['y_test']
            
            # 1. Get decisions from CCE (Accept/Reject)
            is_accepted = trainer.predict(X_test) # Boolean array
            is_rejected = ~is_accepted
            
            # 2. Get predictions from the underlying classifier (Baseline)
            # Since CCE is an ensemble, we use the Majority Vote of the internal models' predictions
            # or simply the aggregated prediction.
            y_pred_baseline = trainer.predict_labels(X_test) 
            
            # --- METRICS COLLECTION ---
            row = {
                "run_id": run_id,
                "period": period_date,
                "n_samples": len(y_test),
                "rejection_rate": np.mean(is_rejected)
            }
            
            # A. Baseline (No Rejection)
            row.update(self._compute_metrics(y_test, y_pred_baseline, np.ones(len(y_test), dtype=bool), "baseline"))
            
            # B. Kept Elements (Accepted)
            row.update(self._compute_metrics(y_test, y_pred_baseline, is_accepted, "kept"))
            
            # C. Rejected Elements (Quarantined)
            row.update(self._compute_metrics(y_test, y_pred_baseline, is_rejected, "rejected"))
            
            # D. Drift Rates (Specific analysis for Malware vs Goodware) 
            # How many real Malware were rejected?
            malware_mask = (y_test == 1)
            if np.sum(malware_mask) > 0:
                row["drift_rate_malware"] = np.sum(is_rejected & malware_mask) / np.sum(malware_mask)
            else:
                row["drift_rate_malware"] = np.nan
                
            # How many real Goodware were rejected?
            goodware_mask = (y_test == 0)
            if np.sum(goodware_mask) > 0:
                row["drift_rate_goodware"] = np.sum(is_rejected & goodware_mask) / np.sum(goodware_mask)
            else:
                row["drift_rate_goodware"] = np.nan

            self.results.append(row)
            print(f"[{period_date}] Rej Rate: {row['rejection_rate']:.2%} | F1 Kept: {row['f1_kept']:.3f}")

        # Final save
        df_results = pl.DataFrame(self.results)
        
        if run_id:
            save_path = self.output_dir / f"cce_experiment_results_{str(run_id)[0:8]}.csv"
        else:
            save_path = self.output_dir / "cce_experiment_results.csv"
            
        df_results.write_csv(save_path)
        print(f"Results saved to {save_path}")
        return df_results