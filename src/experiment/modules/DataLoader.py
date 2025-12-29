from datetime import date
from pathlib import Path

import polars as pl
import numpy as np

from experiment.settings import settings

class DataLoader:

    def __init__(self):
        self.bodmas_file_path = settings.data_path

    def load_bodmas_data(self):
        with np.load(self.bodmas_file_path / "bodmas.npz") as data:
            return data["X"], data["y"]

    def load_bodmas_metadata(self):

        metadata = pl.read_csv(self.bodmas_file_path / "bodmas_metadata.csv").select(
            [
                pl.col("sha"),
                pl.col("timestamp").str.split("+").list.get(0).str.to_datetime(format="%Y-%m-%d %H:%M:%S"),
                pl.col("timestamp").str.split("+").list.get(0).str.to_datetime(format="%Y-%m-%d %H:%M:%S").dt.date().alias("date"),
                pl.col("family").is_not_null().alias("is_malware")
            ]
        ).sort("timestamp").with_row_index("idx")
        
        return metadata

    def split_data(self, metadata, X, y, train_cutoff: date = date(2020, 3, 1)):
        train_pool_df = metadata.filter(pl.col("timestamp") < train_cutoff)
        test_pool_df = metadata.filter(pl.col("timestamp") >= train_cutoff)

        train_pool_indices = train_pool_df["idx"].to_numpy()
        y_train_pool = y[train_pool_indices]
        
        # Get indices for a 90/10 ratio of 0s and 1s
        stratified_train_indices = self.get_stratified_indices(y_train_pool, target_ratio=0.1)

        # We map the relative indices to the original global indices
        final_train_indices = train_pool_indices[stratified_train_indices]

        # We create the final training tensors.
        X_train = X[final_train_indices]
        y_train = y[final_train_indices]

        # 3. Prepare Test Sets (Month-by-Month)
        # The drift is evaluated "on a month-by-month basis".
        test_windows = test_pool_df.sort("timestamp").group_by_dynamic(
            "timestamp", 
            every="2w",
            start_by="datapoint"
        ).agg(pl.col("idx"))

        test_sets = []
        for row in test_windows.iter_rows(named=True):
            window_start = row["timestamp"]
            indices_in_window = np.array(row["idx"])
            
            if len(indices_in_window) == 0:
                continue
            
            # We apply stratification also in the test if you want to maintain consistency,
            # although to evaluate the "natural concept drift" it would be better to use the natural distribution.
            # Here I keep your original logic:
            y_window_raw = y[indices_in_window]
            strat_sub_idx = self.get_stratified_indices(y_window_raw, target_ratio=0.10)
            final_test_idx = indices_in_window[strat_sub_idx]
            
            if len(final_test_idx) > 0:
                test_sets.append({
                    "period_start": window_start,
                    "X_test": X[final_test_idx],
                    "y_test": y[final_test_idx]
                })

        return X_train, y_train, test_sets

    @staticmethod
    def get_stratified_indices(target_array, target_ratio):
        idx_0 = np.where(target_array == 0)[0]
        idx_1 = np.where(target_array == 1)[0]
        
        n_1 = len(idx_1)
        
        if n_1 == 0:
            return np.array([], dtype=int)

        # Equation: n_1 / (n_1 + n_0_needed) = ratio
        # Se ratio 0.1: n_1 / Total = 0.1 -> n_1 = 0.1 * Total -> Total = 10 * n_1
        # n_0 = Total - n_1 = 9 * n_1
        n_0_needed = int(n_1 * (1 - target_ratio) / target_ratio)

        rng = np.random.default_rng(seed=42)
        
        if len(idx_0) < n_0_needed:
            # If we don't have enough benign samples, we reduce the malware
            # n_1_new = n_0_actual * ratio / (1-ratio)
            n_1_adjusted = int(len(idx_0) * target_ratio / (1 - target_ratio))
            selected_0 = idx_0
            selected_1 = rng.choice(idx_1, n_1_adjusted, replace=False)
        else:
            selected_0 = rng.choice(idx_0, n_0_needed, replace=False)
            selected_1 = idx_1
            
        combined_indices = np.concatenate([selected_0, selected_1])
        np.random.shuffle(combined_indices)
        return combined_indices