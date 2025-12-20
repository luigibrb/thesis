from datetime import date
from pathlib import Path

import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:

    def __init__(self):
        self.bodmas_file_path = Path("/home/luigi/Workspace/thesis/data/bodmas")

    def load_bodmas_data(self):

        with np.load(self.bodmas_file_path / "bodmas.npz") as data:
            return data["X"], data["y"]

    def load_bodmas_metadata(self):

        metadata = pl.read_csv(self.bodmas_file_path / "bodmas_metadata.csv").with_row_index("idx").select(
            [
                pl.col("idx"),
                pl.col("sha"),
                pl.col("timestamp").str.split("+").list.get(0).str.to_datetime(format="%Y-%m-%d %H:%M:%S"),
                pl.col("timestamp").str.split("+").list.get(0).str.to_datetime(format="%Y-%m-%d %H:%M:%S").dt.date().alias("date"),
                # pl.col("family"),
                pl.col("family").is_not_null().alias("is_malware")
            ]
        )
        return metadata

    def split_data(self, metadata, X, y, train_cutoff: date = date(2020, 3, 1)):
        train_pool_df = metadata.filter(pl.col("timestamp") < train_cutoff)
        test_pool_df = metadata.filter(pl.col("timestamp") >= train_cutoff)

        train_pool_indices = train_pool_df["idx"].to_numpy()
        y_train_pool = y[train_pool_indices]
        
        # Get indices for a 90/10 ratio of 0s and 1s
        stratified_sub_indices = self.get_stratified_indices(y_train_pool, target_ratio=0.1)

        # These are the indices from the original X and y arrays that will be used for training/val/cal
        resampled_pool_indices = train_pool_indices[stratified_sub_indices]
        y_resampled_pool = y[resampled_pool_indices]

        # Splitting indices to get 70% train and 30% temporary for validation and calibration
        train_indices, temp_indices = train_test_split(
            resampled_pool_indices,
            test_size=0.30,
            stratify=y_resampled_pool,
            random_state=42
        )
        
        y_temp = y[temp_indices]

        # Splitting temporary indices to get 15% validation and 15% calibration (50% of 30%)
        val_indices, cal_indices = train_test_split(
            temp_indices,
            test_size=0.50,
            stratify=y_temp,
            random_state=42
        )

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        X_cal, y_cal = X[cal_indices], y[cal_indices]

        test_windows = test_pool_df.group_by_dynamic("timestamp", every="1w").agg(pl.col("idx"))

        test_sets = []
        for row in test_windows.iter_rows(named=True):
            window_start = row["timestamp"]
            indices_in_window = np.array(row["idx"])
            
            if len(indices_in_window) == 0:
                continue
                
            strat_sub_idx = self.get_stratified_indices(y[indices_in_window], target_ratio=0.10)
            final_test_idx = indices_in_window[strat_sub_idx]
            
            test_sets.append({
                "week_start": window_start,
                "X_test": X[final_test_idx],
                "y_test": y[final_test_idx]
            })

        return X_train, y_train, X_val, y_val, X_cal, y_cal, test_sets


    @staticmethod
    def get_stratified_indices(target_array, target_ratio):
        idx_0 = np.where(target_array == 0)[0]
        idx_1 = np.where(target_array == 1)[0]
        
        n_1 = len(idx_1)
        # Calculate how many 0s we need to make n_1 represent 10% of the total
        # Equation: n_1 / (n_1 + n_0_needed) = 0.10  => n_0_needed = 9 * n_1
        n_0_needed = int(n_1 * 9)

        rng = np.random.default_rng(seed=42)
        
        if len(idx_0) < n_0_needed:
            # If we don't have enough 0s, we downsample the 1s instead
            n_1_adjusted = int(len(idx_0) / 9)
            selected_0 = idx_0
            selected_1 = rng.choice(idx_1, n_1_adjusted, replace=False)
        else:
            selected_0 = rng.choice(idx_0, n_0_needed, replace=False)
            selected_1 = idx_1
            
        combined_indices = np.concatenate([selected_0, selected_1])
        np.random.shuffle(combined_indices)
        return combined_indices
