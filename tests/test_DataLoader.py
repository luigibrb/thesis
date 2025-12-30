import pytest
import polars as pl
import numpy as np
from datetime import date, datetime

from experiment.modules.DataLoader import DataLoader

@pytest.fixture
def mock_data():
    """Provides mock data for tests."""
    # Create a mock metadata DataFrame
    mock_metadata = pl.DataFrame({
        "idx": range(200),
        "sha": [f"sha_{i}" for i in range(200)],
        "timestamp": [
            datetime(2020, 1, 1),
            datetime(2020, 2, 1),
            datetime(2020, 3, 1),
            datetime(2020, 3, 8),
        ] * 50,
        "family": [None] * 180 + ["fam1"] * 20
    })

    # Create mock features and labels
    mock_X = np.random.rand(200, 10)
    mock_y = np.zeros(200)
    malware_indices = mock_metadata.with_row_index().filter(pl.col("family").is_not_null())["index"]
    mock_y[malware_indices.to_numpy()] = 1
    
    return mock_metadata, mock_X, mock_y

def test_split_data(mock_data):
    """Test the data splitting and stratification logic."""
    mock_metadata, mock_X, mock_y = mock_data
    loader = DataLoader()
    train_cutoff = date(2020, 3, 1)

    X_train, y_train, test_sets = loader.split_data(
        mock_metadata, mock_X, mock_y, train_cutoff=train_cutoff
    )

    # Check shapes
    assert X_train.shape[0] == y_train.shape[0]

    # Check split sizes
    # Train pool has 100 samples (d1, d2). 10 malware.
    # Stratification 0.1 ratio -> 10 malware needs 90 benign. We have 90.
    # Takes all 100.
    assert len(y_train) == 100
    assert y_train.mean() == pytest.approx(0.1, abs=0.01)

    # Check test sets
    assert len(test_sets) == 2
    
    # Test week 1
    assert test_sets[0]["period_start"] == datetime(2020, 3, 1)
    assert len(test_sets[0]["y_test"]) == 50
    assert test_sets[0]["y_test"].mean() == pytest.approx(0.1, abs=0.01)

    # Test week 2
    assert test_sets[1]["period_start"] == datetime(2020, 3, 8)
    assert len(test_sets[1]["y_test"]) == 50
    assert test_sets[1]["y_test"].mean() == pytest.approx(0.1, abs=0.01)

@pytest.mark.parametrize("target_array, target_ratio, expected_len, expected_sum", [
    # Case 1: Enough majority samples
    (np.array([0]*90 + [1]*10), 0.1, 100, 10),
    # Case 2: Not enough majority samples, downsample minority
    (np.array([0]*9 + [1]*10), 0.1, 10, 1),
])
def test_get_stratified_indices(target_array, target_ratio, expected_len, expected_sum):
    """Test the stratification logic directly."""
    loader = DataLoader()
    indices = loader.get_stratified_indices(target_array, target_ratio)
    assert len(indices) == expected_len
    assert target_array[indices].sum() == expected_sum
