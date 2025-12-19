from experiment.DataLoader import DataLoader


data_loader = DataLoader()
metadata_df = data_loader.load_bodmas_metadata()
X, y = data_loader.load_bodmas_data()

X_train, y_train, X_val, y_val, X_cal, y_cal, test_sets = data_loader.split_data(metadata_df, X, y)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print(f"Number of weekly test windows: {len(test_sets)}")
