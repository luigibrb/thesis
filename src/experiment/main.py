from experiment.modules.DataLoader import DataLoader
from experiment.modules.CrossConformalTrainer import CrossConformalTrainer
from experiment.modules.ExperimentRunner import ExperimentRunner
from experiment.modules.Plotter import Plotter
from experiment.settings import settings
import uuid

# 1. Setup
loader = DataLoader()
X, y = loader.load_bodmas_data()
meta = loader.load_bodmas_metadata()
X_train, y_train, test_sets = loader.split_data(meta, X, y, granularity="1d")

del X, y, meta

# 2. Train
trainer = CrossConformalTrainer(k=5, voting_threshold=0.8, model_type='rf')

calibration_settings = {
    'n_iter': 5000,
    'target_metric': 'max_f1',
    'rejection_rate_max': 0.10,
    # 'target_metric': 'min_rejection_rate',
    # 'f1_min': 0.995,
}

print("Training & Calibrating CCE...")

trainer.fit_calibrate(X_train, y_train, calibration_settings=calibration_settings)

# 3. Run Experiment
run_id = str(uuid.uuid4())
print(f"Starting Run ID: {run_id}")

runner = ExperimentRunner(output_dir=settings.results_path / "runs")
df_results = runner.run(trainer, test_sets, run_id=run_id)

Plotter.plot_cce_results(df_results)
Plotter.plot_rejection_rate(df_results, W=5)
