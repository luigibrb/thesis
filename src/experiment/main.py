from experiment.modules.DataLoader import DataLoader
from experiment.modules.CrossConformalTrainer import CrossConformalTrainer
from experiment.modules.ExperimentRunner import ExperimentRunner
from experiment.settings import settings
import uuid

# 1. Setup
loader = DataLoader()
X, y = loader.load_bodmas_data()
meta = loader.load_bodmas_metadata()
X_train, y_train, test_sets = loader.split_data(meta, X, y)

del X, y, meta

# 2. Train
trainer = CrossConformalTrainer(k=5, model_type='rf')
print("Training & Calibrating CCE...")
trainer.fit_calibrate(X_train, y_train, calibration_settings={'n_iter': 5000, 'max_rej': 0.10})

# 3. Run Experiment
run_id = str(uuid.uuid4())
print(f"Starting Run ID: {run_id}")

runner = ExperimentRunner(output_dir=settings.results_path / "runs")
df_results = runner.run(trainer, test_sets, run_id=run_id)
