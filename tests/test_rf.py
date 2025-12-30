import numpy as np
from experiment.modules.CrossConformalTrainer import CrossConformalTrainer

def test_rf_initialization_and_fit():
    # Create dummy data
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    # Initialize with rf
    trainer = CrossConformalTrainer(k=2, model_type='rf')
    
    # Fit (using small settings for speed)
    calibration_settings = {'n_iter': 10, "max_rej": 0.1}
    trainer.fit_calibrate(X, y, calibration_settings=calibration_settings)
    
    assert len(trainer.models) == 2
    assert trainer.model_type == 'rf'
    
    # Predict
    X_test = np.random.rand(10, 5)
    preds = trainer.predict(X_test)
    assert len(preds) == 10
