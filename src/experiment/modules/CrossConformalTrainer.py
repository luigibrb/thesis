import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import numpy as np

class CrossConformalTrainer:
    def __init__(self, k=5, model_type='lgb'):
        """
        k: number of folds for Cross Conformal Prediction.
        model_type: 'lgb' for LGBMClassifier, 'svm' for LinearSVC (calibrated), 'rf' for RandomForestClassifier.
        """
        self.k = k
        self.model_type = model_type
        self.models = []
        self.fold_data = [] 

    def _generate_folds(self, X, y):
        kf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=42)
        for train_index, cal_index in kf.split(X, y):
            yield train_index, cal_index

    def fit_calibrate(self, X, y, calibration_settings=None):
        """
        Performs Training and Calibration.
        calibration_settings: dict with parameters for random search (e.g., max_rej_rate)
        """
        self.models = []
        self.fold_data = []
        
        if calibration_settings is None:
            calibration_settings = {'n_iter': 5000, 'max_rej': 0.15}

        # 1. Partition Z equally
        for fold_idx, (train_idx, cal_idx) in enumerate(self._generate_folds(X, y)):
            X_proper_train, X_cal = X[train_idx], X[cal_idx]
            y_proper_train, y_cal = y[train_idx], y[cal_idx]

            # 2. Fit model on proper training set
            if self.model_type == 'lgb':
                model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
            elif self.model_type == 'svm':
                # LinearSVC doesn't support predict_proba natively, so we use CalibratedClassifierCV
                # dual="auto" picks the best solver for the dataset shape
                base_svc = LinearSVC(random_state=42, dual="auto")
                model = CalibratedClassifierCV(base_svc)
            elif self.model_type == 'rf':
                model = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=100, max_depth=12, min_samples_leaf=5)
            elif self.model_type == 'sgd':
                model = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, max_iter=1000, tol=1e-3, n_jobs=-1, random_state=42)
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}. Use 'lgb', 'svm' or 'rf'.")

            model.fit(X_proper_train, y_proper_train)
            self.models.append(model)
            
            # --- CALIBRATION PHASE PER FOLD ---
            
            # Predict and Probability Calculation
            cal_probs = model.predict_proba(X_cal)
            y_pred_cal = model.predict(X_cal)
            
            # Construction of the "Bag" of nonconformity scores (S) using ground truths
            # Note: NCM = 1 - prob(y_true)
            classes = np.unique(y)
            fold_scores_bag = {} 
            
            for c in classes:
                # We select only the points that are ACTUALLY class c in the calibration set
                mask_true = (y_cal == c)
                if np.sum(mask_true) > 0:
                     # NCM: 1 - true class probability
                    fold_scores_bag[c] = 1 - cal_probs[mask_true, c]
                else:
                    fold_scores_bag[c] = np.array([])

            # P-values calculation for each calibration point
            p_values = np.zeros(len(y_cal))
            for i in range(len(y_cal)):
                predicted_label = y_pred_cal[i]
                alpha_i = 1 - cal_probs[i, predicted_label] # NCM of the point
                
                # S is the bag of the PREDICTED class
                S = fold_scores_bag.get(predicted_label, np.array([]))
                
                if len(S) > 0:
                    # Eq (5): proportion of points in the bag "stranger" or equal to the current point
                    p_values[i] = np.sum(S >= alpha_i) / len(S)
                else:
                    p_values[i] = 0.0 # Conservative fallback
            
            # 3. Find Thresholds T*_j for this fold
            # We use Random Search LOCALLY for this fold
            best_thresholds = self._random_search_thresholds(
                y_cal, y_pred_cal, p_values, classes,
                n_iterations=calibration_settings['n_iter'],
                max_rejection_rate=calibration_settings['max_rej']
            )
            
            # We save everything needed for the test phase of this fold
            self.fold_data.append({
                'scores_bag': fold_scores_bag,
                'thresholds': best_thresholds
            })
            
            print(f"Fold {fold_idx+1}/{self.k} completed. Thresholds: {best_thresholds}")

    def _random_search_thresholds(self, y_true, y_pred, p_values, classes, n_iterations, max_rejection_rate):
        best_f1 = -1.0
        # Initialize with high thresholds (all rejected) or low (all accepted) depending on the strategy.
        # Here we start from 0 (all accepted) as a safe baseline.
        best_thresholds = {c: 0.0 for c in classes} 
        
        for _ in range(n_iterations):
            # Sample random thresholds
            current_thresholds = {c: np.random.rand() for c in classes}
            
            # Apply thresholds
            # Note: p_value < threshold => REJECT (Drift). p_value >= threshold => ACCEPT.
            accepted_mask = np.array([
                p_values[i] >= current_thresholds[y_pred[i]] 
                for i in range(len(p_values))
            ])
            
            # Constraint Check (G)
            current_rejection_rate = 1.0 - (np.sum(accepted_mask) / len(accepted_mask))
            
            if current_rejection_rate <= max_rejection_rate:
                # Optimize Metric (F) - F1 on KEPT elements
                if np.sum(accepted_mask) > 0:
                    current_f1 = f1_score(
                        y_true[accepted_mask], 
                        y_pred[accepted_mask], 
                        average='macro'
                    )
                else:
                    current_f1 = 0.0
                
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_thresholds = current_thresholds
                    
        return best_thresholds

    def predict(self, X_test):
        """
        CCE Test Phase.
        """
        n_test = X_test.shape[0]
        votes = np.zeros(n_test)
        
        # Iterate over each model/fold
        for i in range(self.k):
            model = self.models[i]
            bag = self.fold_data[i]['scores_bag']
            thresholds = self.fold_data[i]['thresholds']
            
            # Predict
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)
            
            # Calculate p-value for each test point with respect to fold i
            for j in range(n_test):
                y_hat = preds[j]
                alpha_test = 1 - probs[j, y_hat] # NCM
                
                S = bag.get(y_hat, np.array([]))
                
                if len(S) > 0:
                    p_val = np.sum(S >= alpha_test) / len(S)
                else:
                    p_val = 0.0
                
                # Check threshold
                if p_val >= thresholds[y_hat]:
                    votes[j] += 1
        
        # Majority Vote
        # Emit 1 (Accept) if s > k/2, otherwise 0 (Reject)
        final_decisions = votes > (self.k / 2)
        return final_decisions

    def predict_labels(self, X_test):
        """
        Returns the predicted label by aggregating the k models of the CCE.
        Used to calculate baseline accuracy and F1 of kept/rejected.
        """
        n_test = X_test.shape[0]
        # Sum of probabilities predicted by all k models
        total_probs = np.zeros((n_test, 2)) # Assuming binary
        
        for model in self.models:
            total_probs += model.predict_proba(X_test)
            
        # Average of probabilities (Soft Voting)
        avg_probs = total_probs / self.k
        return np.argmax(avg_probs, axis=1)
