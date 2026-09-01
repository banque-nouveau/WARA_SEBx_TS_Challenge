import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
from pathlib import Path
from copy import deepcopy
from sklearn.metrics import  roc_curve, roc_auc_score


class XGBoostModel:
    def __init__(self, objective, eval_metric, num_boost_round, threshold, scale_pos_weight, learning_rate, verbose_eval, early_stopping_rounds, l2_reg, l1_reg, max_depth, min_child_weight, min_loss_reduction, subsample, colsample_bytree):
        """
        Initialize the XGBoost model.
        Args:
            objective: The objective function to use.
            eval_metric: The evaluation metric to use.
            num_boost_round: The number of boosting rounds to perform.
            threshold: The threshold for the model to predict.
            scale_pos_weight: The scale of the positive class.
            learning_rate: The learning rate.
            verbose_eval: The number of boosting rounds between verbose outputs.
            early_stopping_rounds: The number of boosting rounds to wait before early stopping.
            l2_reg: The L2 regularization parameter.
            l1_reg: The L1 regularization parameter.
            max_depth: The maximum depth of the trees.
            min_child_weight: The minimum child weight.
            min_loss_reduction: The minimum loss reduction to make a split.
            subsample: The fraction of the training data to use.
            colsample_bytree: The fraction of the features to use.
        """

        self.params = {
            'objective': objective,
            'eval_metric': eval_metric,
            'eta': learning_rate,
            'lambda': l2_reg,
            'alpha': l1_reg,
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'gamma': min_loss_reduction,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree
        }
        self.num_boost_round = num_boost_round
        self.verbose_eval = verbose_eval
        self.early_stopping_rounds = early_stopping_rounds
        self.threshold = threshold
    
    def fit(self, x_train, y_train, x_test, y_test):
        """
        Fit the XGBoost model.
        """
        # Convert data to DMatrix, which is XGBoost's optimized data structure
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_test, label=y_test)

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds = self.early_stopping_rounds,
            verbose_eval=self.verbose_eval
        )

    def evaluate_xgb(self, x_train, y_train, x_test, y_test):
        """
        Evaluate the XGBoost model.
        """

        # Convert data to DMatrix, which is XGBoost's optimized data structure
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_test, label=y_test)

        y_pred_train = self.model.predict(dtrain)
        y_pred = self.model.predict(dtest)

        # Convert probabilities to class labels
        threshold = self.threshold
        y_pred_train_labels = (y_pred_train > threshold).astype(int)
        y_pred_labels = (y_pred > threshold).astype(int)

        # Now use these for metrics
        train_acc = accuracy_score(y_train, y_pred_train_labels)
        test_acc = accuracy_score(y_test, y_pred_labels)

        class_report = classification_report(y_test, y_pred_labels)
        
        CF = confusion_matrix(y_test, y_pred_labels)
        
        if CF.shape != (2, 2):
            print("Confusion matrix is not 2x2, skipping false positive/negative rate calculation.")

        tn, fp, fn, tp = CF.ravel()

        fp_rate = fp / (fp + tn) * 100
        fn_rate = fn / (fn + tp) * 100

        # AUC
        auc = roc_auc_score(y_test, y_pred)

        # ROC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        fpr = fpr.tolist()
        tpr = tpr.tolist()
        thresholds = thresholds.tolist()

        self.result = {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "class_report": class_report,
            "fp_rate": fp_rate,
            "fn_rate": fn_rate,
            "auc": auc,
            "roc_curve": {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
        }
        
        return self.result
    
    def save_evaluation(self, dest_path: Path, trainer_cfg):
        """
        Save the evaluation results to a JSON file.
        """
        cfg = deepcopy(trainer_cfg)
        cfg["dset_cfg"].pop("dset_path")
        if not isinstance(cfg["model_cfg"]["_target_"], str):
            # Convert class to string. Assume string is on the format <class 'module.ClassName'>.
            cfg["model_cfg"]["_target_"] = str(cfg["model_cfg"]["_target_"]).split("'")[1]

        # Add config to the results
        self.result["run_cfg"] = cfg["run_cfg"]
        self.result["dset_cfg"] = cfg["dset_cfg"]
        self.result["model_cfg"] = cfg["model_cfg"]
        
        # Save to JSON file in wdir
        if dest_path.exists() and dest_path.stat().st_size > 0:
            with open(dest_path, "r") as f:
                all_results = json.load(f)
        else:
            all_results = []

        all_results.append(self.result)

        with open(dest_path, "w") as f:
            json.dump(all_results, f, indent=2)