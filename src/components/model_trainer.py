import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
try:
    from xgboost import XGBRegressor
    _HAS_XGBOOST = True
except Exception:
    XGBRegressor = None
    _HAS_XGBOOST = False

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
            }

            if _HAS_XGBOOST:
                models["XGBRegressor"] = XGBRegressor()
            else:
                logging.warning("xgboost is not installed; skipping XGBRegressor")
                # keep going without XGB

            models.update({
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            })

            # parameter grid for GridSearchCV
            param_grid = {
                "Random Forest": {"n_estimators": [50, 100], "max_depth": [None, 10]},
                "Decision Tree": {"criterion": ["squared_error"], "max_depth": [None, 10]},
                "Gradient Boosting": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
                "Linear Regression": {"fit_intercept": [True, False]},
                "K-Neighbors Regressor": {"n_neighbors": [3, 5]},
            }

            if _HAS_XGBOOST:
                param_grid["XGBRegressor"] = {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}
            param_grid.update({
                "CatBoosting Regressor": {"depth": [4, 6]},
                "AdaBoost Regressor": {"n_estimators": [50, 100]}
            })

            model_report: dict = evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                                                 models=models, param=param_grid)

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)

                model_report[model_name] = test_model_score
            ## To get the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))

            ### To get the best model name from the dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)       