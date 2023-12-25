import logging

import numpy as np
from sklearn import metrics


class Evaluation_Indicators():
    # MAPE和SMAPE需要自己实现
    def mape(self, y_true, y_pred):
        return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

    def smape(self, y_true, y_pred):
        return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

    def evaluation(self, y_true, y_pred):
        # MSE
        logging.warning("------------------------------------------------------------")
        logging.warning(f"MSE-->0:{metrics.mean_squared_error(y_true, y_pred)}")
        # RMSE
        logging.warning("------------------------------------------------------------")
        logging.warning(f"RMSE-->0:{np.sqrt(metrics.mean_squared_error(y_true, y_pred))}")
        # MAE
        logging.warning("------------------------------------------------------------")
        logging.warning(f"MAE-->0:{metrics.mean_absolute_error(y_true, y_pred)}")
        # MAPE
        logging.warning("------------------------------------------------------------")
        logging.warning(f"MAPE:{self.mape(y_true, y_pred)}")
        # SMAPE
        logging.warning("------------------------------------------------------------")
        logging.warning(f"SMAPE:{self.smape(y_true, y_pred)}")
        # R2
        logging.warning("------------------------------------------------------------")
        logging.warning(f"R2_SCORE-->1:{(y_true, y_pred)}")

    def evaluation_list(self, y_true, y_pred):
        list = []
        list.append(metrics.mean_squared_error(y_true, y_pred))
        list.append(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))
        list.append(metrics.mean_absolute_error(y_true, y_pred))
        list.append(self.mape(y_true, y_pred))
        list.append(self.smape(y_true, y_pred))
        list.append(metrics.r2_score(y_true, y_pred))
        return list