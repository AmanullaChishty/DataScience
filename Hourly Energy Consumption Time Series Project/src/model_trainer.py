import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from src.decorators import log_execution,time_execution

class ModelTrainer:
    """
    Class to train a time series forecasting model.
    Uses lag and rolling features as predictors to forecast the energy consumption value.
    """
    def __init__(self,feature_cols,target_col):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.model = LinearRegression()

    @log_execution
    @time_execution
    def train(self, df: pd.DataFrame):
        # Split the data in time order (no shuffling)
        split_idx = int(len(df)*0.8)
        train_data = df.iloc[:split_idx]
        test_data = df.iloc[split_idx:]

        x_train = train_data[self.feature_cols]
        y_train = train_data[self.target_col]
        X_test = test_data[self.feature_cols]
        y_test = test_data[self.target_col]

        self.model.fit(x_train,y_train)
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return {'model':self.model,'mse':mse,'X_test':X_test,'y_test':y_test, 'predictions':predictions}
    
    def predict(self,X):
        return self.model.predict(X)
    
    def evaluate(self,X_test,y_test):
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test,predictions)
        return mse
        