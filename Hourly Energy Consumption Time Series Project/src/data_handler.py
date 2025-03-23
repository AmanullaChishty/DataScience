import pandas as pd
from src.decorators import log_execution,time_execution

class TimeSeriesDataHandler:
    """
    Class to load and preprocess time series data.
    Includes methods for loading data, handling missing values,
    treating outliers, and ensuring data consistency.
    """

    def __init__(self,data_path):
        self.data_path = data_path
        self.df = None
    
    @log_execution
    @time_execution
    def load_data(self,index_col='Datetime',parse_dates=True):
        """Load CSV data with datetime index."""
        try:
            self.df = pd.read_csv(self.data_path,index_col=index_col,parse_dates=parse_dates)
            self.df.sort_index(inplace=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found at {self.data_path}")
        return self.df
    
    @log_execution
    def handle_missing_values(self,method="ffill"):
        """Fill missing values using specified method."""
        if self.df is not None:
            self.df.fillna(method=method, inplace=True)

            if method == "interpolate":
                self.df.interpolate(inplace=True)
            else:
                raise ValueError("Invalid method for handling missing values.")
        return self.df
    
    @log_execution
    def treat_outliers_iqr(self,column,threshold=1.5):
        """Remove outliers in the specified column using the IQR method."""
        if self.df is not None and column in self.df.columns:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3-Q1
            lower_bound = Q1 - threshold*IQR
            upper_bound = Q3 + threshold*IQR
            self.df = self.df[(self.df[column]>=lower_bound)&(self.df[column]<=upper_bound)]
        else:
            raise ValueError(f"Column '{column}' not found or data not loaded ")
        return self.df

    @log_execution
    def ensure_data_consistency(self):
        """Ensure the datetime index and numeric columns are in correct format."""
        if self.df is not None:
            if not isinstance(self.df.index,pd.DatetimeIndex):
                self.df.index = pd.to_datetime(self.df.index)
            # Ensure numeric columns are of numeric type (if needed)
            numeric_cols = self.df.select_dtypes(include=['object']).columns
            for col in numeric_cols:
                try:
                    self.df[col]=pd.to_numeric(self.df[col])
                except Exception:
                    continue
        return self.df