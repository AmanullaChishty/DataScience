import kagglehub
from kagglehub import KaggleDatasetAdapter

def load_data():   
        df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                "robikscube/hourly-energy-consumption",
                "DOM_hourly.csv"
                )
        return df.to_csv("data/hourly_energy_consumption.csv")