import kagglehub
from kagglehub import KaggleDatasetAdapter

def load_data():   
        df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                "robikscube/hourly-energy-consumption",
                "DOM_hourly.csv"
                )
        df.rename(columns={'DOM_MW':'Value'}, inplace=True)
        return df.to_csv("data/hourly_energy_consumption.csv")