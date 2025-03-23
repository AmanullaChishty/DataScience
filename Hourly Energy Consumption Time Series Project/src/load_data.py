import kagglehub
from kagglehub import KaggleDatasetAdapter
    
df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "robikscube/hourly-energy-consumption",
        "DOM_hourly.csv"
        )
df.to_csv("data/hourly-energy-consumption.csv")