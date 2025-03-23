import os 
import time
import src.load_data as ld
from src.data_handler import TimeSeriesDataHandler
from src.feature_engineer import TimeSeriesFeatureEngineer
from src.model_trainer import ModelTrainer
from src.model_deployer import ModelDeployer

def main():
    # Data Acquisition and Initial Exploration
    data_handler = TimeSeriesDataHandler(data_path="data/hourly_energy_consumption.csv")
    df = data_handler.load_data()
    df = data_handler.handle_missing_values(method='ffill')
    df = data_handler.treat_outliers_iqr(column='Value',threshold=1.5)
    df = data_handler.ensure_data_consistency()

    # Feature Engineering 
    feature_engineer = TimeSeriesFeatureEngineer(df)
    df = feature_engineer.extract_datetime_features()
    df = feature_engineer.create_rolling_stats(column='Value',window=24,stats=['mean', 'std'])
    df = feature_engineer.create_lag_features(column='Value', lags=[1,2,3])
    df = feature_engineer.create_lead_features(column='Value', leads=[1])
    df = feature_engineer.drop_na()

    # Model Training
    feature_cols = ['lag_1','lag_2','lag_3','rolling_mean_24']
    target_col = 'Value'
    model_trainer = ModelTrainer(feature_cols,target_col)
    result = model_trainer.train(df)
    print(f"Test MSE: {result['mse']:.4f}")

    # Model Deployment
    model_deployer = ModelDeployer(model=result['model'])
    model_filename = model_deployer.save_model()
    print(f"Model saved to {model_filename}")

    latest_features = df.iloc[-1:][feature_cols]
    prediction = result['model'].predict(latest_features)
    print(f"Real-time prediction: {prediction[0]:.2f}")

    model_deployer.run_flask_app()



if __name__ == '__main__':
    ld.load_data()

    file_path = "data/hourly_energy_consumption.csv"

    # Wait until the file is created
    while not os.path.exists(file_path):
        print(f"Waiting for {file_path} to be created...")
        time.sleep(10)

    print(f"File {file_path} found. Proceeding...")
    main()
