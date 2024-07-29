import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import argparse
import shutil

def remove_outliers_using_residuals(df, threshold=1.5):
    X = df[['Batch_Size']].values
    y = df['Time_In_SECONDS'].values

    poly = PolynomialFeatures(degree=1)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    y_pred = model.predict(X_poly)
    residuals = np.abs(y - y_pred)

    median_residual = np.median(residuals)
    mad_residual = np.median(np.abs(residuals - median_residual))

    outlier_threshold = median_residual + threshold * mad_residual

    non_outliers = residuals <= outlier_threshold
    return df[non_outliers]

def process_and_save_measurement(gpu_name, time_type, model_name, base_dir, target_dir):
    model_path = os.path.join(base_dir, gpu_name, time_type, model_name)
    if os.path.isdir(model_path):
        if os.path.isdir(model_path):
            model_time_file = os.path.join(model_path, 'model_transfer_time.txt')
            if os.path.isfile(model_time_file):
                target_path = os.path.join(target_dir, time_type, gpu_name, model_name)
                os.makedirs(target_path, exist_ok=True)
                target_model_time_file = os.path.join(target_path, 'model_transfer_time.txt')
                shutil.copyfile(model_time_file, target_model_time_file)
        data = []
        for batch_size in os.listdir(model_path):
            batch_size_path = os.path.join(model_path, batch_size)
            if os.path.isdir(batch_size_path):
                time_file = os.path.join(batch_size_path, 'data_transfer_time.txt')
                if os.path.isfile(time_file):
                    with open(time_file, 'r') as file:
                        time_value = float(file.read().strip())
                        data.append({
                            'Batch_Size': int(batch_size),
                            'Time_In_SECONDS': time_value
                        })
        if not data:
            print(f"No data found for {gpu_name} | {time_type} | {model_name}")
            return
        
        df = pd.DataFrame(data)

        df = df.sort_values(by='Batch_Size')

        target_path = os.path.join(target_dir, time_type, gpu_name, model_name)
        os.makedirs(target_path, exist_ok=True)
        csv_path = os.path.join(target_path, 'data_transfer_time_by_batch_size.csv')
        
        df.to_csv(csv_path, index=False)
        
        df = remove_outliers_using_residuals(df)

        X = df[['Batch_Size']].values
        y = df['Time_In_SECONDS'].values

        poly = PolynomialFeatures(degree=1)
        X_poly = poly.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_poly, y)

        y_pred = model.predict(X_poly)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        print(f'{gpu_name} | {time_type} | {model_name} - MSE: {mse} - R^2: {r2}')

        model_path = os.path.join(target_path, 'data_transfer_regression_model.pkl')
        joblib.dump((model, poly, mse, r2), model_path)

def process_all_measurements(base_dir='raw_data', target_dir='processed_data'):
    for gpu_name in os.listdir(base_dir):
        gpu_path = os.path.join(base_dir, gpu_name)
        if os.path.isdir(gpu_path):
            for time_type in os.listdir(gpu_path):
                time_type_path = os.path.join(gpu_path, time_type)
                if os.path.isdir(time_type_path):
                    for model_name in os.listdir(time_type_path):
                        model_path = os.path.join(time_type_path, model_name)
                        if os.path.isdir(model_path):
                            process_and_save_measurement(gpu_name, time_type, model_name, base_dir, target_dir)


def main():
    parser = argparse.ArgumentParser(description='Process and save model.')
    parser.add_argument('--base_dir', type=str, default='raw_data',
                        help='Base directory containing raw data.')
    parser.add_argument('--target_dir', type=str, default='../../profiled_data/time/',
                        help='Target directory to save processed data and models.')
    args = parser.parse_args()

    process_all_measurements(args.base_dir, args.target_dir)


if __name__ == "__main__":
    main()
