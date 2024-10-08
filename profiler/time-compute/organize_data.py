import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import argparse

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

def process_and_save_model(gpu_name, time_type, model_name, base_dir, target_dir, predict_unknowns):
    model_path = os.path.join(base_dir, gpu_name, time_type, model_name)
    if os.path.isdir(model_path):
        data = []
        for batch_size in os.listdir(model_path):
            batch_size_path = os.path.join(model_path, batch_size)
            if os.path.isdir(batch_size_path):
                time_file = os.path.join(batch_size_path, 'time.txt')
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
        csv_path = os.path.join(target_path, 'time_by_batch_size.csv')
        
        df.to_csv(csv_path, index=False)
        df_original = df.copy()
        
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

        model_path = os.path.join(target_path, 'regression_model.pkl')
        joblib.dump((model, poly, mse, r2), model_path)
        
        if predict_unknowns:
            for i in range(1, 257):
                if i not in df_original['Batch_Size'].values:
                    X_unknown = np.array([[i]])
                    X_unknown_poly = poly.transform(X_unknown)
                    y_unknown = model.predict(X_unknown_poly)
                    df_original = df_original.append({'Batch_Size': i, 'Time_In_SECONDS': y_unknown[0]}, 
                                   ignore_index=True)
            df_original = df_original.sort_values(by='Batch_Size')
            df_original.to_csv(csv_path, index=False)
            

def process_all_models(base_dir='raw_data', target_dir='processed_data', predict_unknowns=True):
    for gpu_name in os.listdir(base_dir):
        gpu_path = os.path.join(base_dir, gpu_name)
        if os.path.isdir(gpu_path):
            for time_type in os.listdir(gpu_path):
                time_type_path = os.path.join(gpu_path, time_type)
                if os.path.isdir(time_type_path):
                    for model_name in os.listdir(time_type_path):
                        model_path = os.path.join(time_type_path, model_name)
                        if os.path.isdir(model_path):
                            process_and_save_model(gpu_name, time_type, model_name, base_dir, target_dir, predict_unknowns)


def main():
    parser = argparse.ArgumentParser(description='Process and save model.')
    parser.add_argument('--base_dir', type=str, default='raw_data',
                        help='Base directory containing raw data.')
    parser.add_argument('--target_dir', type=str, default='../../profiled_data/time/compute',
                        help='Target directory to save processed data and models.')
    parser.add_argument('--predict_unknowns', action='store_true',
                        help='Predict unknown times for batch sizes between 1-256.')
    args = parser.parse_args()

    process_all_models(args.base_dir, args.target_dir, args.predict_unknowns)


if __name__ == "__main__":
    main()
