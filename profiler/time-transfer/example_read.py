import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import argparse


def predict_time(time_type, gpu_name, model_name, batch_sizes, target_dir='processed_data'):
    model_path = f'{target_dir}/{time_type}/{gpu_name}/{model_name}/regression_model.pkl'
    model, poly, mse, r2 = joblib.load(model_path)
    new_batch_sizes = np.array(batch_sizes).reshape(-1, 1)
    new_batch_sizes_poly = poly.transform(new_batch_sizes)
    predicted_times = model.predict(new_batch_sizes_poly)
    return predicted_times, mse, r2


def read_time(time_type, gpu_name, model_name, batch_sizes, target_dir='processed_data'):
    model_path = f'{target_dir}/{time_type}/{gpu_name}/{model_name}/time_by_batch_size.csv'
    df = pd.read_csv(model_path)
    time_values = []
    for batch_size in batch_sizes:
        row = df[df['Batch_Size'] == batch_size]
        if not row.empty:
            time_value = row['Time_In_SECONDS'].values[0]
            time_values.append(time_value)
        else:
            print(f'Batch size {batch_size} not found in the CSV file.')
            time_values.append(-1)
    return time_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read and predict execution times')
    parser.add_argument('--time_type', type=str, default='forward', help='Type of time to predict (forward/backward)')
    parser.add_argument('--gpu_name', type=str, default='Tesla_M40', help='Name of the GPU')
    parser.add_argument('--model_name', type=str, default='alexnet', help='Name of the model')
    parser.add_argument('--new_batch_sizes', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128, 256], help='New batch sizes to predict')
    parser.add_argument('--target_dir', type=str, default='../../profiled_data/time', help='Directory containing the processed data')
    args = parser.parse_args()
    
    time_type = args.time_type
    if time_type in ['forward', 'backward']:
        time_type = f'compute/{time_type}'
    gpu_name = args.gpu_name
    model_name = args.model_name
    new_batch_sizes = args.new_batch_sizes
    target_dir = args.target_dir
    
    new_batch_sizes = range(1, 257)

    predicted_times, mse, r2 = predict_time(time_type, gpu_name, model_name, new_batch_sizes, target_dir)
    read_times = read_time(time_type, gpu_name, model_name, new_batch_sizes, target_dir)
    for i in range(len(new_batch_sizes)):
        print(f'Batch size: {new_batch_sizes[i]} | Predicted time: {predicted_times[i]} | Actual time: {read_times[i]} | Error: {predicted_times[i] - read_times[i]}')
    print(f'Model MSE: {mse}, R^2: {r2}')
