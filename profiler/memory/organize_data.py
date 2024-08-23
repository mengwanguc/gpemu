import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import joblib

def read_float_from_file(filepath):
    try:
        with open(filepath, 'r') as file:
            return float(file.read().strip())
    except:
        return None

def predict_and_save_models(df, batch_sizes, model_dir):
    poly_model_peak = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly_model_persistent = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    
    X_train = df[['batch_size']]
    y_train_peak = df['peak']
    y_train_persistent = df['persistent']
    
    poly_model_peak.fit(X_train, y_train_peak)
    poly_model_persistent.fit(X_train, y_train_persistent)
    
    missing_batch_sizes = [bs for bs in batch_sizes if bs not in df['batch_size'].values]
    
    if missing_batch_sizes:
        X_missing = pd.DataFrame(missing_batch_sizes, columns=['batch_size'])
        
        predicted_peak = poly_model_peak.predict(X_missing)
        predicted_persistent = poly_model_persistent.predict(X_missing)
        
        for i, bs in enumerate(missing_batch_sizes):
            df = df.append({
                'batch_size': bs,
                'peak': predicted_peak[i],
                'persistent': predicted_persistent[i]
            }, ignore_index=True)
    
    df.sort_values('batch_size', inplace=True)
    
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(poly_model_peak, os.path.join(model_dir, 'peak_model.pkl'))
    joblib.dump(poly_model_persistent, os.path.join(model_dir, 'persistent_model.pkl'))
    
    return df

base_dir = 'raw_data'
output_dir = '../../profiled_data/memory/'
required_batch_sizes = list(range(1, 257))

for gpu_name in os.listdir(base_dir):
    gpu_dir = os.path.join(base_dir, gpu_name)
    if os.path.isdir(gpu_dir):
        for arch in os.listdir(gpu_dir):
            arch_dir = os.path.join(gpu_dir, arch)
            if os.path.isdir(arch_dir):
                data = []
                for batch_size in os.listdir(arch_dir):
                    batch_dir = os.path.join(arch_dir, batch_size)
                    if os.path.isdir(batch_dir):
                        peak_file = os.path.join(batch_dir, 'peak.txt')
                        persistent_file = os.path.join(batch_dir, 'persistent.txt')
                        if os.path.isfile(peak_file) and os.path.isfile(persistent_file):
                            peak_value = read_float_from_file(peak_file)
                            persistent_value = read_float_from_file(persistent_file)
                            if peak_value is not None and persistent_value is not None:
                                data.append({
                                    'batch_size': int(batch_size),
                                    'peak': peak_value,
                                    'persistent': persistent_value
                                })
                
                df = pd.DataFrame(data)
                
                model_dir = os.path.join(output_dir, f'{gpu_name}/{arch}/models')
                df = predict_and_save_models(df, required_batch_sizes, model_dir)
                
                target_dir = os.path.join(output_dir, f'{gpu_name}/{arch}')
                os.makedirs(target_dir, exist_ok=True)
                
                output_csv = os.path.join(target_dir, 'memory.csv')
                
                df.to_csv(output_csv, index=False)
