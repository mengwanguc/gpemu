import os
import pandas as pd

def read_float_from_file(filepath):
    with open(filepath, 'r') as file:
        return float(file.read().strip())

base_dir = 'raw_data'
output_dir = '../../profiled_data/memory/'

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
                            data.append({
                                'batch_size': int(batch_size),
                                'peak': peak_value,
                                'persistent': persistent_value
                            })
                
                df = pd.DataFrame(data)
                
                target_dir = os.path.join(output_dir, f'{gpu_name}/{arch}')
                os.makedirs(target_dir, exist_ok=True)
                
                output_csv = os.path.join(target_dir, 'memory.csv')
                
                df.to_csv(output_csv, index=False)
                
                print(f"Data for {gpu_name}/{arch} has been organized and saved to {output_csv}")
