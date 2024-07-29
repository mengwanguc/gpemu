import time
import asyncio
import pandas as pd

class ComputeTimeEmulator:
    def __init__(self, gpu='Tesla_M40', model='resnet18', batch_size=256,
                 forward_time=0.05, backward_time=0.05, 
                 from_databse=False, database_path='~/gpemu-private/profiled_data/',
                 mode='async'):
        self.gpu = gpu
        self.model = model
        self.batch_size = batch_size
        if not from_databse:
            self.forward_time = forward_time
            self.backward_time = backward_time
        else:
            compute_time_path = f'{database_path}/time/compute/'
            self.forward_time = self.get_compute_time_from_database(
                'forward', gpu, model, batch_size, compute_time_path)
            self.backward_time = self.get_compute_time_from_database(
                'backward', gpu, model, batch_size, compute_time_path)
        self.mode = mode
    
    def get_forward_time(self):
        return self.forward_time
    
    def get_backward_time(self):
        return self.backward_time
    
    def get_compute_time_from_database(self, time_type, gpu_name, 
                                       model_name, batch_size, database_dir):
        time_path = f'{database_dir}/{time_type}/{gpu_name}/{model_name}/time_by_batch_size.csv'
        df = pd.read_csv(time_path)
        row = df[df['Batch_Size'] == batch_size]
        if not row.empty:
            compute_time = row['Time_In_SECONDS'].values[0]
        else:
            model_path = f'{database_dir}/{time_type}/{gpu_name}/{model_name}/regression_model.pkl'
            model, poly, mse, r2 = joblib.load(model_path)
            new_batch_sizes = np.array([batch_size]).reshape(-1, 1)
            new_batch_sizes_poly = poly.transform(new_batch_sizes)
            predicted_times = model.predict(new_batch_sizes_poly)
            return predicted_times[0]

    async def async_emulate_forward_pass(self):
        await asyncio.sleep(self.forward_time)

    async def async_emulate_backward_pass(self):
        await asyncio.sleep(self.backward_time)

    def sync_emulate_forward_pass(self):
        time.sleep(self.forward_time)

    def sync_emulate_backward_pass(self):
        time.sleep(self.backward_time)

    async def async_train_batch(self):
        await self.async_emulate_forward_pass()
        await self.async_emulate_backward_pass()

    def sync_train_batch(self):
        self.sync_emulate_forward_pass()
        self.sync_emulate_backward_pass()

    async def emulate_compute(self):
        if self.mode == 'async':
            await self.async_train_batch()
        else:
            self.sync_train_batch()

class TransferTimeEmulator:
    def __init__(self, gpu='Tesla_M40', model='resnet18', batch_size=256,
                 transfer_time=0.01, model_transfer_time=0.1,
                 from_databse=False, 
                 database_path='~/gpemu-private/profiled_data/',
                 mode='async'):
        self.gpu = gpu
        self.model = model
        self.batch_size = batch_size
        if not from_databse:
            self.transfer_time = transfer_time
            self.model_transfer_time = model_transfer_time
        else:
            transfer_time_path = f'{database_path}/time/transfer/'
            self.transfer_time = self.get_transfer_time_from_database(
                gpu, model, batch_size, transfer_time_path)
            self.model_transfer_time = self.get_model_transfer_time_from_database(
                gpu, model, transfer_time_path)
        self.mode = mode
    
    def get_transfer_time(self):
        return self.transfer_time
    
    def get_model_transfer_time(self):
        return self.model_transfer_time
    
    def get_model_transfer_time_from_database(self, gpu_name, model_name, database_dir):
        time_path = f'{database_dir}/{gpu_name}/{model_name}/model_transfer_time.txt'
        model_transfer_time = 0.0
        with open(time_path, 'r') as f:
            model_transfer_time = float(f.read().strip())
        return model_transfer_time
    
    def get_transfer_time_from_database(self, gpu_name, model_name, batch_size, database_dir):
        time_path = f'{database_dir}/{gpu_name}/{model_name}/data_transfer_time_by_batch_size.csv'
        df = pd.read_csv(time_path)
        row = df[df['Batch_Size'] == batch_size]
        if not row.empty:
            transfer_time = row['Time_In_SECONDS'].values[0]
        else:
            model_path = f'{database_dir}/{gpu_name}/{model_name}/data_transfer_regression_model.pkl'
            model, poly, mse, r2 = joblib.load(model_path)
            new_batch_sizes = np.array([batch_size]).reshape(-1, 1)
            new_batch_sizes_poly = poly.transform(new_batch_sizes)
            predicted_times = model.predict(new_batch_sizes_poly)
            return predicted_times[0]

    async def async_emulate_transfer(self):
        await asyncio.sleep(self.transfer_time)

    def sync_emulate_transfer(self):
        time.sleep(self.transfer_time)

    async def emulate_transfer(self):
        if self.mode == 'async':
            await self.async_emulate_transfer()
        else:
            self.sync_emulate_transfer()


class GPUMemoryEmulator:
    def __init__(self, gpu_cap = 11448, gpu = 'Tesla_M40',
                 model = 'resnet18', batch_size = 256,
                 database_path = '~/gpemu-private/profiled_data/'):
        self.gpu_cap = gpu_cap
        self.gpu = gpu
        self.model = model
        self.batch_size = batch_size
        self.database_path = database_path
        memory_data_path = f'{self.database_path}/memory/{self.gpu}/{self.model}/memory.csv'
        df = pd.read_csv(memory_data_path)
        self.peak_memory = df[df['batch_size'] == self.batch_size]['peak'].values[0]
        self.persistent_memory = df[df['batch_size'] == self.batch_size]['persistent'].values[0]
    
    def get_peak_memory(self):
        return self.peak_memory
    
    def get_persistent_memory(self):
        return self.persistent_memory
    
    def check_memory(self):
        if self.peak_memory > self.gpu_cap:
            raise Exception(f"Peak memory usage exceeds GPU capacity: {self.peak_memory} > {self.gpu_cap}")
        return True
        
    
    
    
    