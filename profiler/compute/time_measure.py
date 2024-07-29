import time
import torch


# Class to measure and compute average GPU time
class GPUTimeMeasurement:

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_time = 0.0
        self.batch_count = 0
        self.last_start_time = 0.0
        self.last_end_time = 0.0
        self.average_time = 0.0
        self.adjusted_average_time = 0.0
        self.time_records = []

    def start(self):
        torch.cuda.synchronize()
        self.last_start_time = time.time()

    def end(self):
        torch.cuda.synchronize()
        self.last_end_time = time.time()
        elapsed_time =  self.last_end_time - self.last_start_time
        self.total_time += elapsed_time
        self.time_records.append(elapsed_time)
        self.batch_count += 1

    def get_average_time(self):
        if self.batch_count == 0:
            self.average_time = 0.0
        else:
            self.average_time = self.total_time / self.batch_count
        return self.average_time

    # The time measurement is usually longer than normal for the first 1-2 batches.
    # therefore, we want to remove such outliers
    def get_adjusted_average_time(self, n_largest_to_remove=2, n_smallest_to_remove=0):
        sorted_records = sorted(self.time_records)
        n_largest_to_remove = min(len(sorted_records), n_largest_to_remove)
        sorted_records = sorted_records[0:-n_largest_to_remove]
        n_smallest_to_remove = min(len(sorted_records), n_smallest_to_remove)
        sorted_records = sorted_records[n_smallest_to_remove:]
        adjusted_average_time = sum(sorted_records) / len(sorted_records)
        return adjusted_average_time
    
    def get_records(self):
        return self.time_records
    
    
    

    

