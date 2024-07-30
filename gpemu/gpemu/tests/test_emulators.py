import unittest
import asyncio
import time
from gpemu import ComputeTimeEmulator, TransferTimeEmulator, GPUMemoryEmulator, GPUPreprocessingEmulator

class TestComputeTimeEmulator(unittest.TestCase):

    def test_sync_forward_backward(self):
        emulator = ComputeTimeEmulator(forward_time=1, backward_time=1, mode='sync')
        start_time = time.time()
        emulator.sync_train_batch()
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.assertGreater(elapsed_time, 2)
        self.assertLess(elapsed_time, 3)

    def test_async_forward_backward(self):
        async def run_emulation():
            emulator = ComputeTimeEmulator(forward_time=1, backward_time=1, mode='async')
            start_time = time.time()
            await emulator.emulate_compute()
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.assertGreater(elapsed_time, 2)
            self.assertLess(elapsed_time, 3)
        asyncio.run(run_emulation())

    def test_forward_backward_from_database(self):
        emulator = ComputeTimeEmulator(gpu='Tesla_M40', model='resnet18', batch_size=256, from_databse=True)
        forward_time = emulator.get_forward_time()
        backward_time = emulator.get_backward_time()
        self.assertGreater(forward_time, 0)
        self.assertGreater(backward_time, 0)

import unittest
import time
from gpemu import TransferTimeEmulator

class TestTransferTimeEmulator(unittest.TestCase):

    def test_sync_transfer(self):
        emulator = TransferTimeEmulator(transfer_time=1, mode='sync')
        start_time = time.time()
        emulator.emulate_transfer()  # This should call sync_emulate_transfer()
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.assertGreater(elapsed_time, 1)
        self.assertLess(elapsed_time, 2)

    def test_async_transfer(self):
        async def run_emulation():
            emulator = TransferTimeEmulator(transfer_time=1, mode='async')
            start_time = time.time()
            await emulator.emulate_transfer()  # This should call async_emulate_transfer()
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.assertGreater(elapsed_time, 1)
            self.assertLess(elapsed_time, 2)
        asyncio.run(run_emulation())

    def test_transfer_from_database(self):
        emulator = TransferTimeEmulator(gpu='Tesla_M40', model='resnet18', batch_size=256, from_databse=True)
        transfer_time = emulator.get_transfer_time()
        model_transfer_time = emulator.get_model_transfer_time()
        self.assertGreater(transfer_time, 0)
        self.assertGreater(model_transfer_time, 0)

class TestGPUMemoryEmulator(unittest.TestCase):

    def test_memory_usage(self):
        emulator = GPUMemoryEmulator(gpu='Tesla_M40', model='resnet18', batch_size=256)
        peak_memory = emulator.get_peak_memory()
        persistent_memory = emulator.get_persistent_memory()
        self.assertGreater(peak_memory, 0)
        self.assertGreater(persistent_memory, 0)
    
    def test_memory_exceeds_capacity(self):
        emulator = GPUMemoryEmulator(gpu_cap=1000, gpu='Tesla_M40', model='resnet18', batch_size=256)
        with self.assertRaises(Exception) as context:
            emulator.check_memory()
        self.assertIn('Peak memory usage exceeds GPU capacity', str(context.exception))

        
class TestAsyncTraining(unittest.TestCase):

    def test_async_training(self):
        asyncio.run(self.run_async_training())

    async def run_async_training(self):
        transfer_emulator = TransferTimeEmulator(transfer_time=1, mode='async')
        compute_emulator = ComputeTimeEmulator(forward_time=1, backward_time=1, mode='async')
        gpu_memory_emulator = GPUMemoryEmulator()

        start_time = time.time()
        for epoch in range(1):
            previous_compute_task = None
            for batch in range(5):
                transfer_task = asyncio.create_task(transfer_emulator.emulate_transfer())
                
                if previous_compute_task is not None:
                    await previous_compute_task
                await transfer_task
                previous_compute_task = asyncio.create_task(compute_emulator.emulate_compute())
            if previous_compute_task:
                await previous_compute_task
        end_time = time.time()
        self.assertGreater(end_time - start_time, 11)
        self.assertLess(end_time - start_time, 12)

class TestSyncTraining(unittest.TestCase):

    def test_sync_training(self):
        self.run_sync_training()

    def run_sync_training(self):
        transfer_emulator = TransferTimeEmulator(transfer_time=1, mode='sync')
        compute_emulator = ComputeTimeEmulator(forward_time=1, backward_time=1, mode='sync')
        gpu_memory_emulator = GPUMemoryEmulator()

        start_time = time.time()
        for epoch in range(1):
            for batch in range(5):
                transfer_emulator.emulate_transfer()
                compute_emulator.emulate_compute()
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        self.assertGreater(elapsed_time, 15)
        self.assertLess(elapsed_time, 16)

class TestPreprocessingEmulator(unittest.TestCase):

    def test_sync_preprocessing(self):
        emulator = GPUPreprocessingEmulator(gpu_wait_time=1, cpu_usage=300, cpu_cores=6, cpu_processing_time=1, mode='sync')
        start_time = time.time()
        emulator.emulate_preprocessing()
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.assertGreater(elapsed_time, 1)
        self.assertLess(elapsed_time, 2)

    def test_async_preprocessing(self):
        async def run_emulation():
            emulator = GPUPreprocessingEmulator(gpu_wait_time=1, cpu_usage=300, cpu_cores=6, cpu_processing_time=1, mode='async')
            start_time = time.time()
            await emulator.emulate_preprocessing()
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.assertGreater(elapsed_time, 1)
            self.assertLess(elapsed_time, 2)
        asyncio.run(run_emulation())


if __name__ == '__main__':
    unittest.main()
