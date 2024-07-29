import unittest
import asyncio
from gpemu import ComputeTimeEmulator, TransferTimeEmulator

class TestComputeTimeEmulator(unittest.TestCase):
    def test_sync_training(self):
        emulator = ComputeTimeEmulator(epochs=1, batches_per_epoch=2, forward_time=0.1, backward_time=0.1, mode='sync')
        for _ in range(emulator.epochs):
            for _ in range(emulator.batches_per_epoch):
                emulator.emulate_compute()
        self.assertTrue(True)

    def test_async_training(self):
        async def run_training():
            emulator = ComputeTimeEmulator(epochs=1, batches_per_epoch=2, forward_time=0.1, backward_time=0.1, mode='async')
            for _ in range(emulator.epochs):
                for _ in range(emulator.batches_per_epoch):
                    await emulator.emulate_compute()
            self.assertTrue(True)

        asyncio.run(run_training())

class TestTransferTimeEmulator(unittest.TestCase):
    def test_sync_transfer(self):
        emulator = TransferTimeEmulator(transfer_time=0.1, mode='sync')
        emulator.emulate_transfer()
        self.assertTrue(True)

    def test_async_transfer(self):
        async def run_transfer():
            emulator = TransferTimeEmulator(transfer_time=0.1, mode='async')
            await emulator.emulate_transfer()
            self.assertTrue(True)

        asyncio.run(run_transfer())

class TestAsyncEmulation(unittest.TestCase):
    def test_training_with_dependencies(self):
        async def run_emulation():
            transfer_emulator = TransferTimeEmulator(transfer_time=0.02, mode='async')
            compute_emulator = ComputeTimeEmulator(epochs=1, batches_per_epoch=2, forward_time=0.1, backward_time=0.1, mode='async')

            previous_compute_task = None

            for batch in range(compute_emulator.batches_per_epoch):
                transfer_task = asyncio.create_task(transfer_emulator.emulate_transfer())

                if previous_compute_task is not None:
                    await previous_compute_task

                await transfer_task

                previous_compute_task = asyncio.create_task(compute_emulator.emulate_compute())
                await previous_compute_task

            if previous_compute_task:
                await previous_compute_task

            self.assertTrue(True)

        asyncio.run(run_emulation())

    def test_sync_emulation(self):
        transfer_emulator = TransferTimeEmulator(transfer_time=0.02, mode='sync')
        compute_emulator = ComputeTimeEmulator(epochs=1, batches_per_epoch=2, forward_time=0.1, backward_time=0.1, mode='sync')

        for epoch in range(compute_emulator.epochs):
            for batch in range(compute_emulator.batches_per_epoch):
                transfer_emulator.emulate_transfer()
                compute_emulator.emulate_compute()
                
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
