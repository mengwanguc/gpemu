from gpemu import ComputeTimeEmulator, TransferTimeEmulator, GPUMemoryEmulator
import asyncio

async def async_main():
    transfer_emulator = TransferTimeEmulator(transfer_time=1, mode='async')
    compute_emulator = ComputeTimeEmulator(forward_time=1, backward_time=1, mode='async')
    gpu_memory_emulator = GPUMemoryEmulator()

    for epoch in range(1):
        previous_compute_task = None
        for batch in range(5):
            transfer_task = asyncio.create_task(transfer_emulator.emulate_transfer())
            
            if previous_compute_task is not None:
                await previous_compute_task
            await transfer_task
            previous_compute_task = asyncio.create_task(compute_emulator.emulate_compute())
            await previous_compute_task
        if previous_compute_task:
            await previous_compute_task
        print(f"Epoch {epoch + 1} completed")

asyncio.run(async_main())
