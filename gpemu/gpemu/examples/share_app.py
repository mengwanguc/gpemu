from gpemu import GPUSharingEmulatorKafka as GPUSharingEmulator

def main():
    # Initialize the GPU sharing emulator
    emulator = GPUSharingEmulator(
        kafka_server='localhost:9092',
        gpu='Tesla_M40',
        model='resnet18',
        batch_size=256,
        forward_time=0.05,
        backward_time=0.05
    )

    # Simulate processing multiple batches
    num_batches = 10
    for batch_id in range(num_batches):
        print(f"Requesting GPU resources for batch {batch_id}...")
        emulator.sync_train_batch_with_sharing(batch_id)
        print(f"Completed processing batch {batch_id}.\n")

if __name__ == "__main__":
    main()
