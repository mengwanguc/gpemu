from gpemu import GPUResourceManagerKafka as GPUResourceManager

def main():
    # Initialize the GPU resource manager
    manager = GPUResourceManager(kafka_server='localhost:9092')

    try:
        # Start the manager to listen for incoming requests
        manager.start()
    except KeyboardInterrupt:
        # Stop the manager gracefully
        manager.stop()

if __name__ == "__main__":
    main()
