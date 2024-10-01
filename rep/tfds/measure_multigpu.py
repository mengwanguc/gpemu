import tensorflow as tf
import time

# Step 1: Build AlexNet model
def build_alexnet(input_shape=(224, 224, 3), num_classes=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(96, kernel_size=11, strides=4, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),
        tf.keras.layers.Conv2D(256, kernel_size=5, padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),
        tf.keras.layers.Conv2D(384, kernel_size=3, padding="same", activation='relu'),
        tf.keras.layers.Conv2D(384, kernel_size=3, padding="same", activation='relu'),
        tf.keras.layers.Conv2D(256, kernel_size=3, padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])
    return model

# Step 2: Create synthetic dataset
def create_synthetic_dataset(batch_size=64, input_shape=(224, 224, 3), num_classes=10):
    inputs = tf.random.uniform((batch_size, *input_shape), minval=0, maxval=1, dtype=tf.float32)
    labels = tf.random.uniform((batch_size,), minval=0, maxval=num_classes, dtype=tf.int32)
    return inputs, labels

# Step 3: Multi-GPU Training
def train_multi_gpu_with_synthetic_data(epochs=10, batch_size_per_gpu=64):
    strategy = tf.distribute.MirroredStrategy()

    # Scale the batch size by the number of GPUs
    global_batch_size = batch_size_per_gpu * strategy.num_replicas_in_sync
    print(f"Using {strategy.num_replicas_in_sync} GPUs with global batch size: {global_batch_size}")

    with strategy.scope():
        # Build the model and optimizer
        model = build_alexnet(input_shape=(224, 224, 3), num_classes=10)
        optimizer = tf.keras.optimizers.Adam()
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )

        # Define a training step
        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                per_example_loss = loss_object(labels, predictions)
                loss = tf.reduce_sum(per_example_loss) / global_batch_size  # Scale by global batch size
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        @tf.function
        def distributed_train_step(images, labels):
            per_replica_losses = strategy.run(train_step, args=(images, labels))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        # Training loop with synthetic data
        for epoch in range(epochs):
            epoch_time = 0
            inputs, labels = create_synthetic_dataset(batch_size=global_batch_size)

            start_time = time.time()
            for step in range(100):  # Simulate 100 steps per epoch
                step_start = time.time()
                loss = distributed_train_step(inputs, labels)
                step_end = time.time()
                print(f"Epoch {epoch + 1}, Step {step + 1}, Loss: {loss.numpy():.4f}, Step Time: {step_end - step_start:.4f} seconds")
                epoch_time += step_end - step_start

            print(f"Epoch {epoch + 1} took {epoch_time:.4f} seconds")

if __name__ == "__main__":
    train_multi_gpu_with_synthetic_data(epochs=10, batch_size_per_gpu=64)
