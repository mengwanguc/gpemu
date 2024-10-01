import tensorflow as tf
import time

# Step 1: Build a simple AlexNet model
def build_alexnet(input_shape=(224, 224, 3), num_classes=2):
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

# Step 2: Create a synthetic dataset
def create_synthetic_dataset(batch_size=32, input_shape=(224, 224, 3), num_classes=2):
    inputs = tf.random.uniform((batch_size, *input_shape), minval=0, maxval=1, dtype=tf.float32)
    labels = tf.random.uniform((batch_size,), minval=0, maxval=num_classes, dtype=tf.int32)
    return inputs, labels

# Step 3: Train step
@tf.function
def train_step(model, optimizer, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
        loss = tf.reduce_mean(loss)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

if __name__ == "__main__":
    # Set GPU configuration to use only one GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')  # Use only GPU 0

    # Initialize model, optimizer, and synthetic dataset
    batch_size = 32
    input_shape = (224, 224, 3)
    num_classes = 2
    model = build_alexnet(input_shape=input_shape, num_classes=num_classes)
    optimizer = tf.keras.optimizers.Adam()
    inputs, labels = create_synthetic_dataset(batch_size=batch_size, input_shape=input_shape, num_classes=num_classes)

    # Measure GPU time for training steps
    gpu_times = []
    for step in range(20):  # Run for 20 iterations
        start_time = time.time()  # Start timing
        loss = train_step(model, optimizer, inputs, labels)
        
        # Explicitly synchronize GPU
        tf.raw_ops.EmptyTensorList(
            element_dtype=tf.float32,
            element_shape=tf.constant([], dtype=tf.int32),  # Corrected element_shape
            max_num_elements=0
        )

        end_time = time.time()  # End timing

        gpu_time = (end_time - start_time) * 1000  # Convert to milliseconds
        gpu_times.append(gpu_time)
        print(f"Step {step + 1}, GPU Time: {gpu_time:.2f} ms")

    # Calculate average GPU time
    avg_gpu_time = sum(gpu_times) / len(gpu_times)
    print(f"Average GPU Compute Time: {avg_gpu_time:.2f} ms")
