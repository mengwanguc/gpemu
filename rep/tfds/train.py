import tensorflow as tf
import os
import time

# Step 1: Decode and preprocess images
def decode_and_preprocess_image(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Step 2: Create dataset from file paths and labels
def create_local_dataset(image_dir, batch_size=64):
    class_names = sorted(os.listdir(image_dir))
    file_paths = []
    labels = []

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(image_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.endswith(".JPEG"):
                file_paths.append(os.path.join(class_dir, fname))
                labels.append(label)

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(decode_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset, len(class_names)

# Step 3: Build AlexNet model
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

# Step 4: Train model locally
def train_local(image_dir, epochs=10, batch_size=64):
    dataset, num_classes = create_local_dataset(image_dir, batch_size)
    model = build_alexnet(input_shape=(224, 224, 3), num_classes=num_classes)
    optimizer = tf.keras.optimizers.Adam()

    for epoch in range(epochs):
        epoch_time = 0
        start_time = time.time()
        for step, (images, labels) in enumerate(dataset):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
                loss = tf.reduce_mean(loss)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            end_time = time.time()
            print(f"Epoch {epoch + 1}, Step {step + 1}, Loss: {loss.numpy():.4f}  Time: {end_time-start_time}")
            epoch_time += end_time - start_time
            start_time = time.time()
        print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}  Time: {epoch_time}")

if __name__ == "__main__":
    image_dir = "/home/wangm12/data/imagenette2/train"  # Update with your ImageNette directory
    train_local(image_dir, epochs=10, batch_size=64)