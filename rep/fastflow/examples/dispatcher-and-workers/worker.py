import tensorflow as tf
import sys

w_port = int(sys.argv[1])
print(w_port)

w_config = tf.data.experimental.service.WorkerConfig(
    dispatcher_address="localhost:5000",
    worker_address="10.140.82.252" + ":" + str(w_port),
    port=w_port)
worker = tf.data.experimental.service.WorkerServer(w_config)

worker.join()