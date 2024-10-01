import tensorflow as tf

d_config = tf.data.experimental.service.DispatcherConfig(port=5000)
dispatcher = tf.data.experimental.service.DispatchServer(d_config)

w_port = 5001
w_config = tf.data.experimental.service.WorkerConfig(
    dispatcher_address=dispatcher.target.split("://")[1],
    worker_address="10.140.83.149" + ":" + str(w_port),
    port=w_port)
worker = tf.data.experimental.service.WorkerServer(w_config)

dispatcher.join()