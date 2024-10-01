import tensorflow as tf
import time

d_config = tf.data.experimental.service.DispatcherConfig(port=5000)
dispatcher = tf.data.experimental.service.DispatchServer(d_config)

print(dispatcher.target.split("://")[1])

dispatcher.join()

# # Keep the script running
# try:
#     while True:
#         time.sleep(1)
# except KeyboardInterrupt:
#     print("Dispatcher shutting down.")