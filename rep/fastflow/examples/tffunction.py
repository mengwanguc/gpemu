import tensorflow as tf
import time

# Define a Python function.
def a_regular_function(x, y, b):
  x = tf.matmul(x, y)
  x = x + b
  return x

def my_regular_function():
    for i in range(10):
        time.sleep(1)
    # time.sleep(5)
    return i


# The Python type of `a_function_that_uses_a_graph` will now be a
# `PolymorphicFunction`.
# a_function_that_uses_a_graph = tf.function(a_regular_function)
my_regular_function_graph = tf.function(my_regular_function)

# # Make some tensors.
# x1 = tf.constant([[1.0, 2.0]])
# y1 = tf.constant([[2.0], [3.0]])
# b1 = tf.constant(4.0)

# orig_value = a_regular_function(x1, y1, b1).numpy()
# # Call a `tf.function` like a Python function.
# tf_function_value = a_function_that_uses_a_graph(x1, y1, b1).numpy()

start = time.time()
orig_value = my_regular_function()
end = time.time()
print('original function time: {}'.format(end-start))

start = time.time()
tf_function_value = my_regular_function_graph()
end = time.time()
print('tf function time: {}'.format(end-start))

print('original value: {}'.format(orig_value))
print('tf_function_value: {}'.format(tf_function_value))