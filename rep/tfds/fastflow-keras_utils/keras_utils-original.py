# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved
# Functions are copied from tensorflow/python/keras/engine/training.py
# and modified by Samsung research.

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import copy

import tensorflow as tf
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.framework import ops
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.distribute.coordinator import cluster_coordinator
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.python.profiler import trace
from tensorflow.python.eager import context
from tensorflow.python.keras.utils import tf_utils


def _disallow_inside_tf_function(method_name):
    if ops.inside_function():
        # pylint: disable=line-too-long
        error_msg = (
            'Detected a call to `Model.{method_name}` inside a `tf.function`. '
            '`Model.{method_name} is a high-level endpoint that manages its own '
            '`tf.function`. Please move the call to `Model.{method_name}` outside '
            'of all enclosing `tf.function`s. Note that you can call a `Model` '
            'directly on `Tensor`s inside a `tf.function` like: `model(x)`.'
        ).format(method_name=method_name)
        # pylint: enable=line-too-long
        raise RuntimeError(error_msg)


class IteratorCallback:
    def on_train_batch_begin(self, step, iterator):
        pass

    def on_train_batch_end(self, step, iterator):
        pass

    def on_epoch_begin(self, epoch, iterator):
        pass

    def on_epoch_end(self, epoch, iterator):
        pass

    def on_test_batch_begin(self, step, iterator):
        pass

    def on_test_batch_end(self, step, iterator):
        pass


def flatten_metrics_in_order(logs, metrics_names):
    """Turns the `logs` dict into a list as per key order of `metrics_names`."""
    results = []
    for name in metrics_names:
        if name in logs:
            results.append(logs[name])
    for key in sorted(logs.keys()):
        if key not in metrics_names:
            results.append(logs[key])
    if len(results) == 1:
        return results[0]
    return results


def evaluate(self,
             x=None,
             y=None,
             batch_size=None,
             verbose=1,
             sample_weight=None,
             steps=None,
             callbacks=None,
             iterator_callback=IteratorCallback(),
             max_queue_size=10,
             workers=1,
             use_multiprocessing=False,
             return_dict=False,
             **kwargs):
    version_utils.disallow_legacy_graph('Model', 'evaluate')
    self._assert_compile_was_called()
    self._check_call_args('evaluate')
    _disallow_inside_tf_function('evaluate')
    use_cached_eval_dataset = kwargs.pop('_use_cached_eval_dataset', False)
    if kwargs:
        raise TypeError('Invalid keyword arguments: %s' % (kwargs,))

    if self.distribute_strategy._should_use_with_coordinator:  # pylint: disable=protected-access
        self._cluster_coordinator = cluster_coordinator.ClusterCoordinator(
            self.distribute_strategy)

    with self.distribute_strategy.scope():
        # Use cached evaluation data only when it's called in `Model.fit`
        if (use_cached_eval_dataset
                and getattr(self, '_eval_data_handler', None) is not None):
            data_handler = self._eval_data_handler
        else:
            # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
            data_handler = data_adapter.get_data_handler(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps,
                initial_epoch=0,
                epochs=1,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                model=self,
                steps_per_execution=self._steps_per_execution)

        # Container that configures and calls `tf.keras.Callback`s.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                model=self,
                verbose=verbose,
                epochs=1,
                steps=data_handler.inferred_steps)

        logs = {}
        self.test_function = self.make_test_function()
        self._test_counter.assign(0)
        callbacks.on_test_begin()
        for _, iterator in data_handler.enumerate_epochs():  # Single epoch.
            self.reset_metrics()
            with data_handler.catch_stop_iteration():
                for step in data_handler.steps():
                    with trace.Trace('test', step_num=step, _r=1):
                        callbacks.on_test_batch_begin(step)
                        iterator_callback.on_test_batch_begin(step, iterator)
                        tmp_logs = self.test_function(iterator)
                        if data_handler.should_sync:
                            context.async_wait()
                        logs = tmp_logs  # No error, now safe to assign to logs.
                        end_step = step + data_handler.step_increment
                        callbacks.on_test_batch_end(end_step, logs)
                        iterator_callback.on_test_batch_end(end_step, iterator)
        logs = tf_utils.sync_to_numpy_or_python_type(logs)
        callbacks.on_test_end(logs=logs)

        if return_dict:
            return logs
        else:
            return flatten_metrics_in_order(logs, self.metrics_names)


# A copy of keras.model.fit
def fit(self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose='auto',
        callbacks=None,
        iterator_callback=IteratorCallback(),
        validation_split=0.,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False):
    # Legacy graph support is contained in `training_v0.Model`.
    version_utils.disallow_legacy_graph('Model', 'fit')
    self._assert_compile_was_called()
    self._check_call_args('fit')
    _disallow_inside_tf_function('fit')

    if verbose == 'auto':
        if self.distribute_strategy._should_use_with_coordinator:  # pylint: disable=protected-access
            verbose = 2  # Default to epoch-level logging for PSStrategy.
        else:
            verbose = 1  # Default to batch-level logging otherwise.

    if validation_split:
        # Create the validation data using the training data. Only supported for
        # `Tensor` and `NumPy` input.
        (x, y, sample_weight), validation_data = (
            data_adapter.train_validation_split(
                (x, y, sample_weight), validation_split=validation_split))

    if validation_data:
        val_x, val_y, val_sample_weight = (
            data_adapter.unpack_x_y_sample_weight(validation_data))

    if self.distribute_strategy._should_use_with_coordinator:  # pylint: disable=protected-access
        self._cluster_coordinator = cluster_coordinator.ClusterCoordinator(
            self.distribute_strategy)

    with tf.device("/cpu:0"):
        # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
        data_handler = data_adapter.get_data_handler(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            initial_epoch=initial_epoch,
            epochs=epochs,
            shuffle=shuffle,
            class_weight=class_weight,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            model=self,
            steps_per_execution=self._steps_per_execution)

    with self.distribute_strategy.scope(), \
         training_utils.RespectCompiledTrainableState(self):

        # Container that configures and calls `tf.keras.Callback`s.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                model=self,
                verbose=verbose,
                epochs=epochs,
                steps=data_handler.inferred_steps)

        self.stop_training = False
        self.train_function = self.make_train_function()
        self._train_counter.assign(0)
        callbacks.on_train_begin()
        training_logs = None
        # Handle fault-tolerance for multi-worker.
        # TODO(omalleyt): Fix the ordering issues that mean this has to
        # happen after `callbacks.on_train_begin`.
        data_handler._initial_epoch = (  # pylint: disable=protected-access
            self._maybe_load_initial_epoch_from_ckpt(initial_epoch))
        logs = None
        for epoch, iterator in data_handler.enumerate_epochs():
            self.reset_metrics()
            callbacks.on_epoch_begin(epoch)
            iterator_callback.on_epoch_begin(epoch, iterator)
            with data_handler.catch_stop_iteration():
                for step in data_handler.steps():
                    with trace.Trace(
                            'train',
                            epoch_num=epoch,
                            step_num=step,
                            batch_size=batch_size,
                            _r=1):
                        callbacks.on_train_batch_begin(step)
                        iterator_callback.on_train_batch_begin(step, iterator)
                        tmp_logs = self.train_function(iterator)
                        if data_handler.should_sync:
                            context.async_wait()
                        logs = tmp_logs  # No error, now safe to assign to logs.
                        end_step = step + data_handler.step_increment
                        callbacks.on_train_batch_end(end_step, logs)
                        iterator_callback.on_train_batch_end(end_step, iterator)
                        if self.stop_training:
                            break

            logs = tf_utils.sync_to_numpy_or_python_type(logs)
            if logs is None:
                raise ValueError('Expect x to be a non-empty array or dataset.')
            epoch_logs = copy.copy(logs)

            # Run validation.
            if validation_data and self._should_eval(epoch, validation_freq):
                # Create data_handler for evaluation and cache it.
                if getattr(self, '_eval_data_handler', None) is None:
                    self._eval_data_handler = data_adapter.get_data_handler(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        steps_per_epoch=validation_steps,
                        initial_epoch=0,
                        epochs=1,
                        max_queue_size=max_queue_size,
                        workers=workers,
                        use_multiprocessing=use_multiprocessing,
                        model=self,
                        steps_per_execution=self._steps_per_execution)
                val_logs = evaluate(
                    self,
                    x=val_x,
                    y=val_y,
                    sample_weight=val_sample_weight,
                    batch_size=validation_batch_size or batch_size,
                    steps=validation_steps,
                    callbacks=callbacks,
                    max_queue_size=max_queue_size,
                    workers=workers,
                    use_multiprocessing=use_multiprocessing,
                    return_dict=True,
                    _use_cached_eval_dataset=True)
                val_logs = {
                    'val_' + name: val for name, val in val_logs.items()
                }
                epoch_logs.update(val_logs)

            callbacks.on_epoch_end(epoch, epoch_logs)
            iterator_callback.on_epoch_end(epoch, iterator)
            training_logs = epoch_logs
            if self.stop_training:
                break

        # If eval data_hanlder exists, delete it after all epochs are done.
        if getattr(self, '_eval_data_handler', None) is not None:
            del self._eval_data_handler
        callbacks.on_train_end(logs=training_logs)
        return self.history
