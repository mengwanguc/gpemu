# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved
import argparse
import subprocess

import os
# print("Cleaning cache...")
# os.system("sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")

# using only CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from enum import Enum

import tensorflow as tf
import yaml

from tensorflow import keras
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import fastflow as ff
from fastflow import utils as ff_utils


class OffloadingType(Enum):
    FASTFLOW = "ff"  # fastflow
    TF_DSR = "tf-dsr"  # tf-dsr
    TF_DSR_ALL = "tf-dsr-all"  # tf-dsr
    TF_DSLR = "tf-dslr"  # tf-dslr
    TF_DSLR_ALL = "tf-dslr-all"  # tf-dslr
    DALI = "dali"  # dali
    TENSORFLOW = "tf"  # tensorflow


class GPUType(Enum):
    SINGLE = "single"  # single-gpu training
    MULTI = "multi"  # multi-gpu training


class App:
    """
    Interface for Applications.
    Applications must inherit this class.
    Ex) class CtcAsr(App):
    """

    def __init__(self, args, config):
        """
        :param args:  command line arguments
        :param config:  yaml config
        """
        self.args = args
        self.config = config

    def create_model(self):
        """
        Create a model.
        """
        raise NotImplementedError

    def create_dataset(self, num_parallel):
        """
        Create train dataset input pipeline
        :param num_parallel: number of parallelism
        """
        raise NotImplementedError

    def create_valid_dataset(self, num_parallel):
        """
        Create valid dataset input pipeline
        :param num_parallel:  number of parallelism
        """
        raise NotImplementedError

    def create_manual_offloaded_dataset(self, num_parallel):
        """
        Create the train dataset input pipeline with manual offloading.
        :param num_parallel: number of parallelism
        """
        raise NotImplementedError

    def create_manual_offloaded_valid_dataset(self, num_parallel):
        """
        Create the validation dataset input pipeline with manual offloading.
        :param num_parallel: number of parallelism
        """
        raise NotImplementedError

    def create_all_offload_dataset(self, num_parallel):
        """
        Create the train dataset input pipeline with manual offloading all operations.
        :param num_parallel: number of parallelism
        """
        raise NotImplementedError

    def create_all_offload_valid_dataset(self, num_parallel):
        """
        Create the valid dataset input pipeline with manual offloading all operations.
        :param num_parallel: number of parallelism
        """
        raise NotImplementedError

    def create_dali_dataset(self, num_parallel):
        """
        Create dali train dataset input pipeline
        :param num_parallel: number of parallelism
        """
        raise NotImplementedError

    def create_dali_valid_dataset(self, num_parallel):
        """
        Create dali valid dataset input pipeline
        :param num_parallel:  number of parallelism
        """
        raise NotImplementedError

    def callbacks(self):
        """
        Return callbacks for model.fit()
        :return:
        """
        return []

    def steps_per_epoch_for_dali(self):
        """
        Return steps_per_epoch used in model.fit() for DALI
        :return: steps_per_epoch
        """
        raise NotImplementedError


class AppRunner:
    def __init__(self, app, args, config):
        self.app = app
        self.args = args
        self.config = config

    def run(self):

        if self.args.parallel > 0:
            parallel = self.args.parallel
        else:
            parallel = tf.data.AUTOTUNE

        # Dataset setting
        if self.args.offloading_type in (OffloadingType.TF_DSR,
                                         OffloadingType.TF_DSLR):
            self.config.partial_offload_enabled = False
            dataset = self.app.create_manual_offloaded_dataset(parallel)
            valid_dataset = self.app.create_manual_offloaded_valid_dataset(
                parallel
            )
        elif self.args.offloading_type in (OffloadingType.TF_DSR_ALL,
                                           OffloadingType.TF_DSLR_ALL):
            self.config.partial_offload_enabled = False
            dataset = self.app.create_all_offload_dataset(parallel)
            valid_dataset = self.app.create_all_offload_valid_dataset(
                parallel
            )
        elif self.args.offloading_type is OffloadingType.FASTFLOW:
            self.config.partial_offload_enabled = True
            dataset = self.app.create_dataset(parallel)
            valid_dataset = self.app.create_valid_dataset(parallel)
        elif self.args.offloading_type is OffloadingType.TENSORFLOW:
            dataset = self.app.create_dataset(parallel)
            valid_dataset = self.app.create_valid_dataset(parallel)
        elif self.args.offloading_type is OffloadingType.DALI:
            dataset = self.app.create_dali_dataset(parallel)
            valid_dataset = self.app.create_dali_valid_dataset(parallel)
        else:
            raise RuntimeError("Invalid offloading type: "
                               + self.args.offloading_type)

        # Model setting
        if self.args.gpu_type is GPUType.SINGLE:
            model = self.app.create_model()
            print(model)
        else:
            # Create a MirroredStrategy for multi-gpu training.
            strategy = tf.distribute.MultiWorkerMirroredStrategy()
            print("Number of devices: {}".format(
                strategy.num_replicas_in_sync
            ))
            with strategy.scope():
                model = self.app.create_model()

        # Model running
        if self.args.offloading_type is OffloadingType.FASTFLOW:
            # Run fastflow model for auto offloading
            model.fit(x=dataset,
                      validation_data=valid_dataset,
                      epochs=self.args.epochs,
                      callbacks=self.app.callbacks(),
                      auto_offload_conf=self.config)
        elif self.args.offloading_type is OffloadingType.DALI:
            # Run model with GPU offloading with DALI
            steps_per_epoch = self.app.steps_per_epoch_for_dali()
            keras.models.Model.fit(model,
                                   x=dataset,
                                   # validation_data=valid_dataset,
                                   epochs=self.args.epochs,
                                   callbacks=self.app.callbacks()
                                             + [ff_utils.EpochTimeCallback()],
                                   steps_per_epoch=steps_per_epoch)
        else:
            print("else offloading..")
            # Run the original model not to perform auto-offloading.
            print("dataset ->", dataset)
            print("model ->", model)
            # print("callbacks ->", callbacks)
            # quit()
            print("callback ->", self.app.callbacks() + [ff_utils.EpochTimeCallback()])
            print("callback ->", self.app.callbacks())
            print(ff_utils.EpochTimeCallback())
            # quit()
        
            keras.models.Model.fit(model,
                                   x=dataset,
                                   validation_data=valid_dataset,
                                   epochs=self.args.epochs,
                                   callbacks=self.app.callbacks()
                                             + [ff_utils.EpochTimeCallback()])


def get_arguments():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument("app_file_path", type=str)
    parser.add_argument("data_prefix", type=str)
    parser.add_argument("offloading_type", type=OffloadingType,
                        choices=[OffloadingType.TENSORFLOW,
                                 OffloadingType.TF_DSR,
                                 OffloadingType.TF_DSR_ALL,
                                 OffloadingType.TF_DSLR,
                                 OffloadingType.TF_DSLR_ALL,
                                 OffloadingType.FASTFLOW,
                                 OffloadingType.DALI])
    parser.add_argument("yaml_path", type=str)
    parser.add_argument("--gpu_type", type=GPUType, default=GPUType.SINGLE,
                        choices=[GPUType.SINGLE, GPUType.MULTI])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--parallel", type=int, default=-1)
    parser.add_argument("--num_local_workers", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    yaml_dict = yaml.load(open(args.yaml_path), Loader=yaml.FullLoader)
    return args, yaml_dict


def get_address():
    return subprocess.check_output(["hostname", "-I"]). \
        strip().decode("utf-8").split()[0]


def launch_local_worker(num_local_workers, dispatcher_addr, worker_base_port):
    print("Launch local worker")
    num_workers = num_local_workers
    workers = []
    for i in range(num_workers):
        port = worker_base_port + i + 1
        w_config = tf.data.experimental.service.WorkerConfig(
            dispatcher_address=dispatcher_addr + ":5000",
            worker_address=get_address() + ":" + str(port),
            port=port)

        worker = tf.data.experimental.service.WorkerServer(w_config)
        workers.append(worker)
    return workers


if __name__ == "__main__":
    """
     Arguments
     1) app_file_path: the example app path 
     2) offloading_type: (tf, tf-dsr, tf-dslr, ff)
     3) yaml_path: the yaml config file path fo the example app
     4) gpu_type: ('single', 'multi')
    """
    import time
    start = time.time()
    args, yaml_dict = get_arguments()
    yaml_path = args.yaml_path
    app_path = args.app_file_path

    print('Args: ', args)

    # Extract app
    app_module = __import__(app_path.replace('.py', ''))
    subclasses = app_module.App.__subclasses__()

    if len(subclasses) > 1 or len(subclasses) == 0:
        raise RuntimeError("Subclasses of ExampleApp must be 1, but "
                           + str(len(subclasses)) + str(subclasses))

    config = ff.FastFlowConfig.from_yaml(yaml_path)

    # Launch local dispatcher and worker for FastFlow
    if args.offloading_type is OffloadingType.FASTFLOW:
        d_config = tf.data.experimental.service.DispatcherConfig(port=5000)
        dispatcher = tf.data.experimental.service.DispatchServer(d_config)
        workers_local_dispatcher = launch_local_worker(args.num_local_workers,
                                                       get_address(), 5000)

    if args.offloading_type in (OffloadingType.TF_DSLR,
                                OffloadingType.TF_DSLR_ALL,
                                OffloadingType.FASTFLOW) \
            and not config.autoscale_enabled:
        workers = launch_local_worker(args.num_local_workers,
                                      config.dispatcher_addr, 5500)

    if args.offloading_type in (OffloadingType.TF_DSR,
                                OffloadingType.TF_DSLR,
                                OffloadingType.TF_DSR_ALL,
                                OffloadingType.TF_DSLR_ALL,
                                OffloadingType.FASTFLOW) \
            and not config.dispatcher_addr:
        raise RuntimeError("Offloading Type is " + str(args.offloading_type) +
                           " , but dispatcher_addr "
                           "is not configured in the yaml.")

    if args.offloading_type is OffloadingType.DALI:
        config_gpu_mem = ConfigProto()
        config_gpu_mem.gpu_options.per_process_gpu_memory_fraction = 0.8
        session = InteractiveSession(config=config_gpu_mem)

    app = subclasses[0](args, config)

    runner = AppRunner(app, args, config)
    runner.run()

    end = time.time() - start
    print("total time ->", end)

