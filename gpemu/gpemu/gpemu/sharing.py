import json
import uuid
from kafka import KafkaProducer, KafkaConsumer
from .emulators import ComputeTimeEmulator
import threading
import time
import pika

class GPUSharingEmulatorKafka:
    def __init__(self, kafka_server='localhost:9092', 
                 gpu='Tesla_M40', model='resnet18', 
                 batch_size=256, forward_time=0.05, backward_time=0.05, 
                 from_database=False, database_path='/home/cc/gpemu/profiled_data'):
        self.app_id = str(uuid.uuid4())
        self.request_topic = 'gpu_requests'
        self.response_topic = f'response_{self.app_id}'
        self.producer = KafkaProducer(bootstrap_servers=kafka_server)
        self.consumer = KafkaConsumer(self.response_topic, bootstrap_servers=kafka_server,
                                      auto_offset_reset='earliest', enable_auto_commit=True)
        self.emulator = ComputeTimeEmulator(gpu=gpu, model=model, batch_size=batch_size, 
                                            forward_time=forward_time, backward_time=backward_time, 
                                            from_databse=from_database, database_path=database_path, mode='sync')

    def request_gpu_time(self, batch_id):
        compute_time = self.emulator.get_forward_time() + self.emulator.get_backward_time()
        message = json.dumps({
            'app_id': self.app_id,
            'batch_id': batch_id,
            'compute_time': compute_time,
            'response_topic': self.response_topic
        })
        self.producer.send(self.request_topic, message.encode('utf-8'))

        # Block until a response is received from the resource manager
        for message in self.consumer:
            break

    def sync_train_batch_with_sharing(self, batch_id):
        self.request_gpu_time(batch_id)
        # The GPU compute time has already been handled by the manager

    def run(self, num_batches):
        for batch_id in range(num_batches):
            self.sync_train_batch_with_sharing(batch_id)

class GPUSharingEmulatorRabbitMQ:
    def __init__(self, rabbitmq_server='localhost', request_queue='gpu_requests', 
                 gpu='Tesla_M40', model='resnet18', 
                 batch_size=256, forward_time=0.05, backward_time=0.05,
                 from_database=False, database_path='/home/cc/gpemu/profiled_data'):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_server))
        self.channel = self.connection.channel()
        self.request_queue = request_queue
        self.response_queue = self.channel.queue_declare(queue='', exclusive=True).method.queue
        self.channel.queue_declare(queue=self.request_queue)
        self.app_id = str(uuid.uuid4())
        self.emulator = ComputeTimeEmulator(gpu=gpu, model=model, 
                                            batch_size=batch_size, forward_time=forward_time, backward_time=backward_time, 
                                            mode='sync',
                                            from_database=from_database, database_path=database_path)

    def request_gpu_time(self, batch_id):
        compute_time = self.emulator.get_forward_time() + self.emulator.get_backward_time()
        message = {
            'app_id': self.app_id,
            'batch_id': batch_id,
            'compute_time': compute_time,
            'response_queue': self.response_queue
        }
        self.channel.basic_publish(exchange='', routing_key=self.request_queue, body=json.dumps(message))

        # Wait for a response from the GPU resource manager
        for method, properties, body in self.channel.consume(self.response_queue, inactivity_timeout=None):
            if body:
                print(body.decode())
                self.channel.basic_ack(method.delivery_tag)
                break

    def sync_train_batch_with_sharing(self, batch_id):
        self.request_gpu_time(batch_id)
        # The GPU compute time has already been handled by the manager

    def run(self, num_batches):
        for batch_id in range(num_batches):
            self.sync_train_batch_with_sharing(batch_id)

class GPUSharingEmulator:
    def __init__(self, server='localhost:9092',
                 gpu='Tesla_M40', model='resnet18', 
                 batch_size=256, forward_time=0.05, backward_time=0.05, 
                 from_database=False, database_path='/home/cc/gpemu/profiled_data',
                 engine='kafka'):
        if engine == 'kafka':
            self.emulator = GPUSharingEmulatorKafka(
                kafka_server=server, gpu=gpu, model=model, batch_size=batch_size, 
                forward_time=forward_time, backward_time=backward_time, 
                from_database=from_database, database_path=database_path)
        else:
            self.emulator = GPUSharingEmulatorRabbitMQ(
                rabbitmq_server=server, gpu=gpu, model=model, batch_size=batch_size, 
                forward_time=forward_time, backward_time=backward_time, 
                from_database=from_database, database_path=database_path)
        
        
    def run(self, num_batches):
        self.emulator.run(num_batches)
    
    def sync_train_batch_with_sharing(self, batch_id):
        self.emulator.sync_train_batch_with_sharing(batch_id)
    
    def request_gpu_time(self, batch_id):
        self.emulator.request_gpu_time(batch_id)



    
    