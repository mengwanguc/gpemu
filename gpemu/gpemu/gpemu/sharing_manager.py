import json
import uuid
from kafka import KafkaProducer, KafkaConsumer
import threading
import time
import pika

class GPUResourceManagerKafka:
    def __init__(self, kafka_server='localhost:9092', request_topic='gpu_requests'):
        self.request_topic = request_topic
        self.consumer = KafkaConsumer(self.request_topic, bootstrap_servers=kafka_server,
                                      auto_offset_reset='earliest', enable_auto_commit=True)
        self.producer = KafkaProducer(bootstrap_servers=kafka_server)
        self.running = False

    def process_message(self, message):
        message_data = json.loads(message.value.decode())
        response_topic = message_data['response_topic']
        print(f"Processing batch {message_data['batch_id']} from App {message_data['app_id']}")
        time.sleep(message_data['compute_time'])  # Simulate the GPU compute time

        response = f"Processed batch {message_data['batch_id']} from App {message_data['app_id']}"
        self.producer.send(response_topic, response.encode('utf-8'))

    def start(self):
        self.running = True
        print("GPUResourceManager started.")
        for message in self.consumer:
            if not self.running:
                break
            self.process_message(message)
        print("GPUResourceManager stopped.")

    def stop(self):
        self.running = False
    
    def run_in_thread(self):
        thread = threading.Thread(target=self.start)
        thread.start()
        return thread


class GPUResourceManagerRabbitMQ:
    def __init__(self, rabbitmq_server='localhost', request_queue='gpu_requests'):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_server))
        self.channel = self.connection.channel()
        self.request_queue = request_queue
        self.channel.queue_declare(queue=self.request_queue)

    def process_request(self, ch, method, properties, body):
        message_data = json.loads(body)
        response_queue = message_data['response_queue']
        print(f"Processing batch {message_data['batch_id']} from App {message_data['app_id']}")
        time.sleep(message_data['compute_time'])  # Simulate the GPU compute time

        response = f"Processed batch {message_data['batch_id']} from App {message_data['app_id']}"
        self.channel.basic_publish(exchange='', routing_key=response_queue, body=response)
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def start(self):
        print("GPUResourceManager started.")
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=self.request_queue, on_message_callback=self.process_request)
        self.channel.start_consuming()

    def stop(self):
        self.connection.close()
    
    def run_in_thread(self):
        thread = threading.Thread(target=self.start)
        thread.start()
        return thread

class GPUResourceManager:
    def __init__(self, server='localhost:9092', request_queue='gpu_requests', engine='kafka'):
        if engine == 'kafka':
            self.manager = GPUResourceManagerKafka(kafka_server=server, request_topic=request_queue)
        else:
            self.manager = GPUResourceManagerRabbitMQ(rabbitmq_server=server, request_queue=request_queue)
    
    def process_message(self, message):
        self.manager.process_message(message)
    
    def start(self):
        self.manager.start()
    
    def stop(self):
        self.manager.stop()
    
    def run_in_thread(self):
        return self.manager.run_in_thread()
    
    