from kafka import KafkaProducer, KafkaConsumer
import json
import uuid
import time

class App:
    def __init__(self):
        self.app_id = str(uuid.uuid4())
        self.request_topic = 'batch_requests'
        self.response_topic = f'response_{self.app_id}'

        self.producer = KafkaProducer(bootstrap_servers='localhost:9092')
        self.consumer = KafkaConsumer(self.response_topic, bootstrap_servers='localhost:9092',
                                auto_offset_reset='earliest', enable_auto_commit=True)
    

    def emuSharedCompute(self, batch_id, batch_compute_time):
        message = json.dumps({'app_id': self.app_id, 'batch_id': batch_id, 'duration': batch_compute_time, 'response_topic': self.response_topic})
        self.producer.send(self.request_topic, message.encode('utf-8'))
        # Wait for response
        for message in self.consumer:
            break  # Exit after receiving the response