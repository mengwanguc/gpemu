from kafka import KafkaConsumer, KafkaProducer
import json
import time

request_topic = 'batch_requests'
consumer = KafkaConsumer(request_topic, bootstrap_servers='localhost:9092',
                         auto_offset_reset='earliest', enable_auto_commit=True)
producer = KafkaProducer(bootstrap_servers='localhost:9092')

def process_message(message):
    message_data = json.loads(message.value.decode())
    response_topic = message_data['response_topic']
    print(f"Processing batch {message_data['batch_id']} from App {message_data['app_id']}")
    time.sleep(message_data['duration'])  # Emulate processing time

    # Send response
    response = f"Processed batch {message_data['batch_id']} from App {message_data['app_id']}"
    producer.send(response_topic, response.encode('utf-8'))

for message in consumer:
    process_message(message)
