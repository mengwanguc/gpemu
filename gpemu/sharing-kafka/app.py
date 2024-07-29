from kafka import KafkaProducer, KafkaConsumer
import json
import uuid
import time

app_id = str(uuid.uuid4())  # Unique identifier for each application
request_topic = 'batch_requests'
response_topic = f'response_{app_id}'

producer = KafkaProducer(bootstrap_servers='localhost:9092')
consumer = KafkaConsumer(response_topic, bootstrap_servers='localhost:9092',
                         auto_offset_reset='earliest', enable_auto_commit=True)

def send_batch_request_and_wait(batch_id, batch_size, batch_compute_time):
    batch_start = time.time()
    message = json.dumps({'app_id': app_id, 'batch_id': batch_id, 'duration': batch_compute_time, 'response_topic': response_topic})
    producer.send(request_topic, message.encode('utf-8'))
    # Wait for response
    for message in consumer:
        break  # Exit after receiving the response
    batch_end = time.time()
    batch_duration = batch_end - batch_start
    image_process_speed = batch_size / batch_duration
    print("{}\t{}\t{}\t{}".format(app_id, batch_id, time.time(), image_process_speed))


print(time.time())
for batch_id in range(200):  # Example batch processing loop
    send_batch_request_and_wait(batch_id, 128, 0.5753)
