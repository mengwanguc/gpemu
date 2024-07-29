import pika
import time
import json

def on_request(ch, method, props, body):
    try:
        request = json.loads(body)
        print(f"Processing request from App {request['app_id']} for GPU access.")
        print(request)

        # Simulate GPU processing time
        time.sleep(request['processing_time'])

        response = json.dumps({'app_id': request['app_id'], 'status': 'completed'})
        ch.basic_publish(exchange='',
                         routing_key=props.reply_to,
                         properties=pika.BasicProperties(correlation_id=props.correlation_id),
                         body=response)
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f"Completed processing for App {request['app_id']} and sent response to {props.reply_to}.")
    except Exception as e:
        print(f"Error processing request: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

def start_consuming():
    connection = None
    channel = None
    while True:
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
            channel = connection.channel()

            channel.queue_declare(queue='gpu_requests')
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue='gpu_requests', on_message_callback=on_request)

            print("GPU Manager started. Awaiting requests.")
            channel.start_consuming()
        except pika.exceptions.AMQPConnectionError as e:
            print(f"Connection error: {e}. Reconnecting in 5 seconds...")
            time.sleep(5)
        except KeyboardInterrupt:
            print("Interrupted by user, stopping...")
            if connection and connection.is_open:
                connection.close()
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            if connection and connection.is_open:
                connection.close()
            time.sleep(5)

if __name__ == "__main__":
    start_consuming()
