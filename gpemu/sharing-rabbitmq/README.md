# Emulate time-slice sharing using RabbitMQ

## Install rabbitmq

```
bash install-rabbitmq.sh
```

```
sudo systemctl start rabbitmq-server
```

install pika to interact with rabbitmq

```
pip install pika
```




open another terminal. Launch kafka server:

```
cd ~/kafka_2.13-3.6.1
bin/kafka-server-start.sh config/server.properties
```

open another terminal. 
```
pip install kafka-python
python sharing_manager.py 
```

open another terminal.

```
python main-emulator-share.py  --epoch 1 --profile-batches -1 --workers 8 --gpu-type=p100 --gpu-count=1 --arch=resnet50 --emulator-version=1 ~/data/imagenette2
```



### checklist
kafka delete topic
```
bash bin/kafka-topics.sh --bootstrap-server localhost:9092 --delete --topic batch_requests
```


