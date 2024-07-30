## GPEmu library

The library for GPEmu, a GPU emulator for faster and cheaper prototyping and evaluation of deep learning system research.

### Prerequisites

1. Install asyncio
```
pip install asyncio
```

2. Install mlock

- mlock: https://github.com/gustrain/mlock

3. Install Kafka for sharing:

```
sudo apt update
sudo apt install -y default-jdk
```

Add this to the terminal:
```
export JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"
export PATH="$JAVA_HOME/bin:$PATH"
```

The reload the terminal. Then run:

```
cd ~
wget https://downloads.apache.org/kafka/3.8.0/kafka_2.13-3.8.0.tgz
tar -xzf kafka_2.13-3.8.0.tgz
pip install kafka-python
```

Open one terminal and run:
```
cd ~/kafka_2.13-3.8.0
bin/zookeeper-server-start.sh config/zookeeper.properties
```

Open another terminal and run:
```
cd ~/kafka_2.13-3.8.0
bin/kafka-server-start.sh config/server.properties
```

4. Install RabbitMQ for sharing:

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


### Installation

```bash
pip setup.py install
```