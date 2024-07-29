### Emulate time-slice sharing using Kafka


```
sudo apt update
sudo apt install -y default-jdk
```

confirm it's installed
```
java --version
```


Add this to the end of .bashrc

```
# export JAVA_HOME=$(/usr/lib/jvm/java-11-openjdk-amd64)
# export PATH=$JAVA_HOME/bin:$PATH
export JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"
export PATH="$JAVA_HOME/bin:$PATH"
```

*Reload terminal*

NOTE: When I say "open another terminal", actually you can always use `tmux` instead
if you are familiar with it.



download kafka
```
cd ~
wget https://dlcdn.apache.org/kafka/3.6.1/kafka_2.13-3.6.1.tgz
tar -xzf kafka_2.13-3.6.1.tgz
```

launch zookeeper server
```
cd ~/kafka_2.13-3.6.1
bin/zookeeper-server-start.sh config/zookeeper.properties
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


