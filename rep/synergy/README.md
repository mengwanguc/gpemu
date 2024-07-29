## Synergy with GPEmu

https://github.com/mengwanguc/synergy/

### Setup

Use the docker image `jayashreemohan/synergy_dali`.
```
- docker pull jayashreemohan/synergy_dali:latest
- nvidia-docker run --ipc=host --mount src=/,target=/datadrive/,type=bind -it --rm --network=host --privileged jayashreemohan/synergy_dali:latest /bin/bash
- git clone https://github.com/mengwanguc/synergy/.git
- cd synergy/simulator/deployment
- make
```
Now skip to [Getting Started](README.md#getting-started)

#### Setup profiler
```
- cd src
- cd profiler; ./prereq.sh; cd ..
```

#### Setup simulator
```
- cd simulator/deployment
- ./upgrade_pip.sh
- pip install -r requirements.txt
- make
```

#### Setup iterator
Synergy uses its own PyTorch iterator that is built on top of DALI & CoorDL. So before you run any profiling experiment, or deployment in a real GPU cluster,  build a docker container with this iterator support by following these steps. Note that this is not required to run the simulation experiments.
```
- git clone https://github.com/jayashreemohan29/Synergy-CoorDL.git
- cd Synergy-CoorDL
- git submodule sync --recursive && git submodule update --init --recursive
- git checkout iterator_chk
- cd docker
- CREATE_RUNNER="YES" ./build.sh
```
 This will create a docker container tagged nvidia/dali:py36_cu10.run

Alternately you could use the docker image hosted [here](https://hub.docker.com/repository/docker/jayashreemohan/synergy_dali) using :
```
docker pull jayashreemohan/synergy_dali:latest
```

### Getting Started
The simplest way to get started with Synergy, is to test it out in a simulated cluster (can be run on a local machine without GPUs, or any specific hardware requirement). The test harness is the [runner.py](simulator/runner.py) file. For instance, to evaluate a FIFO scheduling policy using the default GPU-proportional allocation and synergy's tune based allocation, run the following command:

```
python runner.py --cluster_job_log trace/cluster_job_log --plot  2>&1 | tee  out-deploy
```

Other options supported by the test harness are:

* --cluster_job_log : The Philly trace
* --plot : Plot the CDF and JCT of runs
* --multigpu : Allow multi-GPU jobs in the mix
* --no_exp :  Disable the exponential arrival distribution
* --philly_arrival : Use arrival information as is from the Philly trace (must also use --no_exp)
* --rec_trace : Record the generated trace
* --replay_trace : Replay a previously recorded trace
* --config_file : Cluster configuration, default of 128GPUs in configs/default_cluster.ini
* --no_simulate : Run it on a real GPU cluster
* schedulers : List of schedulers to run, for eg., ['FIFO+fair' , 'FIFO+tune']
* jobs_per_hour : List of different arrival rates, for eg., np.arange(1.0, 10, 1)
* class split : Split of <vision. language, speech> models, for eg., class_split=[(20,70,10)]