# Allox with emulator

```
git clone git@github.com:mengwanguc/kubernetes-allox.git
```

## Install old golang

```
sudo apt remove golang-go
sudo rm -rf /usr/local/go

cd /tmp
wget https://go.dev/dl/go1.10.8.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.10.8.linux-amd64.tar.gz
cd -
```

then add this to `~/.bashrc`

```
export PATH=$PATH:/usr/local/go/bin
export GOPATH=$HOME/go
```

then 

```
source ~/.bashrc
```

## reset-pod
cd ~/kubernetes-allox
sudo cp ~/kubernetes-allox/kuberst /usr/local/bin


## compile
https://github.com/kubernetes/website/blob/release-1.12/content/en/docs/tasks/administer-cluster/configure-multiple-schedulers.md

website/content/en/examples/admin/sched/my-scheduler.yaml

https://stackoverflow.com/questions/37586169/creating-custom-scheduler-doesnt-work


```
kubectl edit clusterrole system:kube-scheduler
```
```
- apiGroups:
  - storage.k8s.io
  resources:
  - storageclasses
  verbs:
  - watch
  - list
  - get
```
for resource pod add create permission:
```
- apiGroups:
  - ""
  resources:
  - pods
  verbs:
  - delete
  - get
  - list
  - watch
  - create
```

```
cd kubernetes-allox
make
make kube-scheduler

sudo docker build -t wangm12/my-kube-scheduler:allox-egpu .
sudo docker push wangm12/my-kube-scheduler:allox-egpu
kubectl delete -f my-scheduler.yaml
kubectl create -f my-scheduler.yaml
kubectl get pods --namespace=kube-system
kubectl get pods --all-namespaces
```



### create namespace



```
kubectl describe pod my-scheduler-5dbbfd997f-p6bhs -n kube-system
kubectl logs my-scheduler-96649cb67-6hfxx  -n kube-system
kubectl logs pytorch-egpu -n user1
kubectl describe pod pytorch-egpu -n user1
kubectl exec --stdin --tty pytorch-egpu -n user1 -- /bin/bash

kubectl logs pytorch-cpu -n user1
kubectl logs pytorch-egpu -n user1
kubectl describe pod pytorch-cpu -n user1
kubectl describe pod pytorch-egpu -n user1
```

Force Kubernetes to pull image: https://www.baeldung.com/ops/kubernetes-pull-image-again#:~:text=One%20way%20to%20force%20Kubernetes,already%20present%20on%20the%20node.


# create namespaces

```
cd pod/sched
bash create-namespaces.sh
kubectl get namespaces --show-labels
```




# custom docker images

DOCKER_BUILDKIT=1 sudo docker build -t wangm12/gpemu-pytorch:base .
sudo docker run --ulimit memlock=-1:-1 -it wangm12/gpemu-pytorch:base /bin/bash

sudo docker push wangm12/gpemu-pytorch:base


DOCKER_BUILDKIT=1 sudo docker build -t wangm12/gpemu-pytorch:egpu .
sudo docker push wangm12/gpemu-pytorch:egpu

sudo docker run --ulimit memlock=-1:-1 -v /data:/data --shm-size=10g -it wangm12/gpemu-pytorch:egpu /bin/bash


kubectl logs pytorch-egpu -n user1
kubectl exec --stdin --tty pytorch-egpu -n user1 -- /bin/bash


DOCKER_BUILDKIT=1 sudo docker build -t wangm12/gpemu-pytorch:cpu .
sudo docker push wangm12/gpemu-pytorch:cpu

sudo docker run --ulimit memlock=-1:-1 -v /data:/data --shm-size=10g -it wangm12/gpemu-pytorch:cpu /bin/bash

kubectl logs pytorch-cpu -n user1
kubectl exec --stdin --tty pytorch-cpu -n user1 -- /bin/bash


kubectl get pods --all-namespaces
kubectl logs my-scheduler-96649cb67-cqf97  -n kube-system

## Reserve CPU/memory resources



## Notes

scheduler.go: SchedulerOne() ->
factory.go: getNextPod() ->
scheduling_queue.go: PickNextPod(client clientset.Interface) ->
util.go: EqualShare(allPods []*v1.Pod, client clientset.Interface) ->





strDemands := strings.Split(container.Command[cmdIdx], ",")
cpuDemand: strDemands[0]
gpuDemand: strDemands[1]
memDemand: strDemands[2]
cTime: strDemands[3]


AllocatableResource:&cache.Resource{MilliCPU:96000, Memory:201105616896, EphemeralStorage:405676817761, AllowedPodNumber:110, ScalarResources:map[v1.ResourceName]int64{"hugepages-1Gi":0, "hugepages-2Mi":0, "ucare.cs.uchicago.edu/e-gpu":3, "github.com/e-gpu":0}}}