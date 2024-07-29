## References

https://github.com/everpeace/gpemu-k8s-device-plugin

## Dockerhub log in

```
sudo docker login
```

## Create docker

```
sudo docker build -t wangm12/gpemu-k8s-device-plugin .

sudo docker run wangm12/gpemu-k8s-device-plugin

sudo docker push wangm12/gpemu-k8s-device-plugin
```

```
kubectl create -f gpemu.yaml
kubectl create -f gpemu-device-plugin.yaml
kubectl create -f test-pod.yml 
kubectl create -f test-pod-2.yml 
```

```
kubectl get configmaps --all-namespaces
kubectl get pods --all-namespaces
kubectl get daemonsets --all-namespaces
kubectl describe node worker-1
kubectl describe pod gpemu-k8s-device-plugin-daemonset-sdzd9 -n kube-system
kubectl logs gpemu-k8s-device-plugin-daemonset-h5d75 -n kube-system
kubectl logs gpemu-test
kubectl describe pod gpemu-test
```

```
kubectl delete pod gpemu-test
kubectl delete pod gpemu-test-2
kubectl delete daemonset gpemu-k8s-device-plugin-daemonset -n kube-system
kubectl delete configmap gpemu -n kube-system
```

## edev

```
sudo mkdir /edev
sudo chown -R cc edev
touch /edev/egpu0 /edev/egpu1 /edev/egpu2 
# sudo mknod -m 666 egpu0 b 262 0
# sudo mknod -m 666 egpu1 b 262 1
# sudo mknod -m 666 egpu2 b 262 1

ls /edev
```
Reference to major and minor numbers
https://github.com/torvalds/linux/blob/master/Documentation/admin-guide/devices.txt
