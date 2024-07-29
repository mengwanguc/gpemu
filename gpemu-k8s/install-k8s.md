## Install Docker

```
bash setup/install-docker.sh
```

### Verify docker installation

```
sudo docker run hello-world
```


### Install kubeadm

https://linuxconfig.org/how-to-create-a-kubernetes-cluster

```
bash setup/install-kubeadm.sh
```


### install cri-docker

```
cd /tmp
wget https://github.com/Mirantis/cri-dockerd/releases/download/v0.3.7/cri-dockerd_0.3.7.3-0.ubuntu-jammy_amd64.deb
sudo dpkg -i cri-dockerd_0.3.7.3-0.ubuntu-jammy_amd64.deb
cd -
```


### init master node

```
sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --cri-socket unix:///var/run/cri-dockerd.sock --node-name scheduler
```

You should see output like this:
```
Then you can join any number of worker nodes by running the following on each as root:

kubeadm join 10.52.2.142:6443 --token s3bpyl.5ad51wzm6o6jubyd \
        --discovery-token-ca-cert-hash sha256:7a46f5714c5b0dbf7024833ce21196842b54b2acc8f3a81057e39818e0b538d6
```

*Take a note of your own output , as you will need it to join the cluster*

Set up configuration file so that kubectl can run as non-root user.

```
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
```


create a network pod

```
kubectl apply -f https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml
```

Check the network status

```
kubectl get pods --all-namespaces
```

### Run on worker node

Join the network

```
sudo kubeadm join 10.52.2.142:6443 --token s3bpyl.5ad51wzm6o6jubyd \
        --discovery-token-ca-cert-hash sha256:7a46f5714c5b0dbf7024833ce21196842b54b2acc8f3a81057e39818e0b538d6 \
        --cri-socket unix:///var/run/cri-dockerd.sock \
        --node-name worker-1
```

If you have multiple workers, make sure you modify the `node-name` above because Kubernetes doesn't
allow workers to have same name.




### Reset kubeadm
https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/create-cluster-kubeadm/

```
sudo kubeadm reset --cri-socket unix:///var/run/cri-dockerd.sock


sudo rm -r /etc/cni/net.d
sudo rm -rf ~/.kube /etc/kubernetes /var/lib/etcd /var/lib/kubelet

```


## uninstall kubeadm

```
sudo apt-get purge kubeadm kubectl kubelet kubernetes-cni kube*
sudo apt-get autoremove  
sudo rm -rf ~/.kube
```

```

curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# echo "deb http://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list
cat <<EOF | sudo tee /etc/apt/sources.list.d/kubernetes.list
deb https://apt.kubernetes.io/ kubernetes-xenial main
EOF

sudo apt update

# sudo apt install -y kubernetes-cni=0.6.0-00
# sudo apt-get install -y kubeadm=1.12.0-00 kubelet=1.12.0-00 kubectl=1.12.0-00
sudo apt-get install -y kubelet=1.12.10-00 kubeadm=1.12.10-00 kubectl=1.12.10-00
```


