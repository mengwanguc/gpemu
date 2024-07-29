
## install old docker on ubuntu 16

```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
apt-cache policy docker-ce
sudo apt-get install docker-ce=18.06.0~ce~3-0~ubuntu
sudo systemctl status docker
```


### Verify docker installation

```
sudo docker run hello-world
```

### install kubernetes

https://www.edureka.co/blog/install-kubernetes-on-ubuntu

```
swapoff -a

curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

cat <<EOF | sudo tee /etc/apt/sources.list.d/kubernetes.list
deb https://apt.kubernetes.io/ kubernetes-xenial main
EOF

sudo apt update

sudo apt install -y kubernetes-cni=0.6.0-00
sudo apt install -y kubeadm=1.12.0-00 kubelet=1.12.0-00 kubectl=1.12.0-00

```