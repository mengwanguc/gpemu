# gpemu-k8s


## Set up SSH key

```
ssh-keygen -t rsa -b 4096
```

```
ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts
cat ~/.ssh/id_rsa.pub
```
Copy and paste into: https://github.com/settings/keys

### Configure github

NOTE: the example below is for Meng. You should modify it to your own account.

```
git config --global user.email "mengwanguc@gmail.com"
git config --global user.name "Meng Wang"
```

## clone this repo

```
git clone git@github.com:mengwanguc/gpemu-k8s.git
cd gpemu-k8s
```

## gitc

```
sudo cp gitc /usr/local/bin
```

## Install Kubernetes

Please refer to [install-k8s.md](./install-k8s.md) to install K8s.

## Install gpemu-k8s-device-plugin

Please refer to [install-plugin](./install-plugin.md) to install the k8s device plugin for GPEmu.

## References

https://github.com/everpeace/k8s-host-device-plugin