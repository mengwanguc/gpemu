`kubectl get` - list resources
`kubectl describe` - show detailed information about a resource
`kubectl logs` - print the logs from a container in a pod
`kubectl exec` - execute a command on a container in a pod

```
kubectl get nodes
kubectl get pods
kubectl get pods --all-namespaces
kubectl get deployments
kubectl get events
kubectl config view
```

get pods:

```
kubectl get pods
kubectl get pods --all-namespaces
kubectl get pods -l app=kubernetes-bootcamp
```

list all tainted nodes:

```
kubectl get nodes --selector='!node-role.kubernetes.io/master' -o custom-columns=NODE:.metadata.name,TAINTS:.spec.taints --no-headers=true
```

check log
```
kubectl logs hello-node-5f76cf6ccf-br9b5
```

get pod name
```
export POD_NAME=$(kubectl get pods -o go-template --template '{{range .items}}{{.metadata.name}}{{"\n"}}{{end}}')
echo Name of the Pod: $POD_NAME
```

open proxy
```
kubectl proxy
curl http://localhost:8001/version
curl http://localhost:8001/api/v1/namespaces/default/pods/$POD_NAME/
```

Execute commands on pod:
```
kubectl exec -ti $POD_NAME -- bash
```


Expose service to external traffic

```
kubectl expose deployment/kubernetes-bootcamp --type="NodePort" --port 8080
kubectl describe services/kubernetes-bootcamp
```

```
export NODE_PORT="$(kubectl get services/kubernetes-bootcamp -o go-template='{{(index .spec.ports 0).nodePort}}')"
echo "NODE_PORT=$NODE_PORT"
```

delete

```
kubectl delete service -l app=kubernetes-bootcamp
```

How to configure kubernetes manifest: yaml, configmap, dockerfile, ...