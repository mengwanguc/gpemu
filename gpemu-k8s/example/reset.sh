kubectl delete pod gpemu-test
kubectl delete pod gpemu-test-2
kubectl delete daemonset gpemu-k8s-device-plugin-daemonset -n kube-system
kubectl delete configmap gpemu -n kube-system