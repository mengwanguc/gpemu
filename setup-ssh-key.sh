ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts
cat ~/.ssh/id_rsa.pub