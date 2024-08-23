cd ~
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh -b -p $HOME/anaconda3
$HOME/anaconda3/bin/conda init
source ~/.bashrc
rm Anaconda3-2021.11-Linux-x86_64.sh