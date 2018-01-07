# cifar10
Submission for the CIFAR-10 competition.

Setup
-------
After cloning the repository

    sudo apt-get install python-pip python-dev python-virtualenv
    cd cifar10
    virtualenv env
    source ./env/bin/activate
    pip install -r requirement.txt
    
    
If the setup fails try doing this before:

	pip install Cython
	pip install h5py 
	sudo apt-get install python-tk


CPU
---------

If your system doesn't have a gpu you need to uninstall tensorflow-gpu and
reinstall tensorflow

    pip uninstall tensorflow-gpu
    pip install tensorflow==1.4.0 --ignore-installed


