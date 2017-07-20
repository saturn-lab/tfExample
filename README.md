#Tensorboard example with MNIST data.

# Git clone   

download the code


# Data 


MNIST data is zipped. 

# PLEASE UNZIP IT!!!!

unzip the mnist.zip and put the mnist.mat file in './tfExample/data' foler. 

# Train 

## In Linux:

'$python ./tfExample/model.py' 

'$python -m tensorflow.tensorboard --logdir=./tfExample/log'

## In Windows:
'$python model.py'  (in "./tfExample" folder)

'$python -m tensorflow.tensorboard --logdir=log'


# Show results
Open Web Browser, and type the following URL:

'http://localhost:6006'

