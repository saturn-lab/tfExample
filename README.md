# Tensorboard example with MNIST data.

# Step 1. Git clone   

download the code


# Step 2. prepare Data 


MNIST data is zipped. 

## PLEASE UNZIP IT!!!!

unzip the mnist.zip and put the mnist.mat file in './tfExample/data' foler. 

# Step 3. Train 

## In Windows or Mac:
'$cd tfExample'  (in "./tfExample" folder)

'$python model.py' 

'$tensorboard --logdir=log'


## In Linux:

'$python ./tfExample/model.py' 

'$python -m tensorflow.tensorboard --logdir=./tfExample/log'


# Step 4. Show results
Open Web Browser, and type the following URL:

'http://localhost:6006'

