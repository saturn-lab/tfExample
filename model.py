#!/opt/anaconda3/bin/python

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import math
import numpy
from PIL import Image

CLASS_NUM=10


def variable_summaries(var):
    mean = tf.reduce_mean(var)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('mean', mean)
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def Dense(name, inData, hidden, noActivation = False):
    with tf.variable_scope(name):
        in_node = int(inData.get_shape()[1])
        W = tf.Variable(tf.random_normal(shape = [in_node, hidden], mean = 0.01, stddev = 0.01))
        b = tf.Variable(tf.random_normal(shape = [hidden], mean = 0.02, stddev = 0.01))
        
        with tf.name_scope('Weight'):
            variable_summaries(W)
        with tf.name_scope('Bias'):
            variable_summaries(b)
    
        output = tf.matmul(inData, W) + b
        if not noActivation:
            output = tf.nn.relu(output)

        tf.add_to_collection('Embed', output)
        
        return output

def model():
    global CLASS_NUM
    
    nodeX = tf.placeholder(tf.float32, [None, 784])
    nodeY = tf.placeholder(tf.int64, [None])
    nodeY_onehot = tf.one_hot(nodeY, CLASS_NUM, 1.0, 0.0, -1)
    
    Layer1 = Dense('Dense1', nodeX, 60)
    Layer2 = Dense('Dense2', Layer1, 30)
    logit = Dense('Output', Layer2, CLASS_NUM, noActivation = True)
    
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = nodeY_onehot, logits = logit))
    tf.summary.scalar('loss', loss)
    
    learning_rate = tf.placeholder(tf.float32, None)
    tf.summary.scalar('learning rate', learning_rate)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    
    return nodeX, nodeY, Layer1, Layer2, logit, learning_rate, train

def evaluate(nodeY, logit):
    predict = tf.argmax(logit, axis=1)
    digitize = tf.cast(tf.equal(nodeY, predict), tf.float32)
    win_rate = tf.reduce_mean(digitize)
    tf.summary.scalar('correct rate', win_rate)
    return predict, win_rate
    
def getLearningRate(step):
    return 0.001

def generateMetaData(path, Y):
    with open(path + 'meta.tsv', 'w') as f:
        size = Y.shape[0]
        for i in range(0, size):
            label = str(Y[i])
            f.write(label +'\n')

def generateMetaGraph(path, X, spSize):
    num = X.shape[0]
    W = math.floor(math.sqrt(num))
    H = math.ceil(num/W)
    mat = numpy.zeros((W*28, H*28), dtype = numpy.float32)
    
    for i in range(0, num):
        starth = (i//W) * 28
        startw = (i%H) * 28
        mat[starth:starth+28, startw:startw+28] = X[i, :].reshape((28, 28))
    mat = 255.0 * (1.0 - mat)

    img =Image.fromarray(mat)
    
    img = img.resize([W*spSize[0], H*spSize[1]], Image.BILINEAR)
    img = img.convert('RGB')
    img.save(path + 'meta.png')

def embedding(vlist, rlist, metaPath, spSize):
    vs = []
    for i in range(0, len(vlist)):
        v = tf.Variable(rlist[i], name = vlist[i].name.split('/')[0]) # x/relu
        vs.append(v)
    
    with tf.Session() as sess:
        tf.variables_initializer(vs).run() # assign to vs
        saver = tf.train.Saver(vs) 
        saver.save(sess, './log/model.ckpt', 0) # only contain vs
    
    # get writer and config
    summary_writer = tf.summary.FileWriter('./log/')
    config = projector.ProjectorConfig()

    # set config
    for v in vs:
        embed = config.embeddings.add()
        embed.tensor_name = v.name
        embed.metadata_path = metaPath + 'meta.tsv'
        embed.sprite.image_path = metaPath + 'meta.png'
        embed.sprite.single_image_dim.extend(spSize)
    # write
    projector.visualize_embeddings(summary_writer, config)
    
    
if __name__ == '__main__':
    from dataset import readMNIST, batchGenerator

    X, Y, layer1, layer2, logit, lr, train = model()
    predict, wrate = evaluate(Y, logit)
    
    trainX, trainY, testX, testY = readMNIST(asImage = False)
    gen = batchGenerator(trainX, trainY, 512)
    
    summary_op = tf.summary.merge_all()
    
    with tf.Session() as sess:
        # writer for summary
        writer = tf.summary.FileWriter('./log/', sess.graph)

        tf.global_variables_initializer().run()
        
        for i in range(0, 10):
            datax, datay = next(gen)
            learning_rate = getLearningRate(i)
            
            _, win_rate, summary = sess.run([train, wrate, summary_op], {X:datax, Y:datay, lr:learning_rate})
            writer.add_summary(summary, i)
           
            print('Cycle %d. win rate %f'%(i, win_rate))
        
        # run Test as well as obtain the to-be-embedded variable
        runList = tf.get_collection('Embed')
        runList.insert(0, wrate)
        resultList = sess.run(runList, {X:testX, Y:testY})

        print('final rate: %f'%resultList[0])
    
    # configure for meta Data
    metaPath = './metaEmbed/'
    spriteSize = (12, 12)
    
    embedding(runList[1:], resultList[1:], metaPath, spriteSize)
    import os
    if not os.path.exists(metaPath):
        os.makedirs(metaPath)
    generateMetaData(metaPath, testY)
    generateMetaGraph(metaPath, testX, spriteSize)
