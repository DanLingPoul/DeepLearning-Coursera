import numpy as np
from utils import *
import random

data=open('dinos.txt','r').read()
data=data.lower()
chars=list(set(data))
data_size,vocab_size=len(data),len(chars)

char_to_ix={ch:i for i,ch in enumrate(sorted(chars))}
ix_to_char={i:ch for i,ch in enumrate(sorted(chars))}

#Building blocks of the model

#Clipping the gradients in the optimization loop
def clip(gradients,maxValue):
    '''
    Clips the gradients' values between minimum and maximum.
    
    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue
    
    Returns: 
    gradients -- a dictionary with the clipped gradients.
    '''
    dWaa,dWax,dWya,db,dby=gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
    for gradient in [dWax,dWaa,dWya,db,dby]:
        np.clip(gradient,-maxValue,maxValue,out=gradient)
    gradients={"dWaa":dWaa,"dWax":dWax,"dWya": dWya, "db": db, "dby": dby}
    return gradients

#Sampling
def sample(parameters,char_to_ix,seed):
     """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b. 
    char_to_ix -- python dictionary mapping each character to an index.
    seed -- used for grading purposes. Do not worry about it.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size=by.shape[0]
    n_a=Waa.shape[1]
    #Create the one-hot vector x for the first character(initializing the sequence generation).
    x=np.zeros((vocab_size,1))
    #Initialize a_prev as zeros
    a_prev=np.zeros((n_a,1))
    indices=[]
    idx=-1 #idx is a flag to detect a newline character.
    
    # Loop over time-steps t. At each time-step, sample a character from a probability distribution and append 
    # its index to "indices". We'll stop if we reach 50 characters (which should be very unlikely with a well 
    # trained model), which helps debugging and prevents entering an infinite loop. 
    counter=0
    newline_character=char_to_ix['\n']
    while(idx!=newline_character and counter!=50):
        a=np.tanh(np.dot(Wax,x)+np.dot(Waa,a_prev)+b)
        z=np.dot(Wya,a)+by
        y=softmax(z)
        #How to use np.random.choice():
        #p=np.array([0.1,0.0,0.7,0.2])
        #index=np.random.choice([0,1,2,3],p=p.ravel())
        #This means that you will pick the index according to the distribution: P(index=0)=0.1,P(index=1)=0.0,P(index=2)=0.7,P(index=3)=0.2.
        idx=np.random.choice(list(range(vocab_size)),p=y.ravel())
        indices.append(idx)
        #overwrite the input character as the one corresponding to the sampled index
        x=np.zeros((vocab_size,1))
        x[idx]=1
        a_prev=a
        
        seed+=1
        counter+=1
        
    if(counter==50):
        indices.append(char_to_ix['\n'])
    return indices

#Build the language model
def optimize(X,Y,a_prev,parameters,learning_rate=0.01):
    """
    Execute one step of the optimization to train the model.
    
    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    a_prev -- previous hidden state.
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    learning_rate -- learning rate for the model.
    
    Returns:
    loss -- value of the loss function (cross-entropy)
    gradients -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
    """
    loss,cache=rnn_forward(X,Y,a_prev,parameters)
    gradients,a=rnn_backward(X,Y,parameters,cache)
    gradients=clip(gradients,5)
    parameters=update_parameters(parameters,gradients,learning_rate)
    return loss,gradients,a[len(X)-1]

#Training the model
def model(data,ix_to_char,char_to_ix,num_iterations=35000,n_a=50,dino_names=7,vocab_size=27):
    """
    Trains the model and generates dinosaur names. 
    
    Arguments:
    data -- text corpus
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration. 
    vocab_size -- number of unique characters found in the text, size of the vocabulary
    
    Returns:
    parameters -- learned parameters
    """
    n_x,n_y=vocab_size,vocab_size
    parameters=initialize_parameters(n_a,n_x,n_y)
    loss=get_initial_loss(vocab_size,dino_names)
    # Build list of all dinosaur names (training examples).
    with open("dinos.txt") as f:
        examples=f.readlines()
    examples=[x.lower().strip() for x in examples]
    #Shuffle list of all dinosaur names
    np.random.seed(0)
    np.random.shuffle(examples)
    a_prev=np.zeros((n_a,1))
    #optimization loop
    for j in range(num_iterations):
        index=j%len(examples)
        X=[None]+[char_to_ix[ch] for ch in examples[index]]
        Y=X[1:]+[char_to_ix["\n"]]
        curr_loss,gradients,a_prev=optimize(X,Y,a_prev,parameters)
        loss=smooth(loss,cur_loss)
        
        if j%2000==0:
            print('Iteration: %d,Loss:%f' %(j,loss)+'\n')
            seed=0
            for name in range(dino_names):
                sampled_indices=sample(parameters,char_to_ix,seed)
                print_sample(sampled_indices,ix_to_char)
                seed+=1
            print('\n')
    return parameters

    
    

