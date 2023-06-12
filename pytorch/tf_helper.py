import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Input, ReLU, LeakyReLU, ELU
from tensorflow.keras.models import Model
import numpy as np
 
def predict(interpreter, sample, input_details, output_details):
    interpreter.set_tensor(input_details[0]['index'], [sample])

    interpreter.invoke()
    
    k = [np.argmax(x) for x in interpreter.get_tensor(output_details[0]['index'])]
    #Nc = [int(round(x)) for x in interpreter.get_tensor(output_details[1]['index'])[0]]
    #p = [np.argmax(x) for x in interpreter.get_tensor(output_details[2]['index'])]
    #return (k[0],Nc[0],p[0])
    return k[0]


def create_tf_model(n_layers, n_neurons, regularizer, n_output, activation, alpha):
    inputs = Input(shape=(6,))
    for i in range(1,n_layers+1):
        if i == 1:
            if activation == 'selu':
                x = Dense(n_neurons, activity_regularizer=regularizer, activation="selu")(inputs)
            else:
                x = Dense(n_neurons, activity_regularizer=regularizer)(inputs)
                if activation == 'relu':
                    x = ReLU()(x)
                elif activation == 'leaky':
                    x = LeakyReLU(alpha=alpha)(x)
                elif activation == "elu":
                    x = ELU(alpha=alpha)(x)
        else:
            if activation == 'selu':
                x = Dense(n_neurons, activity_regularizer=regularizer, activation="selu")(x)
            else:
                x = Dense(n_neurons, activity_regularizer=regularizer)(x)
                if activation == 'relu':
                    x = ReLU()(x)
                elif activation == 'leaky':
                    x = LeakyReLU(alpha=alpha)(x)
                elif activation == "elu":
                    x = ELU(alpha=alpha)(x)
    output = Dense(n_output, name="k")(x)
    
    model = Model(inputs=inputs, outputs=[output])
    
    model.compile(metrics=['accuracy'])
    
    return model