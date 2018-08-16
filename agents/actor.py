from keras import layers,models,optimizers
from keras import backend as K 
from keras import initializers

class Actor:
    def __init__(self,state_size,action_size,action_low,action_high):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        self.build_model()
    
    def build_model(self):
        
        states = layers.Input(shape=(self.state_size,),name='states')
        # net = layers.Dense(units=128,kernel_initializer='glorot_normal')(states)
        # net = layers.BatchNormalization()(net)
        # net = layers.LeakyReLU(1e-2)(net)
        
        # net = layers.Dense(units=64,kernel_initializer='glorot_normal')(net)
        # net = layers.BatchNormalization()(net)
        # net = layers.LeakyReLU(1e-2)(net)
        net = layers.Dense(units=32,activation='relu')(states)
        net = layers.Dense(units=64,activation='relu')(net)
        net = layers.Dense(units=32,activation='relu')(net)

        
        #raw_actions = layers.Dense(units=self.action_size,activation='sigmoid',name='raw_actions',kernel_initializer='glorot_normal')(net)
        raw_actions = layers.Dense(units=self.action_size,activation='sigmoid',kernel_initializer=initializers.RandomUniform(minval=-0.0003,maxval=0.0003,seed=None),name='raw_actions')(net)

        actions = layers.Lambda(lambda x:(x*self.action_range)+self.action_low,name='actions')(raw_actions)

        self.model = models.Model(inputs=states,outputs=actions)

        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients*actions)

        optimizer = optimizers.Adam(lr=0.0001)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights,loss=loss)

        self.train_func = K.function(inputs=[self.model.input,action_gradients,K.learning_phase()],outputs=[],updates=updates_op)
