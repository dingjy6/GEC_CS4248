#!/usr/bin/env python
# coding: utf-8

# ## Importing necessary libraries

# In[1]:


import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense,RNN,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pickle
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.bleu_score import sentence_bleu
import shutil
from IPython.display import Image
import io
from tensorflow.keras.callbacks import ModelCheckpoint


# In[2]:


# !wget --header="Host: doc-0c-2g-docs.googleusercontent.com" --header="User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: en-US,en;q=0.9" --header="Cookie: AUTH_p1fg4b07u3raciih5atnsbt7sdk7uaqn=04473764286494136551|1619436900000|ntvoca5q76ejbphnpse8ta9kmglr2hae" --header="Connection: keep-alive" "https://doc-0c-2g-docs.googleusercontent.com/docs/securesc/k57aefr9gh8rpf3r9srcg2pseasqmf46/9k2k5dmu2g7k7ihp80mfmj0ogloeie23/1619436900000/04473764286494136551/04473764286494136551/1ZQzu5rtBPhZHud16ZP7CK6WZQTg8Zkyd?e=download&authuser=0" -c -O 'fasttext.zip'
# !unzip 'fasttext.zip'


# In[3]:


# !wget --header="Host: doc-0c-9k-docs.googleusercontent.com" --header="User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: en-US,en;q=0.9" --header="Cookie: AUTH_9oqjiictg1uaimhtdjkjtggkcuv7nfmb=12189577772596464101|1619437050000|4s90nmlanu4jned4ft1okv1cjsapkmd4" --header="Connection: keep-alive" "https://doc-0c-9k-docs.googleusercontent.com/docs/securesc/numa293bcjsiiaqru6kqq1t6plv2vd43/rgldscsv4meta3o9j2sg3n63vsd898cc/1619437050000/12189577772596464101/12189577772596464101/1h5WKH-dX3UmSqZEw-6R9kVuKURSy_TOR?e=download&authuser=3" -c -O 'attention80.zip'
# !unzip 'attention80.zip'


# ## Making data model ready

# In[4]:


## loading train, test and CV data
train = pd.read_csv("/kaggle/input/preprocessing-lan-data/train.csv",dtype=str)
cv = pd.read_csv("/kaggle/input/preprocessing-lan-data/val.csv",dtype=str)
test = pd.read_csv("/kaggle/input/preprocessing-lan-data/dev.csv",dtype=str)


# In[5]:


train["correct_inp"] = "<start> " + train["output"].astype(str)
train["correct_out"] = train["output"].astype(str) + " <end>"

cv["correct_inp"] = "<start> " + cv["output"].astype(str)
cv["correct_out"] = cv["output"].astype(str) + " <end>"

test["correct_inp"] = "<start> " + test["output"].astype(str)
test["correct_out"] = test["output"].astype(str) + " <end>"


# In[6]:


train.head()


# In[7]:


train.dropna(subset=["input"], inplace=True) 
cv.dropna(subset=["input"], inplace=True) 


# In[8]:


tokenizer_incorr = Tokenizer(filters="",lower=False)
tokenizer_incorr.fit_on_texts(train["input"].values)


# In[9]:


tokenizer_corr_inp = Tokenizer(filters="",lower=False)
tokenizer_corr_inp.fit_on_texts(train["correct_inp"].values)


# In[10]:


tokenizer_corr_out = Tokenizer(filters="",lower=False)
tokenizer_corr_out.fit_on_texts(train["correct_out"].values)


# In[11]:


#save keras tokenizer

with open("tokenizer_incorr_word_attention.pickle","wb") as temp1:
     pickle.dump(tokenizer_incorr,temp1)
    
with open("tokenizer_corr_inp_word_attention.pickle","wb") as temp2:
    pickle.dump(tokenizer_corr_inp,temp2)
    
with open("tokenizer_corr_out_word_attention.pickle","wb") as temp3:
    pickle.dump(tokenizer_corr_out,temp3)


# In[12]:


# code reference https://stackoverflow.com/questions/45735070/keras-text-preprocessing-saving-tokenizer-object-to-file-for-scoring
# loading saved tokenizer
with open("tokenizer_incorr_word_attention.pickle","rb") as temp1:
    tokenizer_incorr = pickle.load(temp1)
    
with open("tokenizer_corr_inp_word_attention.pickle","rb") as temp2:
    tokenizer_corr_inp = pickle.load(temp2)
    
with open("tokenizer_corr_out_word_attention.pickle","rb") as temp3:
    tokenizer_corr_out = pickle.load(temp3)


# In[13]:


incorr_train = tokenizer_incorr.texts_to_sequences(train["input"].values)
incorr_cv = tokenizer_incorr.texts_to_sequences(cv["input"].values)
print("vocab size of incorrrect sentences is",len(tokenizer_incorr.word_index))


# In[14]:


corr_train_inp = tokenizer_corr_inp.texts_to_sequences(train["correct_inp"].values)
corr_cv_inp = tokenizer_corr_inp.texts_to_sequences(cv["correct_inp"].values)
print("vocab size of corrrect sentences is",len(tokenizer_corr_inp.word_index))


# In[15]:


corr_train_out = tokenizer_corr_out.texts_to_sequences(train["correct_out"].values)
corr_cv_out = tokenizer_corr_out.texts_to_sequences(cv["correct_out"].values)
print("vocab size of corrrect sentences is",len(tokenizer_corr_out.word_index))


# In[16]:


incorr_train = np.array(pad_sequences(incorr_train,maxlen=40,padding="post",truncating='post'))
corr_train_inp = np.array(pad_sequences(corr_train_inp,maxlen=40,padding="post",truncating='post'))
corr_train_out = np.array(pad_sequences(corr_train_out,maxlen=40,padding="post",truncating='post'))

incorr_cv = np.array(pad_sequences(incorr_cv,maxlen=40,padding="post",truncating='post'))
corr_cv_inp = np.array(pad_sequences(corr_cv_inp,maxlen=40,padding="post",truncating='post'))
corr_cv_out = np.array(pad_sequences(corr_cv_out,maxlen=40,padding="post",truncating='post'))


# Tokenizing senetence for feeding to encoder

# In[17]:


"""# tokenizer_incorr = Tokenizer(filters="",lower=False)
# tokenizer_incorr.fit_on_texts(train["incorrect"].values)
incorr_train = np.array(tokenizer_incorr.texts_to_sequences(train["correct_inp"].values))
#incorr_cv = np.array(tokenizer_incorr.texts_to_sequences(cv["input"].values))
#print("vocab size of incorrrect sentences is",len(tokenizer_incorr.word_index))"""


# Tokenizing senetence for feeding to decoder as inpput

# In[18]:


"""# tokenizer_corr_inp = Tokenizer(filters="",lower=False)
# tokenizer_corr_inp.fit_on_texts(train["correct_inp"].values)
corr_train_inp = np.array(tokenizer_corr_inp.texts_to_sequences(train["correct_inp"].values))
#corr_cv_inp = np.array(tokenizer_corr_inp.texts_to_sequences(cv["correct_inp"].values))
#print("vocab size of corrrect sentences is",len(tokenizer_corr_inp.word_index))"""


# Tokenizing senetence which will be output of decoder

# In[19]:


"""# tokenizer_corr_out = Tokenizer(filters="",lower=False)
# tokenizer_corr_out.fit_on_texts(train["correct_out"].values)
corr_train_out = np.array(tokenizer_corr_out.texts_to_sequences(train["correct_out"].values))
#corr_cv_out = np.array(tokenizer_corr_out.texts_to_sequences(cv["correct_out"].values))
#print("vocab size of corrrect sentences is",len(tokenizer_corr_out.word_index))"""


# In[20]:


## #save keras tokenizer

# with open("tokenizer_incorr_word_attention.pickle","wb") as temp1:
#     pickle.dump(tokenizer_incorr,temp1)
    
# with open("tokenizer_corr_inp_word_attention.pickle","wb") as temp2:
#     pickle.dump(tokenizer_corr_inp,temp2)
    
# with open("tokenizer_corr_out_word_attention.pickle","wb") as temp3:
#     pickle.dump(tokenizer_corr_out,temp3)


# Padding training data

# In[21]:


"""incorr_train = np.array(pad_sequences(incorr_train,maxlen=25,padding="post",truncating='post'))
corr_train_inp = np.array(pad_sequences(corr_train_inp,maxlen=25,padding="post",truncating='post'))
corr_train_out = np.array(pad_sequences(corr_train_out,maxlen=25,padding="post",truncating='post'))"""


# #### Using pretrained fasttext embeddings

# In[22]:


import io
import numpy as np
import subprocess


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data

embeddings_index = load_vectors("/kaggle/input/attention-pre-trained/wiki-news-300d-1M.vec")
#embeddings_index


# In[23]:


embedding_matrix1 = np.zeros((52614, 300)) 
for word, i in tokenizer_incorr.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix1[i] = embedding_vector


# In[24]:


embedding_matrix1 


# In[25]:


embedding_matrix2 = np.zeros((47127, 300))  
for word, i in tokenizer_corr_inp.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix2[i] = embedding_vector

#np.save("C://Users//Sirui Wang//Desktop//4248data//corr_matrix", embedding_matrix2)


# In[26]:


embedding_matrix2 


# #### Defining model artchitecture

# In[27]:


# code taken from attention mechanism assignment

############################## Encoder class #############################################################

class Encoder(tf.keras.layers.Layer):
    '''
    Encoder model -- That takes a input sequence and returns encoder-outputs,encoder_final_state_h,encoder_final_state_c
    '''

    def __init__(self,inp_vocab_size,embedding_size,lstm_size,input_length):
        
        super().__init__()
        self.lstm_size = lstm_size
        self.embedding = Embedding(input_dim=inp_vocab_size, 
                                   output_dim=50, 
                                   #input_length=input_length,
                                   mask_zero=True,
                                   name="embedding_layer_encoder", 
                                   #weights=[embedding_matrix1], 
                                   trainable=False)
        self.lstmcell = tf.keras.layers.LSTMCell(lstm_size)
        self.encoder_lstm = RNN(self.lstmcell,return_sequences=True, return_state=True)


    def call(self,input_sequence,states):

        output1 = self.embedding(input_sequence)
        enco_output, enco_state_h, enco_state_c = self.encoder_lstm(output1, initial_state=states)
        return enco_output, enco_state_h, enco_state_c

    
    def initialize_states(self,batch_size):

        initial_hidden_state = tf.zeros([batch_size,self.lstm_size])
        initial_cell_state = tf.zeros([batch_size,self.lstm_size])
        
        return [initial_hidden_state,initial_cell_state]
    
############################## Decoder class #############################################################
    
# code reference for concat scoing scoring function from https://www.tensorflow.org/tutorials/text/nmt_with_attention
from tensorflow.keras.layers import Input, Softmax, RNN, Dense, Embedding, LSTM
class Attention(tf.keras.layers.Layer):
  '''
    Class the calculates score based on the scoring_function using Bahdanu attention mechanism.
  '''
  def __init__(self,scoring_function,att_units):


    # Please go through the reference notebook and research paper to complete the scoring functions
    super().__init__()
    self.scoring_function = scoring_function
    
    if self.scoring_function=='dot':
      # Intialize variables needed for Dot score function here
        #self.similarity = []
        self.softmax = Softmax(axis=1)
        #self.similarity = [j for j in range(att_units)]
        pass
    if scoring_function == 'general':
      # Intialize variables needed for General score function here
        self.softmax = Softmax(axis=1)
        self.att_units = att_units
        self.W = tf.keras.layers.Dense(att_units)
        pass
    if scoring_function == 'concat':
      # Intialize variables needed for Concat score function here
        self.softmax = Softmax(axis=1)
        self.att_units = att_units
        self.W = tf.keras.layers.Dense(att_units)
        self.V = tf.keras.layers.Dense(1)
        pass
  
  def call(self,decoder_hidden_state,encoder_output):
    
    if self.scoring_function == 'dot':
        # Implement Dot score function here
        #print(decoder_hidden_state.shape,encoder_output.shape)
        attention_weight = tf.matmul(encoder_output,tf.expand_dims(decoder_hidden_state,axis=2))
        #print(attention_weight.shape)
        context = tf.matmul(tf.transpose(encoder_output, perm=[0,2,1]),attention_weight)
        context = tf.squeeze(context,axis=2)
        output = self.softmax(attention_weight)
        return context,output
    
class One_Step_Decoder(tf.keras.Model):
  def __init__(self,tar_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units):

      # Initialize decoder embedding layer, LSTM and any other objects needed
        super().__init__()
        self.tar_vocab_size = tar_vocab_size
        self.lstm_size = dec_units
        self.att_units = att_units
        self.score_fun = score_fun
        #print("output vocan size ",tar_vocab_size)
        self.embedding = Embedding(input_dim=tar_vocab_size, 
                                   output_dim=300, 
                                   #input_length=input_length,
                                   mask_zero=True,
                                   name="embedding_layer_encoder",
                                   #weights=[embedding_matrix2], 
                                   trainable=False)
        self.lstmcell = tf.keras.layers.LSTMCell(dec_units)
        self.decoder_lstm = RNN(self.lstmcell,return_sequences=True, return_state=True)
        self.dense   = Dense(tar_vocab_size)
        #self.decoder_lstm = LSTM(lstm_size, return_state=True, return_sequences=True, name="decoder_LSTM")
        self.attention=Attention(self.score_fun,self.att_units)


  def call(self,input_to_decoder, encoder_output, state_h,state_c):

        output2 = self.embedding(input_to_decoder)
        #print("one step decoder SHAPE after embedding:",output2.shape)
        output2 = tf.squeeze(output2,axis=1)
        #print("one step decoder SHAPE after embedding and sqeezing:",output2.shape)

        # step b
#         attention=Attention(self.score_fun,self.att_units)
        context_vector,attention_weights=self.attention(state_h,encoder_output)
        # step c
        output3 = tf.concat([context_vector,output2],1)
        #print("shape after concating ",output3.shape)
        output3 = tf.expand_dims(output3,1)
        deco_output, deco_state_h, deco_state_c = self.decoder_lstm(output3,initial_state=[state_h,state_c])
        # step e
        output4 = self.dense(deco_output)
        output4 = tf.squeeze(output4,axis=1)
        #print("shape afyer dense layer and softmax ",output4.shape)
        return output4,deco_state_h, deco_state_c,attention_weights,context_vector
    
class Decoder(tf.keras.Model):
    def __init__(self,out_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units):
      #Intialize necessary variables and create an object from the class onestepdecoder
        super().__init__()
        self.out_vocab_size = out_vocab_size
        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.att_units = att_units
        self.input_length = input_length
        self.score_fun = score_fun
        self.onestepdecoder = One_Step_Decoder(self.out_vocab_size,self.embedding_dim,self.input_length,self.dec_units,self.score_fun,self.att_units)
        
    @tf.function    
    def call(self, input_to_decoder,encoder_output,decoder_hidden_state,decoder_cell_state):


        all_outputs = tf.TensorArray(tf.float32,size=input_to_decoder.shape[1])
        for timestep in range(input_to_decoder.shape[1]):
            output,decoder_hidden_state,decoder_cell_state,attention_weights,context_vector=self.onestepdecoder(input_to_decoder[:,timestep:timestep+1],encoder_output,decoder_hidden_state,decoder_cell_state)
            all_outputs = all_outputs.write(timestep,output)
        # Return the tensor array
        all_outputs = tf.transpose(all_outputs.stack(),[1,0,2])
        #print("all outpt shape is ",all_outputs.shape)
        return all_outputs
    
class encoder_decoder(tf.keras.Model):
  def __init__(self,inp_vocab_size,out_vocab_size,embedding_size,lstm_size,input_length,batch_size,score_fun,att_units,*args):
    #Intialize objects from encoder decoder
    super().__init__() # https://stackoverflow.com/a/27134600/4084039
    #print("input vocab size in encoder decoder class",inp_vocab_size)
    self.encoder = Encoder(inp_vocab_size,embedding_size,lstm_size,input_length)
    #print("output vocab size in encoder decoder class",out_vocab_size)
    self.decoder = Decoder(out_vocab_size,embedding_size,input_length,lstm_size,score_fun,att_units)
    #self.dense   = Dense(out_vocab_size, activation='softmax')
    #self.flatten = Flatten()
    self.batch = batch_size
  
  def call(self,data):
    input,output = data[0], data[1]
    #Intialize encoder states, Pass the encoder_sequence to the embedding layer
    l = self.encoder.initialize_states(self.batch)
    #print("WE ARE INITIALIZING encoder WITH initial STATES as zeroes :",l[0].shape, l[1].shape)
    encoder_output,encoder_final_state_h,encoder_final_state_c = self.encoder(input,l)
    decoder_output = self.decoder(output,encoder_output,encoder_final_state_h,encoder_final_state_c)
    return decoder_output


# In[28]:


inp_vocab_size = 17067
out_vocab_size = 14826
embedding_dim=50
input_length=25
lstm_size=64
batch_size=512
score_fun = "dot"
att_units = 64
model = encoder_decoder(inp_vocab_size,out_vocab_size,embedding_dim,lstm_size,input_length,batch_size,score_fun,att_units)

optimizer = tf.keras.optimizers.Adam()
#defining custom loss function which will not consider loss for padded zeroes
# code taken from attention assignment
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)
model.compile(optimizer=optimizer,loss=loss_function)


# In[29]:


##Load the TensorBoard notebook extension
import datetime
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().system('rm -rf ./logs/')


# #### Train the model first time for 25 epochs

# #### Train the model again for 15 epochs and total of (15+25) 40 epochs

# In[30]:


log_dir = "/content/drive/MyDrive/attention_model/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=[incorr_train,corr_train_inp],y=corr_train_out, epochs=15,batch_size=512,callbacks=[tensorboard_callback])


# #### Train the model again for 15 epochs and total of (40+25) 65 epochs

# In[ ]:


log_dir = "/content/drive/MyDrive/attention_model/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=[incorr_train,corr_train_inp],y=corr_train_out, epochs=25,batch_size=512,callbacks=[tensorboard_callback])


# #### Train the model again for 15 epochs and total of (65+15) 80 epochs

# In[ ]:


log_dir = "/content/drive/MyDrive/attention_model/logs/fit2/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=[incorr_train[:360448],corr_train_inp],y=corr_train_out, epochs=15,batch_size=512,callbacks=[tensorboard_callback])


# train further for 10 epoch after 80 epochs with high learning rate

# In[ ]:


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model = encoder_decoder(inp_vocab_size,out_vocab_size,embedding_dim,lstm_size,input_length,batch_size,score_fun,att_units)
model.compile(optimizer=optimizer,loss=loss_function)
model.train_on_batch([incorr_train[:512],corr_train_inp[:512]],corr_train_out[:512])
model.load_weights('attention80')


# In[ ]:


log_dir = "/content/drive/MyDrive/attention_model/logs/fit3/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=[incorr_train,corr_train_inp],y=corr_train_out, epochs=10,batch_size=512,callbacks=[tensorboard_callback])


# train further for 10 epoch after 80 epochs with small learning rate

# In[ ]:


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model = encoder_decoder(inp_vocab_size,out_vocab_size,embedding_dim,lstm_size,input_length,batch_size,score_fun,att_units)
model.compile(optimizer=optimizer,loss=loss_function)
model.train_on_batch([incorr_train[:512],corr_train_inp[:512]],corr_train_out[:512])
model.load_weights('attention80')


# In[ ]:


log_dir = "/content/drive/MyDrive/attention_model/logs/fit4/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=[incorr_train[:360448],corr_train_inp[:360448]],y=corr_train_out[:360448], epochs=10,batch_size=512,callbacks=[tensorboard_callback])


# #### Load trained model

# In[ ]:


## code taken from https://colab.research.google.com/drive/172D4jishSgE3N7AO6U2OKAA_0wNnrMOq#scrollTo=mJqOn0snzCRy
#model.save_weights('attention80', save_format='tf')

## relaod saved model
model = encoder_decoder(inp_vocab_size,out_vocab_size,embedding_dim,lstm_size,input_length,batch_size,score_fun,att_units)
model.compile(optimizer=optimizer,loss=loss_function)
model.train_on_batch([incorr_train[:512],corr_train_inp[:512]],corr_train_out[:512])
model.load_weights('attention90low')


# In[ ]:


corr_dict = tokenizer_corr_out.word_index
inv_corr = {v: k for k, v in corr_dict.items()}
"""
def predict(input_sentence):
    
    input_sentence = tokenizer_incorr.texts_to_sequences([input_sentence])[0]
    #print(input_sentence)
    #x = len(input_sentence)
    #attention_plot = np.zeros((22,22))# here 22,22 is length of italian and english sentences
    initial_hidden_state = tf.zeros([1,64])
    initial_cell_state = tf.zeros([1,64])
    encoder_initial_state = [initial_hidden_state,initial_cell_state]
    input_sentence = tf.keras.preprocessing.sequence.pad_sequences([input_sentence],maxlen=25,padding='post')
    input_sentence = input_sentence[0]
    enc_output, enc_state_h, enc_state_c = model.layers[0](np.expand_dims(input_sentence,0),encoder_initial_state)
    #states_values = [enc_state_h, enc_state_c]
    pred = []
    sentence = []
    cur_vec = np.ones((1, 1),dtype='int')
    for i in range(26):
        #print(i)
        #print(enc_output.shape)
        infe_output,deco_state_h, deco_state_c,attention_weights,context_vector = model.layers[1].onestepdecoder(cur_vec, enc_output,enc_state_h, enc_state_c)
        enc_state_h, enc_state_c = deco_state_h, deco_state_c
        cur_vec = np.reshape(np.argmax(infe_output), (1, 1))
        #print(cur_vec)
        if inv_corr[cur_vec[0][0]] == '@':
            break
        pred.append(cur_vec[0][0])
    for i in pred:
        sentence.append(inv_corr[i])
    return " ".join(sentence)
"""


# In[ ]:


def predict(input_sentence):
    """
    this function takes incorrect input sentences as input and returns correct sentences
    """
    input_sequence = tokenizer_incorr.texts_to_sequences([input_sentence])
    
    input_sequence_padded = pad_sequences(input_sequence, maxlen=40, padding='post', truncating='post')
    
    initial_hidden_state = tf.zeros([1, 64])
    initial_cell_state = tf.zeros([1, 64])
    qwst = [initial_hidden_state, initial_cell_state]
    
    pred = []
    sentence = []
    

    enc_output, enc_state_h, enc_state_c = model.layers[0](input_sequence_padded, qwst)
    states_values = [enc_state_h, enc_state_c]
    
    cur_vec = np.array([[tokenizer_corr_inp.word_index['<start>']]])
    
    for i in range(40):
        dec_output, dec_state_h, dec_state_c = model.layers[1](cur_vec, states_values)
        infe_output = model.layers[2](dec_output)
        states_values = [dec_state_h, dec_state_c]
        cur_vec = np.reshape(np.argmax(infe_output, axis=-1), (1, 1))
        
        pred_token = cur_vec[0][0]
        if pred_token == tokenizer_corr_out.word_index['<end>']:
            break
        pred.append(pred_token)
    
    sentence = [inv_corr[token] for token in pred if token in inv_corr]
    
    return " ".join(sentence)


# #### GLUE Score on test data

# In[ ]:


"""gleu_score_test = 0
length = 1000

for i in range(length):
    reference = [test["correct"].values[i:i+1][0].split()]
    candidate = predict(test["incorrect"].values[i:i+1][0]).split()
    gleu_score_test = gleu_score_test + sentence_gleu(reference, candidate)
    #bleu_score_test = bleu_score_test + sentence_bleu(reference, candidate)
    #print(gleu_score,bleu_score)
print("Final GLEU Score on Test data are",gleu_score_test/length)"""


# In[ ]:


from nltk.translate.gleu_score import sentence_gleu

gleu_score_test = 0
length = 100

def clean_sentence(sentence):
    return [word for word in sentence if word]

for i in range(length):
    reference = [cv["output"].values[i:i+1][0].split()]
    reference = [clean_sentence(sentence) for sentence in reference]    
    corr_texts.append(reference)
    
    origin = [cv["input"].values[i:i+1][0].split()]
    origin = [clean_sentence(sentence) for sentence in origin]    
    orig_texts.append(origin)
    
    #print(reference)
    candidate = predict(cv["input"].values[i:i+1][0]).split()
    predicted_texts.append(candidate)
    #candidate = clean_sentence(candidate)
    #print(candidate)
   
    gleu_score = sentence_gleu(reference, candidate, min_len=1, max_len=1)
    gleu_score_test += gleu_score
    
    #gleu_score_test = gleu_score_test + sentence_gleu(reference, candidate)
    #print(gleu_score_test)
print("Final GLEU Score on Test data are",gleu_score_test/length)


# In[ ]:


orig_file_path = "/kaggle/input/result-15/cor.txt"  
cor_file_path = "/kaggle/input/result-15/cor.txt"  
ref_file_path = "/kaggle/input/result-15/ref.txt"


with open(orig_file_path, 'r') as file:
    data1 = file.read()
with open("/kaggle/working/orig.txt", 'w') as file:
    file.write(data1)
with open(cor_file_path, 'r') as file:
    data2 = file.read()
with open("/kaggle/working/cor.txt", 'w') as file:
    file.write(data2)
with open(ref_file_path, 'r') as file:
    data3 = file.read()
with open("/kaggle/working/ref.txt", 'w') as file:
    file.write(data3)


orig_file_path = "/kaggle/working/orig.txt"  
cor_file_path = "/kaggle/working/cor.txt"  
ref_file_path = "/kaggle/working/ref.txt"

import subprocess
get_ipython().system('pip install errant')
def calculate_errant_scores(orig_path, cor_path, ref_path):

    cor_m2_path = cor_path.replace('.txt', '.m2')
    subprocess.run(['errant_parallel', '-orig', orig_path, '-cor', cor_path, '-out', cor_m2_path], check=True)

    ref_m2_path = ref_path.replace('.txt', '.m2')
    subprocess.run(['errant_parallel', '-orig', orig_path, '-cor', ref_path, '-out', ref_m2_path], check=True)

    result = subprocess.run(['errant_compare', '-hyp', cor_m2_path, '-ref', ref_m2_path], capture_output=True, text=True)
  
    print(result.stdout)
    return result.stdout
calculate_errant_scores(orig_file_path, cor_file_path, ref_file_path)



# #### Prediting results on train data

# In[ ]:


#predicted sentences
for i in train["incorrect"].values[:10]:
  print(predict(i))


# In[ ]:


#actual sentences
train["correct"].values[:10]


# #### Prediting results on CV data

# In[ ]:


#predicted sentences
for i in cv["incorrect"].values[:10]:
  print(predict(i))


# In[ ]:


#actual sentences
cv["correct"].values[:10]


# #### Prediting results on test data

# In[ ]:


#predicted sentences
for i in test["incorrect"].values[:10]:
  print(predict(i))


# In[ ]:


#actual sentences
test["correct"].values[:10]

