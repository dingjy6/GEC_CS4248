# %%
import argparse
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import pandas as pd
import re
import shutil
import io
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense,RNN,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import TimeDistributed
import numpy as np
from wordcloud import WordCloud,STOPWORDS
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import pickle
import nltk.translate.bleu_score as bleu
import subprocess
from IPython.display import Image
!pip install errant

# %%
train = pd.read_csv("/kaggle/input/preprocessing-lan-data/train.csv",dtype=str)
cv = pd.read_csv("/kaggle/input/preprocessing-lan-data/val.csv",dtype=str)
test = pd.read_csv("/kaggle/input/preprocessing-lan-data/dev.csv",dtype=str)

# %%
train.head()

# %%
train["correct_inp"] = "<start> " + train["output"].astype(str)
train["correct_out"] = train["output"].astype(str) + " <end>"

cv["correct_inp"] = "<start> " + cv["output"].astype(str)
cv["correct_out"] = cv["output"].astype(str) + " <end>"

test["correct_inp"] = "<start> " + test["output"].astype(str)
test["correct_out"] = test["output"].astype(str) + " <end>"

# %%
train.head()

# %%
train.dropna(subset=["input"], inplace=True) 
cv.dropna(subset=["input"], inplace=True)  
test.dropna(subset=["input"], inplace=True)

# %%
tokenizer_incorr = Tokenizer(filters="",lower=False)
tokenizer_incorr.fit_on_texts(train["input"].values)

# %%
tokenizer_corr_inp = Tokenizer(filters="",lower=False)
tokenizer_corr_inp.fit_on_texts(train["correct_inp"].values)

# %%
tokenizer_corr_out = Tokenizer(filters="",lower=False)
tokenizer_corr_out.fit_on_texts(train["correct_out"].values)

# %%
#save keras tokenizer

with open("tokenizer_incorr_word.pickle","wb") as temp1:
     pickle.dump(tokenizer_incorr,temp1)
    
with open("tokenizer_corr_inp_word.pickle","wb") as temp2:
    pickle.dump(tokenizer_corr_inp,temp2)
    
with open("tokenizer_corr_out_word.pickle","wb") as temp3:
    pickle.dump(tokenizer_corr_out,temp3)


# %%
# loading saved tokenizer
with open("tokenizer_incorr_word.pickle","rb") as temp1:
    tokenizer_incorr = pickle.load(temp1)
    
with open("tokenizer_corr_inp_word.pickle","rb") as temp2:
    tokenizer_corr_inp = pickle.load(temp2)
    
with open("tokenizer_corr_out_word.pickle","rb") as temp3:
    tokenizer_corr_out = pickle.load(temp3)

# %%
incorr_train = tokenizer_incorr.texts_to_sequences(train["input"].values)
incorr_cv = tokenizer_incorr.texts_to_sequences(cv["input"].values)
print("vocab size of incorrrect sentences is",len(tokenizer_incorr.word_index))

# %%
corr_train_inp = tokenizer_corr_inp.texts_to_sequences(train["correct_inp"].values)
corr_cv_inp = tokenizer_corr_inp.texts_to_sequences(cv["correct_inp"].values)
print("vocab size of corrrect sentences is",len(tokenizer_corr_inp.word_index))

# %%
corr_train_out = tokenizer_corr_out.texts_to_sequences(train["correct_out"].values)
corr_cv_out = tokenizer_corr_out.texts_to_sequences(cv["correct_out"].values)
print("vocab size of corrrect sentences is",len(tokenizer_corr_out.word_index))

# %%
incorr_train = np.array(pad_sequences(incorr_train,maxlen=40,padding="post",truncating='post'))
corr_train_inp = np.array(pad_sequences(corr_train_inp,maxlen=40,padding="post",truncating='post'))
corr_train_out = np.array(pad_sequences(corr_train_out,maxlen=40,padding="post",truncating='post'))

incorr_cv = np.array(pad_sequences(incorr_cv,maxlen=40,padding="post",truncating='post'))
corr_cv_inp = np.array(pad_sequences(corr_cv_inp,maxlen=40,padding="post",truncating='post'))
corr_cv_out = np.array(pad_sequences(corr_cv_out,maxlen=40,padding="post",truncating='post'))

# %%
embedding_matrix1 = np.load("/kaggle/input/pretrain/incorr_matrix.npy")

# %%
embedding_matrix2 = np.load("/kaggle/input/pretrain/corr_matrix.npy")

# %%

############################## Encoder class #############################################################

class Encoder(tf.keras.layers.Layer):
    '''
    Encoder model -- That takes a input sequence and returns encoder-outputs,encoder_final_state_h,encoder_final_state_c
    '''

    def __init__(self,inp_vocab_size,embedding_size,lstm_size,input_length):
        
        super().__init__()
        self.lstm_size = lstm_size
        self.embedding = Embedding(input_dim=inp_vocab_size, output_dim=300, input_shape=(input_length,),
                           mask_zero=True,name="embedding_layer_encoder",trainable=False)
        self.embedding.build(input_shape=(None,))
        self.embedding.set_weights([embedding_matrix1])
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
    
class Decoder(tf.keras.layers.Layer):

    def __init__(self,out_vocab_size,embedding_size,lstm_size,input_length):

        super().__init__()
        self.lstm_size = lstm_size
        self.embedding = Embedding(input_dim=out_vocab_size, output_dim=300,input_shape=(input_length,),
                           mask_zero=True,name="embedding_layer_encoder",trainable=False)
        self.embedding.build(input_shape=(None,))
        self.embedding.set_weights([embedding_matrix2])
        self.lstmcell = tf.keras.layers.LSTMCell(lstm_size)
        self.decoder_lstm = RNN(self.lstmcell,return_sequences=True, return_state=True)

    def call(self,target_sequence,initial_states):

        output2 = self.embedding(target_sequence)
        deco_output, deco_state_h, deco_state_c = self.decoder_lstm(output2, initial_state=initial_states)
      
        return deco_output, deco_state_h, deco_state_c
    
##############################encoder decoder class#############################################################
    
qw_state = 0
class Encoder_decoder(tf.keras.Model):
    
    def __init__(self,inp_vocab_size,out_vocab_size,embedding_size,lstm_size,input_length,batch_size,*args):
        
        super().__init__()
        self.encoder = Encoder(inp_vocab_size,embedding_size,lstm_size,input_length)
        self.decoder = Decoder(out_vocab_size,embedding_size,lstm_size,input_length)
        self.dense   = Dense(out_vocab_size)#, activation='softmax')
        self.batch = batch_size
    
    
    def call(self,data,*args):
        
        input,output = data[0], data[1]
        # initializing initial states of encoder
        l = self.encoder.initialize_states(self.batch)
        encoder_output,encoder_final_state_h,encoder_final_state_c = self.encoder(input,l)
        m = list((encoder_final_state_h,encoder_final_state_c))
        decoder_output,decoder_final_state_h,decoder_final_state_c = self.decoder(output,m)
        qw_output = self.dense(decoder_output)
        return qw_output

# %%

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    """ Custom loss function that will not consider the loss for padded zeros.
    in this loss function we are ignoring the loss
    for the padded zeros. i.e when the input is zero then we do not need to worry what the output is. 
    This padded zeros are added from our end
    during preprocessing to make equal length for all the sentences.

    """
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# %%
from tensorflow.keras.optimizers import Nadam
inp_vocab_size = len(tokenizer_incorr.word_index)+1
out_vocab_size = len(tokenizer_corr_out.word_index)+1
embedding_dim=300 
input_length=40 #input length of each sentence
lstm_size= 256
batch_size=256
model = Encoder_decoder(inp_vocab_size,out_vocab_size,embedding_dim,lstm_size,input_length,batch_size)
model.compile(optimizer=Nadam(learning_rate=0.001),loss=loss_function)

# %%
train_trunc_idx = (incorr_train.shape[0]//batch_size)*batch_size 
val_trunc_idx = (incorr_cv.shape[0]//batch_size)*batch_size 

train_enc_inp_truncated = incorr_train[:train_trunc_idx]
train_dec_inp_truncated = corr_train_inp[:train_trunc_idx]
train_dec_out_truncated = corr_train_out[:train_trunc_idx]

val_enc_inp_truncated = incorr_cv[:val_trunc_idx]
val_dec_inp_truncated = corr_cv_inp[:val_trunc_idx]
val_dec_out_truncated = corr_cv_out[:val_trunc_idx]

# %%
import datetime
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1,write_graph=True)
earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1,mode='min')
reducelr = ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.9,mode='min')

model.fit(x=[train_enc_inp_truncated,train_dec_inp_truncated],y=train_dec_out_truncated,validation_data=([val_enc_inp_truncated,val_dec_inp_truncated],val_dec_out_truncated),epochs=60,batch_size=batch_size,callbacks=[tensorboard_callback,earlystop,reducelr])

# %%
model.summary()

# %%
corr_dict = tokenizer_corr_out.word_index
inv_corr = {v: k for k, v in corr_dict.items()}

# %%
def predict(input_sentence):
    """
    this function takes incorrect input sentences as input and returns correct sentences
    """
    # transform the sentence into sequence
    input_sequence = tokenizer_incorr.texts_to_sequences([input_sentence])
    
    # padding the sequence
    input_sequence_padded = pad_sequences(input_sequence, maxlen=40, padding='post', truncating='post')
    
    initial_hidden_state = tf.zeros([1, 256])
    initial_cell_state = tf.zeros([1, 256])
    qwst = [initial_hidden_state, initial_cell_state]
    
    pred = []
    sentence = []
    
    # input padding sequence
    enc_output, enc_state_h, enc_state_c = model.layers[0](input_sequence_padded, qwst)
    states_values = [enc_state_h, enc_state_c]
    
    cur_vec = np.array([[tokenizer_corr_inp.word_index['<start>']]])
    
    for i in range(41):
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


# %%

from nltk.translate.gleu_score import sentence_gleu

gleu_score_test = 0
length = 100

def clean_sentence(sentence):
    return [word for word in sentence if word]

for i in range(length):
    reference = [test["output"].values[i:i+1][0].split()]
    reference = [clean_sentence(sentence) for sentence in reference]    
    #corr_texts.append(reference)
    
    origin = [test["input"].values[i:i+1][0].split()]
    origin = [clean_sentence(sentence) for sentence in origin]    
    #orig_texts.append(origin)
    
    #print(reference)
    candidate = predict(test["input"].values[i:i+1][0]).split()
    #predicted_texts.append(candidate)
    #candidate = clean_sentence(candidate)
    #print(candidate)
   
    gleu_score = sentence_gleu(reference, candidate, min_len=1, max_len=1)
    gleu_score_test += gleu_score
    
    #gleu_score_test = gleu_score_test + sentence_gleu(reference, candidate)
    #print(gleu_score_test)
print("Final GLEU Score on Test data are",gleu_score_test/length)


# %%
test =test[test["input"].apply(lambda x: isinstance(x, str) and len(x) > 1)]

# %%
test = test.dropna().reset_index(drop=True)

# %%
orig_file_path = "/kaggle/working/orig.txt"  #orginal sentence
cor_file_path = "/kaggle/working/cor.txt"  # predict sentence
ref_file_path = "/kaggle/working/ref.txt"  # correct sentence

# write the file
with open(orig_file_path, "w", encoding="utf-8") as orig_f, \
     open(cor_file_path, "w", encoding="utf-8") as cor_f, \
     open(ref_file_path, "w", encoding="utf-8") as ref_f:
    for index, row in test.iterrows():
        # get the original sentence
        orig_sentence = row["input"]
        ref_sentence = row["output"]
        
        # predice
        cor_sentence = predict(orig_sentence)

        # write to the file
        orig_f.write(orig_sentence + "\n")
        cor_f.write(cor_sentence + "\n")
        ref_f.write(ref_sentence + "\n")
        
def evaluate_with_errant():

    annotate_command = ["errant_parallel", "-orig", "orig.txt", "-cor", "corr.txt", "-out", "annotated.m2"]
    annotate_result = subprocess.run(annotate_command, capture_output=True, text=True)
    print("Annotation Result:", annotate_result.stdout)

    compare_command = ["errant_compare", "-hyp", "sys_out.txt", "-ref", "annotated.m2"]
    compare_result = subprocess.run(compare_command, capture_output=True, text=True)
    print("Comparison Result:", compare_result.stdout)
    
evaluate_with_errant()
import subprocess

def calculate_errant_scores(orig_path, cor_path, ref_path):

    # generate m2 file
    #predict
    cor_m2_path = cor_path.replace('.txt', '.m2')
    subprocess.run(['errant_parallel', '-orig', orig_path, '-cor', cor_path, '-out', cor_m2_path], check=True)
    
    # correct
    ref_m2_path = ref_path.replace('.txt', '.m2')
    subprocess.run(['errant_parallel', '-orig', orig_path, '-cor', ref_path, '-out', ref_m2_path], check=True)
    
    # errant score
    result = subprocess.run(['errant_compare', '-hyp', cor_m2_path, '-ref', ref_m2_path], capture_output=True, text=True)
    
    # print and return
    print(result.stdout)
    return result.stdout

# calculate errant score

calculate_errant_scores(orig_file_path, cor_file_path, ref_file_path)

# %%
#predicted sentences for 10 sentences
for input_sentence, correct_sentence in zip(train["input"].values[:10], train["output"].values[:10]):
    predicted_sentence = predict(input_sentence)
    print("input:", input_sentence)
    print("predict:", predicted_sentence)
    print("correct:", correct_sentence)
    print("\n")  # Adds a newline for better readability between entries

# %%
#predicted sentences for 10 sentences
for input_sentence, correct_sentence in zip(test["input"].values[:10], test["output"].values[:10]):
    predicted_sentence = predict(input_sentence)
    print("input:", input_sentence)
    print("predict:", predicted_sentence)
    print("correct:", correct_sentence)
    print("\n")  # Adds a newline for better readability between entries

# %%
test_sentence = ['I ate a apple','That is a good news','I loves my dog']
for i in test_sentence:
  print(predict(i))


