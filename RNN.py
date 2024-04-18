#!/usr/bin/env python
# coding: utf-8

# python3.11.6

# In[2]:


import argparse
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pandas as pd
import re
import os
import math
import random
import numpy as np
from wordcloud import WordCloud,STOPWORDS
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import pickle
import nltk.translate.bleu_score as bleu
from IPython.display import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.tensorboard import SummaryWriter
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from collections import Counter
import torch.utils.data as data


# In[3]:


def process_m2file(m2_file, out_file, id=0):

  m2 = open(m2_file).read().strip().split("\n\n")
  out = open(out_file, "w")
  # Do not apply edits with these error types
  skip = {"noop", "UNK", "Um"}
  total_edits = []

  for sent in m2:
    sent = sent.split("\n")
    cor_sent = sent[0].split()[1:] # Ignore "S "
    edits = sent[1:]
    offset = 0
    cur_edit = 0
    for edit in edits:
      edit = edit.split("|||")
      if edit[1] in skip: continue # Ignore certain edits
      coder = int(edit[-1])
      if coder != id: continue # Ignore other coders
      span = edit[0].split()[1:] # Ignore "A "
      start = int(span[0])
      end = int(span[1])
      cor = edit[2].split()
      cor_sent[start+offset:end+offset] = cor
      cur_edit+=1
      offset = offset-(end-start)+len(cor)
    total_edits.append(cur_edit)
    out.write(" ".join(cor_sent)+"\n")
  out.close()

  file1 = open(m2_file,"r")
  s1 = file1.read()

  each_sent = s1.split("\n\n")

  incorrect = []
  for i in range(len(each_sent)):
      temp = each_sent[i].split("\n")
      temp = temp[0]
      temp = temp.split(" ")
      temp = temp[1:]# ignore S
      temp = ' '.join(temp)
      incorrect.append(temp)

  #preprocessing correct sentences

  file2 = open(out_file, "r")
  s2 = file2.read()

  correct = s2.split("\n")

  print("Preparing dataframe")
  df = pd.DataFrame()
  df["correct"] = correct
  df["incorrect"] = incorrect
  total_edits.append(0)
  df['total_edits'] = total_edits

  return df


# In[4]:


train_m2_file = "/kaggle/input/grammar-error-correction/ABC.train.gold.bea19.m2"
train_out_file = "/kaggle/working/ABC.train.gold.bea19.txt"

train_df = process_m2file(train_m2_file, train_out_file, id=0)

print(train_df.shape)
print(train_df.head())


# In[5]:


dev_m2_file = "/kaggle/input/grammar-error-correction/ABCN.dev.gold.bea19.m2"
dev_out_file = "/kaggle/working/ABCN.dev.gold.bea19.txt"

dev_df = process_m2file(dev_m2_file, dev_out_file, id=0)

print(dev_df.shape)
print(dev_df.head())


# In[6]:


train_df = train_df[train_df['correct'].str.len() > 1]
print(train_df.shape)


# In[7]:


train_df[train_df.duplicated(keep=False)].sort_values('correct')


# In[8]:


train_df = train_df.drop_duplicates(keep='first')
print(train_df.shape)


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))
sns.histplot(train_df['correct'].str.len(), kde=True)
plt.title('Distribution of sentence length')
plt.xlabel('Length of sentence')
plt.ylabel('Number of sentences')
plt.show()


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))

sns.histplot(train_df['total_edits'], bins=30)
plt.title('Distribution of total edits',fontsize=16,fontweight='bold')
plt.xlabel('Total edits',fontsize=16)
plt.ylabel('Number of sentences',fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.show()


# In[11]:


def pre_processing(df):
  # Removing null values and duplicates
  df = df.dropna()
  df = df[df.notnull()]
  df.astype(str)
  df.reset_index(inplace = True)

  # Get incorrect and correct sentences
  inp1 = df['incorrect']
  tgt1 = df['correct']
  data = pd.DataFrame()
  data['y'] = list('1'*len(inp1))
  data['input'] = inp1
  data['output'] = tgt1

  return data


# In[12]:


train = pre_processing(train_df)
dev = pre_processing(dev_df)


# In[13]:


print(train)
print(train.describe())


# In[14]:


print(dev)
print(dev.describe())


# In[15]:


def process(text):
  #text = text.astype(str)
  text = re.sub('<.*>', '', text)
  text = re.sub('\(.*\)', '', text)
  text = re.sub('\[.*\]', '', text)
  text = re.sub('{.*}', '', text)
  text = re.sub("[-+@#^/|*(){}$~`<>=_]","",text)
  text = text.replace("\\","")
  text = re.sub("\[","",text)
  text = re.sub("\]","",text)
  text = re.sub("[0-9]","",text)
  return text

def decontract(text):
  #text = text.astype(str)
  text = re.sub(r"won\'t", "will not", text)
  text = re.sub(r"can\'t", "can not", text)
  text = re.sub(r"n\'t", " not", text)
  text = re.sub(r"\'re", " are", text)
  text = re.sub(r"\'s", " is", text)
  text = re.sub(r"\'d", " would", text)
  text = re.sub(r"\'ll", " will", text)
  text = re.sub(r"\'t", " not", text)
  text = re.sub(r"\'ve", " have", text)
  text = re.sub(r"\'m", " am", text)
  return text

def preprocess(text):
  text = re.sub("\n","",text)
  text = process(text)
  text = decontract(text)
  text = text.lower()
  return text

def clean(df):
    df["enc_input"] = df.input.apply(preprocess)
    df["dec_input"] = df.input.apply(preprocess)
    df["dec_output"] = df.output.apply(preprocess)
    df = df.drop(columns=["input", "output"])
    #df["input"] = df.input.apply(preprocess)
    #df["output"] = df.output.apply(preprocess)
    return df


# In[16]:


train = clean(train)
dev = clean(dev)


# In[17]:


train.head()
#here input is the wrong sentence and output is the correct sentence


# In[18]:


dev.head()


# ## Tokenization

# In[19]:


dev


# In[20]:


train_token = train.copy()
dev_token = dev.copy()


# In[21]:


# Add <start> and <end> tokens to the dec_input column
train_token["dec_input"] = train_token["dec_input"].apply(lambda x: "<start> " + x)
dev_token["dec_input"] = dev_token["dec_input"].apply(lambda x: "<start> " + x)

# Add <start> and <end> tokens to the dec_output column
train_token["dec_output"] = train_token["dec_output"].apply(lambda x: x + " <end>")
dev_token["dec_output"] = dev_token["dec_output"].apply(lambda x: x + " <end>")


# In[22]:


#split into train and test
train_token, test_token = train_test_split(train_token, test_size=0.1, random_state=42)


# In[23]:


def count_token_occurrences(tokens_list):
  counter = Counter()
  for sentence_tokens in tokens_list:
    counter.update(sentence_tokens)
  return counter

# Define the tokenizer function (use basic_english tokenizer)
tokenizer = get_tokenizer('basic_english')
traindata_in = train_token.dec_input.apply(str).tolist()
traindata_out = train_token.dec_output.apply(str).tolist()

# Tokenization and preprocessing for encoder input
enc_input_tokens = [tokenizer(sentence) for sentence in traindata_in]

# Tokenization and preprocessing for decoder input
dec_input_tokens = [tokenizer(sentence) for sentence in traindata_out]

# Build vocabulary for encoder input
counter_enc = count_token_occurrences(enc_input_tokens)
tk_inp = build_vocab_from_iterator(enc_input_tokens,specials=['<pad>','<unk>'])
# Build vocabulary for decoder input
counter_dec = count_token_occurrences(dec_input_tokens)
tk_out = build_vocab_from_iterator(dec_input_tokens, specials=['<pad>', '<sos>', '<eos>', '<unk>'])


# In[24]:


input_vocab_size = len(tk_inp.get_stoi())
output_vocab_size = len(tk_out.get_stoi())

print("Input Vocabulary Size:", input_vocab_size)
print("Output Vocabulary Size:", output_vocab_size)


# ## **Text data into integer sequences**
# 
# We now try to convert the text data into integer sequences wich also has a padding. This padding of sequences is necessary to ensure that all sequences in a batch have the same length. Padding adds special tokens (pad token) to the sequences to that all sequences pocesses the same amount of tokens.

# In[25]:


class conv_dataset(TorchDataset):
  def __init__(self, data, tk_inp, tk_out, max_len):
    self.encoder_in = data["enc_input"].apply(str).values
    self.decoder_in = data["dec_input"].apply(str).values
    self.decoder_out = data["dec_output"].apply(str).values
    self.tk_inp = tk_inp.get_stoi()
    self.tk_out = tk_out.get_stoi()
    self.tokenizer = get_tokenizer('basic_english')
    self.max_len = max_len
    self.unk_index_inp = self.tk_inp.get('<unk>', -1)  # 如果没有 '<unk>', 使用 -1 作为占位符
    self.unk_index_out = self.tk_out.get('<unk>', -1)

  def __getitem__(self, i):
    # Input sequences
    encoder_seq = self.encoder_in[i]
    encoder_tokens = self.tokenizer(encoder_seq)
    encoder_indices = [self.tk_inp.get(token, self.unk_index_inp) for token in encoder_tokens]
    encoder_tensor = torch.tensor(encoder_indices)

    # Input encoder sequences
    decoder_in_seq = self.decoder_in[i]
    decoder_in_tokens = self.tokenizer(decoder_in_seq)
    # Special handling for <start> and <end> tokens
    decoder_in_indices = [self.tk_inp.get(token, self.unk_index_inp) for token in decoder_in_tokens]
    decoder_in_tensor = torch.tensor(decoder_in_indices)

    # Input decoder sequences
    decoder_out_seq = self.decoder_out[i]
    decoder_out_tokens = self.tokenizer(decoder_out_seq)
    # Special handling for <start> and <end> tokens
    decoder_out_indices = [self.tk_out.get(token, self.unk_index_out) for token in decoder_out_tokens]
    decoder_out_tensor = torch.tensor(decoder_out_indices)

    # Tokenizer padding
    encoder_tensor = F.pad(encoder_tensor, pad=(0, self.max_len - len(encoder_tensor))).float()
    decoder_in_tensor = F.pad(decoder_in_tensor, pad=(0, self.max_len - len(decoder_in_tensor))).float()
    decoder_out_tensor = F.pad(decoder_out_tensor, pad=(0, self.max_len - len(decoder_out_tensor)))

    return encoder_tensor, decoder_in_tensor, decoder_out_tensor

  def __len__(self):
    return len(self.encoder_in)


# In[26]:


train_data = conv_dataset(train_token, tk_inp, tk_out, 40)
dev_data = conv_dataset(test_token, tk_inp, tk_out, 40)
test_data = conv_dataset(dev_token, tk_inp, tk_out, 40)


# In[27]:


dev_data


# ## RNN

# In[28]:


class EarlyStopping:
    """早停以防止过拟合"""
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f'Score improved. New best score: {score:.4f}')
                
import subprocess


# In[29]:


class RNN(nn.Module):

  def __init__(self, input_size, embedding_size, hidden_size, num_layers, vocab_size):
    super(RNN, self).__init__()
    self.input_size = input_size
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.vocab_size = vocab_size

    # Layers of the model
    # -> x.shape() = (batch_size, seq, input_size)
    self.embedding = nn.Embedding(input_size, embedding_size)
    self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, vocab_size)

  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
    embeds = self.embedding(x)
    out, h_out = self.rnn(embeds, h0)
    out = self.fc(out) # use the entire output for correction
    return out


# In[30]:


def indices_to_text(indices_seq, vocab):
    # vocab.itos是索引到字符串的映射
    return ' '.join([vocab.itos[index] for index in indices_seq])

"""def predict(model, data_loader, vocab, filepath_pred, filepath_true):
    model.eval()  # 将模型设置为评估模式
    with open(filepath_pred, 'w', encoding='utf-8') as f_pred, open(filepath_true, 'w', encoding='utf-8') as f_true:
        with torch.no_grad():
            for enc_input, _, dec_output in data_loader:
                enc_input, dec_output = enc_input.to(device), dec_output.to(device)
                inputs = enc_input.long()

                # 进行预测
                outputs = model(inputs)
                _, predicted_indices = torch.max(outputs, dim=2)

                # 处理每个预测结果
                for idx in range(inputs.size(0)):
                    # 将索引转换回文本
                    predicted_text = ' '.join([vocab.itos[i] for i in predicted_indices[idx].cpu().numpy()])
                    true_text = ' '.join([vocab.itos[i] for i in dec_output[idx].cpu().numpy()])

                    # 写入文件
                    f_pred.write(predicted_text + '\n')
                    f_true.write(true_text + '\n')"""



def save_dev_texts(data_loader, vocab, filepath_orig, filepath_corr):
    itos = vocab.get_itos()  # 获取索引到字符串的映射
    with open(filepath_orig, 'w', encoding='utf-8') as f_orig, open(filepath_corr, 'w', encoding='utf-8') as f_corr:
        for enc_input, _, dec_output in data_loader:
            for idx in range(enc_input.size(0)):
                original_text = ' '.join([itos[int(i)] for i in enc_input[idx].cpu().numpy() if i < len(itos)])
                corrected_text = ' '.join([itos[int(i)] for i in dec_output[idx].cpu().numpy() if i < len(itos)])

                f_orig.write(original_text + '\n')
                f_corr.write(corrected_text + '\n')


# In[31]:


# Model Hipper parameters
num_layers = 3
learning_rate = 0.001
num_epochs = 10

# x hipper parameters
batch_size = 256
input_size = 40
sequence_length = 40
output_size = 40
hidden_size = 256 # encoding units

vocab_size = len(tk_inp)
num_classes = vocab_size
embedding_size = 150

# Momentum
beta1 = 0.1  # Momentum value for the momentum term in Adam
beta2 = 0.1  # Value for the squared gradient term in Adam
early_stopping = EarlyStopping(patience=10, verbose=True, delta=0.001)


# In[32]:


# Model RNN instance
import torch
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_rnn = RNN(input_size = vocab_size,
                embedding_size = embedding_size,
                hidden_size = hidden_size,
                num_layers = num_layers,
                vocab_size = vocab_size).to(device)

# Loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_rnn.parameters(),
                             lr = learning_rate
                             #,betas=(beta1, beta2)
                            )


# In[33]:


def evaluate(model, loader, criterion):
    model.eval()  # 设置为评估模式
    total_loss = 0
    total_samples = 0

    with torch.no_grad(): 
        for batch_data in loader:
            enc_input, _, dec_output = batch_data
            enc_input, dec_output = enc_input.to(device), dec_output.to(device)

            outputs = model(enc_input.long())
            loss = criterion(outputs.view(-1, vocab_size), dec_output.view(-1))
            total_loss += loss.item() * enc_input.size(0)
            total_samples += enc_input.size(0)

    average_loss = total_loss / total_samples
    model.train()  # 重置为训练模式
    return average_loss


# In[34]:


#Training procedure
import torch
torch.cuda.empty_cache()

train_loader = DataLoader(batch_size=8, dataset=train_data, shuffle=True)
train_DL = train_loader

n_total_steps = len(train_DL)
loss_history_epochs = []
loss_history_batches = []

dev_loader = DataLoader(batch_size=8, dataset=dev_data, shuffle=True)
dev_DL = dev_loader
save_dev_texts(dev_loader, tk_out, 'dev_orig.txt', 'dev_corr.txt')

for epoch in range(num_epochs):
    model_rnn.train()  # 明确设置为训练模式
    print("Training mode set for epoch:", epoch + 1)
    total_loss = 0
    for i, batch_data in enumerate(train_DL):
        enc_input, dec_input, dec_output = batch_data
        enc_input, dec_input, dec_output = enc_input.to(device), dec_input.to(device), dec_output.to(device)

        # Forward pass
        outputs = model_rnn(enc_input.long())
        # Loss and Backpropagation
        loss = criterion(outputs.view(-1, vocab_size), dec_output.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 100 == 0:
            avg_loss = total_loss / input_size
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {avg_loss:.4f}')
            total_loss = 0
            loss_history_batches.append(avg_loss)
    model_rnn.eval()  # 切换到评估模式进行评估
    print("Evaluation mode set for validation")
    dev_loss = evaluate(model_rnn, dev_DL, criterion)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Dev Loss: {dev_loss:.4f}')
    model_rnn.train()  # 评估完成后再次切换回训练模式
    print("Training mode reset after validation")

    
    torch.save(model_rnn.state_dict(), f'model_rnn_{epoch}.pth')
    loss_history_epochs.append(avg_loss)
    print(f'Model saved after epoch {epoch + 1}')


# In[35]:


def predict_example(model, sentence, vocab_inp, vocab_out):
    model.eval()
    if sentence.strip() == '':
        return ''
    # 假设sentence是一个字符串，需要分词并转换为索引
    tokens = [vocab_inp.get_stoi()[word] if word in vocab_inp.get_stoi() else vocab_inp.get_stoi()['<unk>'] for word in sentence.split()]
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # 增加一个批次维度
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_indices = torch.max(outputs, dim=2)
        predicted_seq = ' '.join([vocab_out.get_itos()[i] for i in predicted_indices[0].cpu().numpy()])
    return predicted_seq


# In[36]:


example_sentences = ["i was will be there"]

predicted_correction = predict_example(model_rnn, example_sentences[0], tk_inp, tk_out)
print("Original:", example_sentences[0])
print("Predicted Correction:", predicted_correction)


# In[37]:


example_sentences = [""]

predicted_correction = predict_example(model_rnn, example_sentences[0], tk_inp, tk_out)
print("Original:", example_sentences[0])
print("Predicted Correction:", predicted_correction)


# In[38]:


def clean_sentence(sentence):
    # 假设句子已经是分词后的列表，如果不是，需要先进行分词处理
    return " ".join([word for word in sentence.split() if word])  # 过滤掉空的单词并重新组合为字符串

# 应用这个函数到每一行
dev['dec_input'] = dev['dec_input'].apply(clean_sentence)
with open('dev_ori.txt', 'w', encoding='utf-8') as file:
    for text in dev['dec_input']:
        file.write(text + '\n')
        
dev['dec_output'] = dev['dec_output'].apply(clean_sentence)
with open('dev_ref.txt', 'w', encoding='utf-8') as file:
    for text in dev['dec_output']:
        file.write(text + '\n')    

def predict(model, dataframe, column_name, file_path, tokenizer_input, tokenizer_output):
    with open(file_path, 'w', encoding='utf-8') as file:
        for index, row in dataframe.iterrows():
            original_sentence = row[column_name]
            predicted_correction = predict_example(model, original_sentence, tokenizer_input, tokenizer_output)
            file.write(predicted_correction + '\n')
            #print(f"Processed {index + 1}/{len(dataframe)}")

predict(model_rnn, dev, 'dec_input', 'dev_pred.txt', tk_inp, tk_out)


# #### Gleu Score

# In[40]:


import torch
from nltk.translate.gleu_score import sentence_gleu

def calculate_gleu(pred_path, true_path):
    with open(pred_path, 'r', encoding='utf-8') as f_pred, open(true_path, 'r', encoding='utf-8') as f_true:
        pred_sentences = f_pred.readlines()
        true_sentences = f_true.readlines()

        gleu_scores = []
        for pred, true in zip(pred_sentences, true_sentences):
            pred_words = pred.strip().split()
            true_words = true.strip().split()
            gleu_score = sentence_gleu([true_words], pred_words, min_len=1, max_len=1)
            gleu_scores.append(gleu_score)

    return sum(gleu_scores) / len(gleu_scores) if gleu_scores else 0

# 计算GLEU
average_gleu = calculate_gleu('dev_pred.txt', 'dev_ref.txt')
print(f"Average GLEU Score: {average_gleu:.4f}")


# #### Errant Score

# In[46]:


get_ipython().system('pip install errant')
import subprocess
def calculate_errant(orig_file, ref_file, pred_file):
    pred_m2 = pred_file + ".m2"
    subprocess.run(['errant_parallel', '-orig', orig_file, '-cor', pred_file, '-out', pred_m2], check=True)

    ref_m2 = ref_file + ".m2"
    subprocess.run(['errant_parallel', '-orig', orig_file, '-cor', ref_file, '-out', ref_m2], check=True)

    result = subprocess.run(['errant_compare', '-hyp', pred_m2, '-ref', ref_m2], capture_output=True, text=True)

    print(result.stdout)
    
calculate_errant('dev_ori.txt','dev_ref.txt','dev_pred.txt')

