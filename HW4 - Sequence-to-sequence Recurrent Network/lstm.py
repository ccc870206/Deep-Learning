from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from os import system
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import json
import os
from tensorboardX import SummaryWriter




"""========================================================================================
The sample.py includes the following template functions:

1. Encoder, decoder
2. Training function
3. BLEU-4 score function

You have to modify them to complete the lab.
In addition, there are still other functions that you have to 
implement by yourself.

1. Your own dataloader (design in your own way, not necessary Pytorch Dataloader)
2. Output your results (BLEU-4 score, correction words)
3. Plot loss/score
4. Load/save weights
========================================================================================"""

folder_name = 'lstm_t0.5_newsymbol_hidden_dropout0.2/'
params_save_dir = './experiments/'+folder_name
log_dir = './logs/'+folder_name
if not os.path.isdir(params_save_dir):
    os.makedirs(params_save_dir)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)


cuda = True if torch.cuda.is_available() else False
GPUID = [0]
torch.cuda.set_device(GPUID[0])
SOS_token = 0
EOS_token = 1
#----------Hyper Parameters----------#
hidden_size = 256
#The number of vocabulary
vocab_size = 29
teacher_forcing_ratio = 0.5
dropout_ratio = 0.2
LR = 0.05
MAX_LENGTH = 20
epoch_num = 100
is_train = False


################################
#Example inputs of compute_bleu
################################
#compute BLEU-4 score
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)


#Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden, c_state):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded

        output, (hidden, c_state) = self.lstm(output, (hidden, c_state))
        return output, hidden, c_state

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

#Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, c_state):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        
        output, (hidden, c_state) = self.lstm(output, (hidden, c_state))
        output = self.out(output[0])
        return output, hidden, c_state

        
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_c_state = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    #----------sequence to sequence part for encoder----------#
    for ei in range(input_length):
        encoder_output, encoder_hidden, encoder_c_state = encoder(input_tensor[ei].cuda(GPUID[0]), encoder_hidden.cuda(GPUID[0]), encoder_c_state.cuda(GPUID[0]))
    
    decoder_input = torch.tensor([[SOS_token]])

    decoder_hidden = encoder_output
    decoder_c_state = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
	

    #----------sequence to sequence part for decoder----------#
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            use_dropout = True if random.random() < dropout_ratio else False
            decoder_output, decoder_hidden, decoder_c_state = decoder(
                decoder_input.cuda(GPUID[0]), decoder_hidden.cuda(GPUID[0]), decoder_c_state.cuda(GPUID[0]))
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
            # print(decoder_input)
            if use_dropout:
                decoder_input = torch.tensor(28).cuda(GPUID[0])

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            use_dropout = True if random.random() < dropout_ratio else False
            decoder_output, decoder_hidden, decoder_c_state = decoder(
                decoder_input.cuda(GPUID[0]), decoder_hidden.cuda(GPUID[0]), decoder_c_state.cuda(GPUID[0]))
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            
            if decoder_input.item() == EOS_token:
                break
            if use_dropout:
                decoder_input = torch.tensor(28).cuda(GPUID[0])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def str_to_int(x_str):
    return np.array([ord(x)-ord('a')+2 for x in x_str])

def int_to_str(x_int):
    return "".join(np.array([chr(x + ord('a')-2) for x in x_int]))


def dataloader(mode):
    if mode == 'train':
        with open('./train.json') as f:
            data_input = json.load(f)
        data = []
        for i in data_input:
            word_tar = torch.from_numpy(np.append(str_to_int(i['target']), EOS_token)).view(-1, 1)
            for j in i['input']:
                word_input = torch.from_numpy(str_to_int(j)).view(-1, 1)
                data.append((word_input, word_tar))

        return data
    else:
        with open('./test.json') as f:
            data_input = json.load(f)
        data = []
        origin = []
        for i in data_input:
            word_tar = torch.from_numpy(str_to_int(i['target'])).view(-1, 1)
            word_input = torch.from_numpy(str_to_int(i['input'][0])).view(-1, 1)
            origin.append((i['input'][0], i['target']))
            data.append((word_input, word_tar))

        return data, origin


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    encoder.train()
    decoder.train()

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate,  weight_decay=1e-4)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate,  weight_decay=1e-4)
    # your own dataloader
    training_pairs = dataloader('train')

    criterion = nn.CrossEntropyLoss()

    for iter in range(1, n_iters + 1):
        # print(iter)
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor.cuda(GPUID[0]), target_tensor.cuda(GPUID[0]), encoder.cuda(GPUID[0]),
                     decoder.cuda(GPUID[0]), encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
    plot_loss_total /= n_iters
    writer.add_scalar('/loss/loss_avg', plot_loss_total, epoch)
    torch.save(encoder.state_dict(), '%sencoder_%d.pth'%(params_save_dir, epoch))
    torch.save(decoder.state_dict(), '%sdecoder_%d.pth'%(params_save_dir, epoch))

def test(input_tensor, encoder, decoder, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_c_state = encoder.initHidden()

    input_length = input_tensor.size(0)

    loss = 0

    #----------sequence to sequence part for encoder----------#
    for ei in range(input_length):
        encoder_output, encoder_hidden, encoder_c_state = encoder(input_tensor[ei].cuda(GPUID[0]), encoder_hidden.cuda(GPUID[0]),\
                                                                     encoder_c_state.cuda(GPUID[0]))

    decoder_input = torch.tensor([[SOS_token]])

    decoder_hidden = encoder_output
    decoder_c_state = encoder_hidden
	

    #----------sequence to sequence part for decoder----------#

    # Without teacher forcing: use its own predictions as the next input
    pred = []
    for di in range(max_length):
        
        decoder_output, decoder_hidden, decoder_c_state = decoder(
            decoder_input.cuda(GPUID[0]), decoder_hidden.cuda(GPUID[0]), decoder_c_state.cuda(GPUID[0]))
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        

        if decoder_input.item() == EOS_token:
            break
            
        pred.append(decoder_input.cpu().numpy())
    
    return int_to_str(pred)

def testIters(encoder, decoder, n_iters, testing_pairs, origin_txt):
    start = time.time()
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder.eval()
    decoder.eval()

    bleu = 0

    with torch.no_grad():
        for iter in range(1, n_iters + 1):
            # print(iter)
            testing_pair = testing_pairs[iter - 1]
            input_tensor = testing_pair[0]
            target_tensor = testing_pair[1]

            predict = test(input_tensor.cuda(GPUID[0]), encoder.cuda(GPUID[0]),
                        decoder.cuda(GPUID[0]))
            # print("==========================================")
            # print("input:%s" % (origin_txt[iter - 1][0]))
            # print("target:%s" % (origin_txt[iter - 1][1]))
            # print("pred:%s" % (predict))
            bleu += compute_bleu(origin_txt[iter - 1][1], predict)
        bleu /= n_iters
        print("BLEU-4 score:%f" % (bleu))
        writer.add_scalar('/score/BLEU-4', bleu, epoch)


def testFinal(encoder, decoder, load_model, n_iters, testing_pairs, origin_txt):
    start = time.time()
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder.load_state_dict(torch.load('%sencoder_%d.pth'%(params_save_dir, load_model)))
    decoder.load_state_dict(torch.load('%sdecoder_%d.pth'%(params_save_dir, load_model)))
    encoder.eval()
    decoder.eval()
    
    bleu = 0

    with torch.no_grad():
        for iter in range(1, n_iters + 1):
            # print(iter)
            testing_pair = testing_pairs[iter - 1]
            input_tensor = testing_pair[0]
            target_tensor = testing_pair[1]

            predict = test(input_tensor.cuda(GPUID[0]), encoder.cuda(GPUID[0]),
                        decoder.cuda(GPUID[0]))
            
            print("==========================================")

            print("input:%s" % (origin_txt[iter - 1][0]))
            print("target:%s" % (origin_txt[iter - 1][1]))
            print("pred:%s" % (predict))
            bleu += compute_bleu(origin_txt[iter - 1][1], predict)
        bleu /= n_iters
        print("BLEU-4 score:%f" % (bleu))
        # writer.add_scalar('/test_score/BLEU-4', bleu, epoch)

if is_train:
    writer = SummaryWriter(log_dir)
    encoder1 = EncoderRNN(vocab_size, hidden_size)
    decoder1 = DecoderRNN(hidden_size, vocab_size)
    encoder1.cuda(GPUID[0])

    testing_pairs, origin_txt = dataloader('test')
    for epoch in range(epoch_num):
        trainIters(encoder1.cuda(GPUID[0]), decoder1.cuda(GPUID[0]), 12925, print_every=6000)
        testIters(encoder1.cuda(GPUID[0]), decoder1.cuda(GPUID[0]), 50, testing_pairs, origin_txt)
else:
    encoder1 = EncoderRNN(vocab_size, hidden_size)
    decoder1 = DecoderRNN(hidden_size, vocab_size)
    encoder1.cuda(GPUID[0])
    decoder1.cuda(GPUID[0])
    testing_pairs, origin_txt = dataloader('test')
    testFinal(encoder1.cuda(GPUID[0]), decoder1.cuda(GPUID[0]), 99, 50, testing_pairs, origin_txt)