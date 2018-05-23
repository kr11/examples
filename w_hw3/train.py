# coding: utf-8
import math
import os
import sys
import torch
import torch.nn as nn

sys.path.append('../')
sys.path.append('../w_hw3')
import w_hw3.model as model
import w_hw3.utils as utils
import data

import time

start = time.time()
print("start!====")

# prepare
eval_batch_size = 10
args = utils.get_args_parser()
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
device = utils.check_device(args)

###############################################################################
# Load data
###############################################################################
if not os.path.exists(args.save):
    os.makedirs(args.save)

corpus = data.Corpus()
corpus.set_train(os.path.join(args.data, 'train.txt'), args.train_size)
corpus.set_valid(os.path.join(args.data, 'valid.txt'), args.valid_size)
corpus.save_dictionary(os.path.join(args.save, 'rnn_model_dict'))

train_data = utils.batchify(corpus.train, args.batch_size, device)
val_data = utils.batchify(corpus.valid, eval_batch_size, device)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(
    device)
criterion = nn.CrossEntropyLoss()


###############################################################################
# Training code
###############################################################################

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = utils.get_batch(args, train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = utils.repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                              elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = utils.evaluate(val_data, model, corpus, criterion, args, eval_batch_size)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(os.path.join(args.save, 'model.pt'), 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
