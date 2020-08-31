import argparse
import numpy as np
from data_loader import load_data
from train import train

np.random.seed(555)


parser = argparse.ArgumentParser()

# movie
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='con', help='which aggregator to use: no/sum/con/bi')
parser.add_argument('--neighbor_sample_size', type=int, default=32, help='the number of neighbors to be sampled')
parser.add_argument('--embedding_size', type=int, default=8, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1,help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--l2_weight', type=float, default=0.001, help='weight of l2 regularization')
parser.add_argument('--kge_weight', type=float, default=0.005, help='weight of the KGE term')
parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
parser.add_argument('--update_type', type=str, default='hidden_out', help='the output of LSTM: probs or hidden_out')
parser.add_argument('--use_cuda', type=bool, default=True, help='to use CUDA or not')
parser.add_argument('--n_epoch', type=int, default=50, help='the number of epochs')

'''
# book
parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='con', help='which aggregator to use: no/sum/con/bi')
parser.add_argument('--embedding_size', type=int, default=6, help='dimension of user and entity embeddings')
parser.add_argument('--neighbor_sample_size', type=int, default=16, help='the number of neighbors to be sampled')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--kge_weight', type=float, default=0.005, help='weight of the KGE term')
parser.add_argument('--l2_weight', type=float, default=2e-3, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
parser.add_argument('--update_type', type=str, default='hidden_out', help='the output of LSTM: probs or hidden_out')
parser.add_argument('--use_cuda', type=bool, default=True, help='to use CUDA or not')
parser.add_argument('--n_epoch', type=int, default=50, help='the number of epochs')
'''

'''
# music
parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='con', help='which aggregator to use: no/sum/con/bi')
parser.add_argument('--embedding_size', type=int, default=8, help='dimension of user and entity embeddings')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--n_iter', type=int, default=1,help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--l2_weight', type=float, default=0.0045, help='weight of l2 regularization')
parser.add_argument('--kge_weight', type=float, default=0.005, help='weight of the KGE term')
parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')
parser.add_argument('--update_type', type=str, default='hidden_out', help='the output of LSTM: probs or hidden_out')
parser.add_argument('--use_cuda', type=bool, default=True, help='to use CUDA or not')
parser.add_argument('--n_epoch', type=int, default=50, help='the number of epochs')
'''


args = parser.parse_args()
data = load_data(args)
train(args, data)
