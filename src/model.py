import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from data_loader import load_data
import numpy as np


USE_CUDA = torch.cuda.is_available()


def to_gpu(var):
    if USE_CUDA:
        return var.cuda()
    return var


class KTRN(nn.Module):
    def __init__(self, args, n_user, n_entity, n_relation, link_path):
        super(KTRN, self).__init__()
        self._parse_args(args, n_user, n_entity, n_relation, link_path)

        self.user_embeddings = nn.Embedding(self.n_user, self.embedding_size)
        self.entity_embeddings = nn.Embedding(self.n_entity, self.embedding_size)
        self.relation_embeddings = nn.Embedding(self.n_relation, self.embedding_size)
        self.relationHyper = nn.Embedding(self.n_relation, self.embedding_size)
        self.distfn = nn.PairwiseDistance(2)
        # self.rnn = nn.RNN()
        self.rnns = []
        for _ in range(self.n_iter):
            self.rnns.append(nn.LSTM(input_size=2*self.embedding_size, batch_first=True, hidden_size=self.embedding_size, num_layers=1))
        # self.out = nn.Linear(32, 1)
        self.out = nn.Linear(self.embedding_size, 1)
        self.agg_sum = nn.Linear(self.embedding_size, self.embedding_size)
        self.agg_con = nn.Linear(self.embedding_size*2, self.embedding_size)

        for i in range(len(self.rnns)):
            self.rnns[i] = to_gpu(self.rnns[i])
        self.user_embeddings = to_gpu(self.user_embeddings)
        self.entity_embeddings = to_gpu(self.entity_embeddings)
        self.relation_embeddings = to_gpu(self.relation_embeddings)
        self.relationHyper = to_gpu(self.relationHyper)
        self.out = to_gpu(self.out)
        self.agg_sum = to_gpu(self.agg_sum)
        self.agg_con = to_gpu(self.agg_con)

    def _parse_args(self, args, n_user, n_entity, n_relation, link_path):
        self.n_user = n_user
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.link_path = link_path

        self.n_iter = args.n_iter
        self.aggregator = args.aggregator
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.embedding_size = args.embedding_size
        self.l2_weight = args.l2_weight
        # self.kge_weight = args.kge_weight
        self.lr = args.lr
        self.update_type = args.update_type
        self.criterion = nn.BCELoss()

    def forward(self, train_data):
        users = torch.LongTensor(train_data[:, 0]).cuda()
        items = torch.LongTensor(train_data[:, 1]).cuda()
        ratings = torch.LongTensor(train_data[:, 2]).cuda()

        users_emb = self.user_embeddings(users)
        # items_emb_update, kge_loss = self.update_item(users_emb, items, self.update_type)
        items_emb_update = self.update_item(users_emb, items, self.update_type)
        users_emb_expand = torch.unsqueeze(users_emb, dim=1)
        items_emb_update_ex = torch.unsqueeze(items_emb_update, dim=2)
        scores = torch.squeeze(torch.matmul(users_emb_expand, items_emb_update_ex))
        scores_normalized = torch.sigmoid(scores)
        rs_loss = self.criterion(scores_normalized, ratings.float())
        # loss = rs_loss + self.kge_weight * kge_loss
        loss = rs_loss
        
        return loss, scores_normalized, items

    def update_item(self, users_emb, items, update_type):

        items_emb = self.entity_embeddings(items)

        for hop in range(self.n_iter):
            hop_link = torch.LongTensor(self.link_path[hop]).cuda()
            item_links = hop_link[items]
            users_emb_hop = users_emb.repeat(1, hop_link.shape[1])

            users_emb_hop = users_emb_hop.view(-1, self.embedding_size)
            seed = item_links[:, :, 0]
            seed_emb = self.entity_embeddings(seed).view(-1, self.embedding_size)

            link_user = [torch.cat([users_emb_hop, seed_emb], 1)]
            link_index = [seed]
            link = [users_emb_hop, seed_emb]
            for j in range(1, 2 * (hop + 1) + 1):
                if not j % 2 == 0:
                    relation = item_links[:, :, j]
                    link_index.append(relation)
                    relation_emb = self.relation_embeddings(relation).view(-1, self.embedding_size)
                    link.append(relation_emb)
                    tail = item_links[:, :, j+1]
                    link_index.append(tail)
                    tail_emb = self.entity_embeddings(tail).view(-1, self.embedding_size)
                    link.append(tail_emb)
                    link_user.append(torch.cat([relation_emb, tail_emb], 1))
            user_add_link = torch.stack(link_user, 1)

            # r_out, (h_n, h_c) = self.rnns[hop](user_link, None)
            r_out, (h_n, h_c) = self.rnns[hop](user_add_link, None)

            if update_type == 'hidden_out':
                final_out = r_out[:, -1, :].view(-1, hop_link.shape[1], self.embedding_size).sum(dim=1)

            elif update_type == 'probs':
                r_out_last = r_out[:, -1, :].view(-1, hop_link.shape[1], self.embedding_size)
                probs = torch.sigmoid(self.out(r_out_last))
                tail = link[-1].reshape(probs.shape[0], hop_link.shape[1], self.embedding_size)
                final_out = torch.squeeze(torch.mul(probs, tail)).sum(dim=1)

            if self.aggregator == 'sum':
                items_emb = self.agg_sum(items_emb + final_out)
            elif self.aggregator == 'con':
                concat = torch.cat((items_emb, final_out), 1) 
                items_emb = self.agg_con(concat)
            elif self.aggregator == 'no':
                items_emb = items_emb + final_out
            elif self.aggregator == 'bi':
                concat = torch.cat((items_emb, final_out), 1)
                items_emb = self.agg_con(concat) + self.agg_sum(items_emb + final_out)
        
        # kge_loss = self.compute_kge_loss(link_index[0].view(-1), link_index[1].view(-1), link_index[2].view(-1)) 
        # return items_emb, kge_loss
        return items_emb

    def compute_kge_loss(self, head, relation, tail):

        head = torch.squeeze(self.entity_embeddings(head), dim=1)
        relHyper = torch.squeeze(self.relationHyper(relation), dim=1)
        relation = torch.squeeze(self.relation_embeddings(relation), dim=1)
        tail = torch.squeeze(self.entity_embeddings(tail), dim=1)

        head = head - relHyper * torch.sum(head * relHyper, dim=1, keepdim=True)
        tail = tail - relHyper * torch.sum(tail * relHyper, dim=1, keepdim=True)
        
        loss = self.distfn(head+relation, tail).mean()

        return loss

    def evaluate(self, scores, ratings):
        scores = scores.detach().cpu().numpy()
        ratings = ratings.cpu().numpy()
        auc = roc_auc_score(y_true=ratings, y_score=scores)
        f1 = f1_score(y_true=ratings, y_pred=scores.round())
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, ratings))
        return auc, acc, f1


if __name__ == '__main__':
    import argparse
    import numpy as np

    np.random.seed(555)

    parser = argparse.ArgumentParser()

    # movie
    parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
    parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use:sum/con/no')
    parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
    parser.add_argument('--neighbor_sample_size', type=int, default=4, help='the number of neighbors to be sampled')
    parser.add_argument('--embedding_size', type=int, default=32, help='dimension of user and entity embeddings')
    parser.add_argument('--n_iter', type=int, default=2,
                        help='number of iterations when computing entity representation')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of l2 regularization')
    parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
    parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')
    parser.add_argument('--update_type', type=str, default='hidden_out', help='size of training dataset')
    args = parser.parse_args()

    # n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, adj_entity, adj_relation, hop_link
    data = load_data(args)
    adj_entity, adj_relation, hop_link = data[-3], data[-2], data[-1]
    n_user = data[0]
    n_entity = data[2]
    n_relation = data[3]
    train_data = data[4][:args.batch_size]
    np.random.shuffle(train_data)
    model = PathNet(args, n_user, n_entity, n_relation, hop_link)
    model(train_data)