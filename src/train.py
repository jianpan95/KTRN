import numpy as np
import torch
from model import KTRN
from data_loader import load_data
import datetime


def train(args, data):
    adj_entity, adj_relation, hop_link = data[-3], data[-2], data[-1]
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, test_data, eval_data = data[4], data[5], data[6]

    model = KTRN(args, n_user, n_entity, n_relation, hop_link)
    user_list, train_record, test_record, item_set, k_list = topk_settings(True, train_data, test_data, n_item)
    # 用不用CUDA
    if args.use_cuda:
        model.cuda()

    # filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表
    # Adam第一个参数是待优化参数的iterable或者是定义了参数组的dict，第二个参数是lr (float, 可选) – 学习率（默认：1e-3）
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), args.lr)

    for step in range(args.n_epoch):
        # training
        np.random.shuffle(train_data)
        start = 0
        while start < train_data.shape[0]:
            # forward(self, train_data, ripple_set,)
            loss, _, _ = model(train_data[start:start+args.batch_size])

            # L2 normalization
            l2_regularization = torch.tensor([0],dtype =torch.float32).cuda()
            for param in model.parameters():
                l2_regularization += torch.norm(param, 2)
            
            loss = loss + args.l2_weight*l2_regularization

            # 置零，如果不置零,Variable 的梯度在每次 backward 的时候都会累加
            optimizer.zero_grad()
            loss.backward()
            # 用了optimizer.step()，模型才会更新
            optimizer.step()

            start += args.batch_size
            # print(start, 'loss:', float(loss))

        # evaluation
        train_auc, train_acc = evaluation(args, model, train_data, args.batch_size)
        eval_auc, eval_acc = evaluation(args, model, eval_data, args.batch_size)
        test_auc, test_acc = evaluation(args, model, test_data, args.batch_size)

        print('epoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'
              % (step, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))

        if False:
                precision, recall = topk_eval(model, user_list, train_record, test_record, item_set, k_list, args.batch_size)
                print('precision: ', end='')
                for i in precision:
                    print('%.4f\t' % i, end='')
                print()
                print('  recall : ', end='')
                for i in recall:
                    print('%.4f\t' % i, end='')
                print('\n')


def evaluation(args, model, data, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    # model.eval() ：让model变成测试模式，不启用 BatchNormalization 和 Dropout
    model.eval()
    while start < data.shape[0]:
        data_batch = data[start:start+args.batch_size]
        _, scores, _ = model(data_batch)
        ratings = torch.LongTensor(data_batch[:, 2])
        auc, acc, _ = model.evaluate(scores, ratings)
        auc_list.append(auc)
        acc_list.append(acc)
        start += batch_size
    # 启用 BatchNormalization 和 Dropout
    model.train()
    return float(np.mean(auc_list)), float(np.mean(acc_list))

def topk_eval(model, user_list, train_record, test_record, item_set, k_list, batch_size):
    # 记录各个topk的值的字典
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    for user in user_list:
        # 去掉训练过的item
        test_item_list = list(item_set - train_record[user])
        # 每个item对应的score
        item_score_map = dict()
        start = 0
        model.eval()
        user_indices = np.array([user] * batch_size)
        rating_indices = np.random.randint(0, 2, batch_size)
        while start + batch_size <= len(test_item_list):
            item_indices = np.array(test_item_list[start:start + batch_size])
            data_indice = np.array([user_indices, item_indices, rating_indices]).T
            _, scores, items = model(data_indice)
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            item_indices = np.array(test_item_list[start:] + [test_item_list[-1]] * (batch_size - len(test_item_list) + start))
            data_indice = np.array([user_indices, item_indices, rating_indices]).T
            _, scores, items = model(data_indice)
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        model.train()
        for k in k_list:
            item_sorted_to_int = [ int(i.detach().cpu().numpy()) for i in item_sorted[:k]]
            hit_num = len(set(item_sorted_to_int) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]

    return precision, recall

def topk_settings(show_topk, train_data, test_data, n_item):
    if show_topk:
        user_num = 100
        k_list = [2, 5, 10, 20, 50]
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(test_data, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        if len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)
        item_set = set(list(range(n_item)))
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5

def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict

