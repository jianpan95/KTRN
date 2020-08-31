import numpy as np
import os


def load_data(args):
    n_user, n_item, train_data, eval_data, test_data = load_rating(args)
    n_entity, n_relation, adj_entity, adj_relation = load_kg(args)
    hop_link = construct_link_path(args, adj_entity, adj_relation)
    print('data loaded.')

    return n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, adj_entity, adj_relation, hop_link


def load_rating(args):
    dataset_file = 'reading ' + args.dataset + ' rating file ...'
    print(dataset_file)

    # reading rating file
    rating_file = '../data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)

    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    train_data, eval_data, test_data = dataset_split(rating_np, args)

    return n_user, n_item, train_data, eval_data, test_data


def dataset_split(rating_np, args):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    if args.ratio < 1:
        train_indices = np.random.choice(list(train_indices), size=int(len(train_indices) * args.ratio), replace=False)

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data


def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    kg = construct_kg(kg_np)
    adj_entity, adj_relation = construct_adj(args, kg, n_entity)

    return n_entity, n_relation, adj_entity, adj_relation


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = dict()
    for triple in kg_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
    return kg


def construct_adj(args, kg, entity_num):
    print('constructing adjacency matrix ...')
    # each line of adj_entity stores the sampled neighbor entities for a given entity
    # each line of adj_relation stores the corresponding sampled neighbor relations
    adj_entity = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    for entity in range(entity_num):
        neighbors = kg[entity]
        n_neighbors = len(neighbors)
        if n_neighbors >= args.neighbor_sample_size:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=True)
        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

    return adj_entity, adj_relation


def construct_link_path(args, adj_entity, adj_relation):
    print('constructing link path ...')
    # args.neighbor_sample_size
    # arg.n_iter
    hop_links = list()
    all_link_path = list()
    for hop in range(args.n_iter):
        if hop == 0:
            for i in range(adj_entity.shape[0]):
                entity_link = []
                for j in range(args.neighbor_sample_size):
                    one_link = [i, adj_relation[i][j], adj_entity[i][j]]
                    entity_link.append([adj_relation[i][j], adj_entity[i][j]])
                    all_link_path.append(one_link)
            hop_links.append(np.array(all_link_path))
            # all_link_path = np.array(all_link_path)
        else:
            temp_link_path = list()
            for path_index in range(len(all_link_path)):
                path = all_link_path[path_index]
                for i in range(args.neighbor_sample_size):
                    temp = path[:]
                    tail = temp[-1]
                    temp.append(adj_relation[tail][i])
                    temp.append(adj_entity[tail][i])
                    temp_link_path.append(temp)
            all_link_path = temp_link_path
            hop_links.append(np.array(all_link_path))

    users_link = []
    for hop in range(args.n_iter):
        temp = []
        hop_link = hop_links[hop]
        numbers = args.neighbor_sample_size**(hop+1)
        start = 0
        while start+numbers < hop_link.shape[0]:
            user_link = hop_link[start:start+numbers]
            temp.append(user_link)
            start += numbers
        users_link.append(np.array(temp))

    return users_link


if __name__ == '__main__':
    import argparse
    import numpy as np

    np.random.seed(555)

    parser = argparse.ArgumentParser()

    # movie
    parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
    parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
    parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
    parser.add_argument('--neighbor_sample_size', type=int, default=4, help='the number of neighbors to be sampled')
    parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
    parser.add_argument('--n_iter', type=int, default=2,
                        help='number of iterations when computing entity representation')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of l2 regularization')
    parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
    parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')
    args = parser.parse_args()
    data = load_data(args)
    adj_entity, adj_relation, hop_link = data[-3], data[-2], data[-1]
    print(len(hop_link))
    print(hop_link[0])
    print(hop_link[0].shape)
    print(hop_link[1])
    print(hop_link[1].shape)


