from __future__ import print_function
from os.path import join
import numpy as np
import networkx as nx
import argparse
import torch
import sklearn
from tqdm import tqdm
from sklearn import preprocessing
from utils import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


cmd_opt = argparse.ArgumentParser(description='Argparser for graph_classification')
cmd_opt.add_argument('-mode', default='gpu', help='cpu/gpu')
cmd_opt.add_argument('-gm', default='mean_field', help='mean_field/loopy_bp')
cmd_opt.add_argument('-data', default=None, help='data folder name')
cmd_opt.add_argument('-batch_size', type=int, default=64, help='minibatch size')
cmd_opt.add_argument('-seed', type=int, default=1, help='seed')
cmd_opt.add_argument('-feat_dim', type=int, default=0, help='dimension of discrete node feature (maximum node tag)')
cmd_opt.add_argument('-num_class', type=int, default=2, help='#classes')
cmd_opt.add_argument('-fold', type=int, default=1, help='fold (1..10)')
cmd_opt.add_argument('-test_number', type=int, default=0,
                     help='if specified, will overwrite -fold and use the last -test_number graphs as testing data')
cmd_opt.add_argument('-num_epochs', type=int, default=1000, help='number of epochs')
cmd_opt.add_argument('-latent_dim', type=str, default='64', help='dimension(s) of latent layers')
cmd_opt.add_argument('-sortpooling_k', type=float, default=30, help='number of nodes kept after SortPooling')
cmd_opt.add_argument('-out_dim', type=int, default=1024, help='s2v output size')
cmd_opt.add_argument('-hidden', type=int, default=100, help='dimension of regression')
cmd_opt.add_argument('-max_lv', type=int, default=4, help='max rounds of message passing')
cmd_opt.add_argument('-learning_rate', type=float, default=0.0001, help='init learning_rate')
cmd_opt.add_argument('-dropout', type=bool, default=False, help='whether add dropout after dense layer')
cmd_opt.add_argument('-printAUC', type=bool, default=False,
                     help='whether to print AUC (for binary classification only)')
cmd_opt.add_argument('-extract_features', type=bool, default=False, help='whether to extract final graph features')

cmd_args, _ = cmd_opt.parse_known_args()

cmd_args.latent_dim = [int(x) for x in cmd_args.latent_dim.split('-')]
if len(cmd_args.latent_dim) == 1:
    cmd_args.latent_dim = cmd_args.latent_dim[0]


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        self.num_nodes = len(g)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.degs = list(dict(g.degree).values())

        if len(g.edges()) != 0:
            x, y = zip(*g.edges())
            self.num_edges = len(x)
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])


def load_data():
    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open(
            '/Users/zfj/workspace/code/network-embedding/StructPool-mine/pytorch_structure2vec-master/graph_classification/data/%s/%s.txt' % (
            cmd_args.data, cmd_args.data), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            # assert len(g.edges()) * 2 == n_edges  (some graphs in COLLAB have self-loops, ignored here)
            assert len(g) == n
            g_list.append(S2VGraph(g, l, node_tags, node_features))
    for g in g_list:
        g.label = label_dict[g.label]
    cmd_args.num_class = len(label_dict)
    cmd_args.feat_dim = len(feat_dict)  # maximum node label (tag)
    if node_feature_flag == True:
        cmd_args.attr_dim = node_features.shape[1]  # dim of node features (attributes)
    else:
        cmd_args.attr_dim = 0

    print('# classes: %d' % cmd_args.num_class)
    print('# maximum node tag: %d' % cmd_args.feat_dim)

    if cmd_args.test_number == 0:
        train_idxes = np.loadtxt(
            '/Users/zfj/workspace/code/network-embedding/StructPool-mine/pytorch_structure2vec-master/graph_classification/data/%s/10fold_idx/train_idx-%d.txt' % (
            cmd_args.data, cmd_args.fold), dtype=np.int32).tolist()
        test_idxes = np.loadtxt(
            '/Users/zfj/workspace/code/network-embedding/StructPool-mine/pytorch_structure2vec-master/graph_classification/data/%s/10fold_idx/test_idx-%d.txt' % (
            cmd_args.data, cmd_args.fold), dtype=np.int32).tolist()
        return [g_list[i] for i in train_idxes], [g_list[i] for i in test_idxes]
    else:
        return g_list[: n_g - cmd_args.test_number], g_list[n_g - cmd_args.test_number:]


def gen_graph(adj, inf_features, label, cur_node_features):
    # g = nx.Graph()
    # g.add_nodes_from(list(range(len(cur_vids))))
    g = nx.from_numpy_array(adj)
    node_features = np.concatenate((cur_node_features, inf_features), axis=1)  #todo
    g.label = label
    g.remove_nodes_from(list(nx.isolates(g)))
    g.node_features = node_features
    return g


def process_g(self, g):
    g.feas = torch.FloatTensor(g.node_tags)
    A = torch.FloatTensor(nx.to_numpy_matrix(g))
    g.A = A + torch.eye(g.number_of_nodes())
    return g


def load_self_data(args):
    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    file_dir = join(settings.DATA_DIR, args.data)
    print('loading data ...')

    graphs = np.load(join(file_dir, "adjacency_matrix.npy")).astype(np.float32)

    # wheather a user has been influenced
    # wheather he/she is the ego user
    influence_features = np.load(
        join(file_dir, "influence_feature.npy")).astype(np.float32)
    logger.info("influence features loaded!")

    labels = np.load(join(file_dir, "label.npy"))
    logger.info("labels loaded!")

    vertices = np.load(join(file_dir, "vertex_id.npy"))
    logger.info("vertex ids loaded!")

    vertex_features = np.load(join(file_dir, "vertex_feature.npy"))
    vertex_features = preprocessing.scale(vertex_features)
    # vertex_features = torch.FloatTensor(vertex_features)
    logger.info("global vertex features loaded!")

    n_g = len(graphs)

    for i in tqdm(range(n_g), desc="Create graph", unit='graphs'):
        cur_vids = vertices[i]
        cur_node_features = vertex_features[cur_vids]
        g = gen_graph(graphs[i], influence_features[i], labels[i], cur_node_features)
        node_tags = list(range(len(influence_features[0])))
        g_list.append(S2VGraph(g, g.label, node_tags, g.node_features))

        if i > settings.TEST_SIZE:
            break

    # new_g_list = []
    # for g in tqdm(g_list, desc="Process graph", unit='graphs'):
    #     new_g_list.append(process_g(g))

    n_g = len(g_list)
    cmd_args.feat_dim = len(influence_features[0])

    cmd_args.attr_dim = g_list[0].node_features.shape[1]  # dim of node features (attributes)
    print("attr dim", cmd_args.attr_dim)

    print('# classes: %d' % cmd_args.num_class)
    print('# maximum node tag: %d' % cmd_args.feat_dim)


    g_list = sklearn.utils.shuffle(g_list, random_state=args.seed)

    train_ratio = 0.5
    valid_ratio = 0.25
    n_train = int(n_g * train_ratio)
    n_valid = int((train_ratio + valid_ratio) * n_g) - n_train
    train_gs = [g_list[i] for i in range(0, n_train)]
    valid_gs = [g_list[i] for i in range(n_train, n_train + n_valid)]
    test_gs = [g_list[i] for i in range(n_train + n_valid, n_g)]

    return train_gs, valid_gs, test_gs
