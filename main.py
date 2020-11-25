import sys
import os
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import math
from DGCNN_embedding import DGCNN
from mlp_dropout import MLPClassifier
from sklearn import metrics
from embedding import EmbedMeanField, EmbedLoopyBP
from util import cmd_args, load_data, load_self_data

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp


sys.path.append(
    '%s/pytorch_structure2vec-master/s2v_lib' % os.path.dirname(
        os.path.realpath(__file__)))


class Classifier(nn.Module):
    def __init__(self, cmd_args):
        super(Classifier, self).__init__()
        if cmd_args.gm == 'mean_field':
            model = EmbedMeanField
        elif cmd_args.gm == 'loopy_bp':
            model = EmbedLoopyBP
        elif cmd_args.gm == 'DGCNN':
            model = DGCNN
        else:
            print('unknown gm %s' % cmd_args.gm)
            sys.exit()

        print("attr dim ########", cmd_args.attr_dim)
        if cmd_args.gm == 'DGCNN':
            self.s2v = model(
                latent_dim=cmd_args.latent_dim,
                output_dim=cmd_args.out_dim,
                num_node_feats=cmd_args.feat_dim+cmd_args.attr_dim,
                num_edge_feats=0,
                k=cmd_args.sortpooling_k)
        else:
            self.s2v = model(
                latent_dim=cmd_args.latent_dim,
                output_dim=cmd_args.out_dim,
                num_node_feats=cmd_args.feat_dim+cmd_args.attr_dim,
                num_edge_feats=0,
                max_lv=cmd_args.max_lv)
        out_dim = cmd_args.out_dim
        if out_dim == 0:
            if cmd_args.gm == 'DGCNN':
                out_dim = self.s2v.dense_dim
            else:
                out_dim = cmd_args.latent_dim
        self.mlp = MLPClassifier(
            input_size=out_dim, hidden_size=cmd_args.hidden,
            num_class=cmd_args.num_class, with_dropout=cmd_args.dropout)

    def PrepareFeatureLabel(self, batch_graph):
        labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0
        # print("len batch graph", len(batch_graph))

        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            concat_tag = []
        else:
            node_tag_flag = False

        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            concat_feat = []
        else:
            node_feat_flag = False

        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            if node_tag_flag:
                # print("cur batch node tags", batch_graph[i].node_tags)
                concat_tag += batch_graph[i].node_tags
            if node_feat_flag:
                tmp = torch.from_numpy(
                    batch_graph[i].node_features).type('torch.FloatTensor')
                concat_feat.append(tmp)

        if node_tag_flag:
            concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
            node_tag = torch.zeros(n_nodes, cmd_args.feat_dim)
            # print("node tag", node_tag.shape)
            # print("concat tag", concat_tag.shape)
            node_tag.scatter_(1, concat_tag, 1)

        if node_feat_flag:
            node_feat = torch.cat(concat_feat, 0)

        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags (node labels)
            # with continuous node features
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag is False and node_tag_flag:
            node_feat = node_tag
        elif node_feat_flag and node_tag_flag is False:
            pass
        else:
            node_feat = torch.ones(n_nodes, 1)

        if cmd_args.mode == 'gpu':
            node_feat = node_feat.cuda()
            labels = labels.cuda()

        return node_feat, labels

    def forward(self, batch_graph):
        node_feat, labels = self.PrepareFeatureLabel(batch_graph)
        embed = self.s2v(batch_graph, node_feat, None)

        return self.mlp(embed, labels)

    def output_features(self, batch_graph):
        node_feat, labels = self.PrepareFeatureLabel(batch_graph)
        embed = self.s2v(batch_graph, node_feat, None)
        return embed, labels


def loop_dataset(g_list, classifier, sample_idxes, optimizer=None,
                 bsize=cmd_args.batch_size):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize # noqa
    pbar = tqdm(range(total_iters), unit='batch')
    all_targets = []
    all_scores = []

    n_samples = 0
    # print("bsize", bsize)
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize: (pos + 1) * bsize]

        batch_graph = [g_list[idx] for idx in selected_idx]
        targets = [g_list[idx].label for idx in selected_idx]
        all_targets += targets
        logits, loss, acc = classifier(batch_graph)
        all_scores.append(logits[:, 1].detach())  # for binary classification

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().numpy()
        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc))

        total_loss.append(np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores).cpu().numpy()

    # np.savetxt('test_scores.txt', all_scores)  # output test predictions

    all_targets = np.array(all_targets)
    fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    avg_loss = np.concatenate((avg_loss, [auc]))

    return avg_loss


def evaluate(g_list, classifier, sample_idxes, bsize=cmd_args.batch_size, thr=None, return_best_thr=False):
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize # noqa
    pbar = tqdm(range(total_iters), unit='batch')

    total = 0
    y_true, y_pred, y_score = [], [], []
    losses = []

    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize: (pos + 1) * bsize]

        batch_graph = [g_list[idx] for idx in selected_idx]
        targets = [g_list[idx].label for idx in selected_idx]
        # all_targets += targets
        out, loss, acc = classifier(batch_graph)
        # all_scores.append(logits[:, 1].detach())  # for binary classification

        loss = loss.data.cpu().numpy()
        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc))

        # total_loss.append(np.array([loss, acc]) * len(selected_idx))
        losses.append(loss)

        y_true += targets
        y_pred += out.max(1)[1].data.tolist()
        y_score += out[:, 1].data.tolist()

        total += len(selected_idx)

    if thr is not None:
        logger.info("using threshold %.4f", thr)
        y_score = np.array(y_score)
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > thr] = 1

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    logger.info("loss: %.4f AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f",
                sum(losses) / total, auc, prec, rec, f1)
    loss_ret = sum(losses) / total

    if return_best_thr:
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]
        logger.info("best threshold=%4f, f1=%.4f", best_thr, np.max(f1s))
        return loss_ret, [prec, rec, f1, auc], best_thr
    else:
        return loss_ret, [prec, rec, f1, auc], None


if __name__ == '__main__':
    print(cmd_args)
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    cmd_args.data = "twitter"

    # train_graphs, test_graphs = load_data()
    train_graphs, valid_graphs, test_graphs = load_self_data(cmd_args)
    print("attr dim", cmd_args.attr_dim)
    print("---------------------")
    print('# train: %d, valid: %d, # test: %d' % (len(train_graphs), len(valid_graphs), len(test_graphs)))

    if cmd_args.sortpooling_k <= 1:
        num_nodes_list = sorted([
            g.num_nodes for g in train_graphs + valid_graphs + test_graphs])
        cmd_args.sortpooling_k = num_nodes_list[
            int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
        cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
        print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

    classifier = Classifier(cmd_args)
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()

    optimizer = optim.Adam(
        classifier.parameters(), lr=cmd_args.learning_rate, amsgrad=True,
        weight_decay=0.0008)

    train_idxes = list(range(len(train_graphs)))
    best_loss = None
    max_acc = 0.0

    val_loss, val_metrics, thr = evaluate(valid_graphs, classifier, list(range(len(valid_graphs))),
                                          return_best_thr=True)
    print("validation loss:", val_loss, "metrics", val_metrics, "thr:", thr)

    test_loss, test_metrics, _ = evaluate(test_graphs, classifier, list(range(len(test_graphs))), thr=thr)
    print("test loss:", test_loss, "metrics", test_metrics)

    for epoch in range(cmd_args.num_epochs):
        random.shuffle(train_idxes)
        classifier.train()
        avg_loss = loop_dataset(
            train_graphs, classifier, train_idxes, optimizer=optimizer)
        if not cmd_args.printAUC:
            avg_loss[2] = 0.0
        print('\033[92m\naverage training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2])) # noqa

        classifier.eval()
        # test_loss = loop_dataset(
        #     test_graphs, classifier, list(range(len(test_graphs))))
        val_loss, val_metrics, thr = evaluate(valid_graphs, classifier, list(range(len(valid_graphs))), return_best_thr=True)
        print("\nvalidation loss:", val_loss, "metrics", val_metrics, "thr:", thr)

        test_loss, test_metrics, _ = evaluate(test_graphs, classifier, list(range(len(test_graphs))), thr=thr)
        print("\ntest loss:", test_loss, "metrics", test_metrics)

        # if not cmd_args.printAUC:
        #     test_loss[2] = 0.0
        # print('\033[93maverage test of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, test_loss[0], test_loss[1], test_loss[2])) # noqa
        # max_acc = max(max_acc, test_loss[1])
        if val_metrics[-1] > max_acc:
            max_acc = val_metrics[-1]
            best_thr = thr
            best_valid_metrics = val_metrics
            best_test_metrics = test_metrics


    # with open('acc_result.txt', 'a+') as f:
    #     f.write(str(test_loss[1]) + '\n')
        # f.write(str(max_acc) + '\n')

    # if cmd_args.printAUC:
    #     with open('auc_results.txt', 'a+') as f:
    #         f.write(str(test_loss[2]) + '\n')

    # if cmd_args.extract_features:
    #     features, labels = classifier.output_features(train_graphs)
    #     labels = labels.type('torch.FloatTensor')
    #     np.savetxt('extracted_features_train.txt', torch.cat(
    #         [labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(),
    #             '%.4f')
    #     features, labels = classifier.output_features(test_graphs)
    #     labels = labels.type('torch.FloatTensor')
    #     np.savetxt('extracted_features_test.txt', torch.cat(
    #         [labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(),
    #             '%.4f')
