#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, house_dim, house_num, housd_num, thred,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = int(hidden_dim / house_dim)
        self.house_dim = house_dim
        self.house_num = house_num
        self.epsilon = 2.0
        self.housd_num = housd_num
        self.thred = thred
        if model_name == 'HousE' or model_name == 'HousE_plus':
            self.house_num = house_num + (2*self.housd_num)
        else:
            self.house_num = house_num

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / (self.hidden_dim * (self.house_dim ** 0.5))]),
            requires_grad=False
        )
        
        # self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        # self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        self.entity_dim = self.hidden_dim
        self.relation_dim = self.hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim, self.house_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim, self.house_dim*self.house_num))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.k_dir_head = nn.Parameter(torch.zeros(nrelation, 1, self.housd_num))
        nn.init.uniform_(
            tensor=self.k_dir_head,
            a=-0.01,
            b=+0.01
        )

        self.k_dir_tail = nn.Parameter(torch.zeros(nrelation, 1, self.housd_num))
        with torch.no_grad():
            self.k_dir_tail.data = - self.k_dir_head.data
        
        self.k_scale_head = nn.Parameter(torch.zeros(nrelation, self.relation_dim, self.housd_num))
        nn.init.uniform_(
            tensor=self.k_scale_head,
            a=-1,
            b=+1
        )

        self.k_scale_tail = nn.Parameter(torch.zeros(nrelation, self.relation_dim, self.housd_num))
        nn.init.uniform_(
            tensor=self.k_scale_tail,
            a=-1,
            b=+1
        )

        self.relation_weight = nn.Parameter(torch.zeros(nrelation, self.relation_dim, self.house_dim))
        nn.init.uniform_(
            tensor=self.relation_weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['HousE_r', 'HousE', 'HousE_r_plus', 'HousE_plus']:
            raise ValueError('model %s not supported' % model_name)

    def norm_embedding(self, mode):
        entity_embedding = self.entity_embedding
        r_list = torch.chunk(self.relation_embedding, self.house_num, 2)
        normed_r_list = []
        for i in range(self.house_num):
            r_i = torch.nn.functional.normalize(r_list[i], dim=2, p=2)
            normed_r_list.append(r_i)
        r = torch.cat(normed_r_list, dim=2)
        self.k_head = self.k_dir_head * torch.abs(self.k_scale_head)
        self.k_head[self.k_head>self.thred] = self.thred
        self.k_tail = self.k_dir_tail * torch.abs(self.k_scale_tail)
        self.k_tail[self.k_tail>self.thred] = self.thred
        return entity_embedding, r

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''

        entity_embedding, r = self.norm_embedding(mode)

        if mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, self.entity_dim, -1)

            k_head = torch.index_select(
                self.k_head,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            k_tail = torch.index_select(
                self.k_tail,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            re_weight = torch.index_select(
                self.relation_weight,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            relation = torch.index_select(
                r,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            k_head = torch.index_select(
                self.k_head,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            k_tail = torch.index_select(
                self.k_tail,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            re_weight = torch.index_select(
                self.relation_weight,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            relation = torch.index_select(
                r,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, self.entity_dim, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        if self.model_name == 'HousE_r':
            score = self.HousE_r(head, relation, tail, mode)
        elif self.model_name == 'HousE':
            score = self.HousE(head, relation, k_head, k_tail, tail, mode)
        elif self.model_name == 'HousE_r_plus':
            score = self.HousE_r_plus(head, relation, re_weight, tail, mode)
        elif self.model_name == 'HousE_plus':
            score = self.HousE_plus(head, relation, re_weight, k_head, k_tail, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def HousE_r(self, head, relation, tail, mode):

        r_list = torch.chunk(relation, self.house_num, 3)
        if mode == 'head-batch':
            for i in range(self.house_num):
                tail = tail - 2 * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]
            cos_score = tail - head
            cos_score = torch.sum(cos_score.norm(dim=3, p=2), dim=2)
        else:
            for i in range(self.house_num):
                j = self.house_num - 1 - i
                head = head - 2 * (r_list[j] * head).sum(dim=-1, keepdim=True) * r_list[j]
            cos_score = head - tail
            cos_score = torch.sum(cos_score.norm(dim=3, p=2), dim=2)
            
        score = self.gamma.item() - (cos_score)
        return score

    def HousE_r_plus(self, head, relation, re_weight, tail, mode):

        r_list = torch.chunk(relation, self.house_num, 3)

        if mode == 'head-batch':
            for i in range(self.house_num):
                tail = tail - 2 * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]
            tail = tail - re_weight
            cos_score = tail - head
            cos_score = torch.sum(cos_score.norm(dim=3, p=2), dim=2)
        else:
            head = head + re_weight
            for i in range(self.house_num):
                j = self.house_num - 1 - i
                head = head - 2 * (r_list[j] * head).sum(dim=-1, keepdim=True) * r_list[j]
            cos_score = head - tail
            cos_score = torch.sum(cos_score.norm(dim=3, p=2), dim=2)

        score = self.gamma.item() - (cos_score)
        return score

    def HousE(self, head, relation, k_head, k_tail, tail, mode):

        r_list = torch.chunk(relation, self.house_num, 3)

        if mode == 'head-batch':
            for i in range(self.housd_num):
                k_tail_i = k_tail[:, :, :, i].unsqueeze(dim=3)
                tail = tail - (0 + k_tail_i) * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]

            for i in range(self.housd_num, self.house_num-self.housd_num):
                tail = tail - 2 * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]
            
            for i in range(self.housd_num):
                k_head_i = k_head[:, :, :, i].unsqueeze(dim=3)
                head = head - (0 + k_head_i) * (r_list[self.house_num-1-i] * head).sum(dim=-1, keepdim=True) * r_list[self.house_num-1-i]

            cos_score = tail - head
            cos_score = torch.sum(cos_score.norm(dim=3, p=2), dim=2)
        else:
            for i in range(self.housd_num):
                k_head_i = k_head[:, :, :, i].unsqueeze(dim=3)
                head = head - (0 + k_head_i) * (r_list[self.house_num-1-i] * head).sum(dim=-1, keepdim=True) * r_list[self.house_num-1-i]
            
            for i in range(self.housd_num, self.house_num-self.housd_num):
                j = self.house_num - 1 - i
                head = head - 2 * (r_list[j] * head).sum(dim=-1, keepdim=True) * r_list[j]
            
            for i in range(self.housd_num):
                k_tail_i = k_tail[:, :, :, i].unsqueeze(dim=3)
                tail = tail - (0 + k_tail_i) * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]

            cos_score = head - tail
            cos_score = torch.sum(cos_score.norm(dim=3, p=2), dim=2)

        score = self.gamma.item() - (cos_score)
        return score
    
    def HousE_plus(self, head, relation, re_weight, k_head, k_tail, tail, mode):

        r_list = torch.chunk(relation, self.house_num, 3)

        if mode == 'head-batch':
            for i in range(self.housd_num):
                k_tail_i = k_tail[:, :, :, i].unsqueeze(dim=3)
                tail = tail - (0 + k_tail_i) * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]
            
            for i in range(self.housd_num, self.house_num-self.housd_num):
                tail = tail - 2 * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]
            
            tail = tail - re_weight
            for i in range(self.housd_num):
                k_head_i = k_head[:, :, :, i].unsqueeze(dim=3)
                head = head - (0 + k_head_i) * (r_list[self.house_num-1-i] * head).sum(dim=-1, keepdim=True) * r_list[self.house_num-1-i]
            
            cos_score = tail - head
            cos_score = torch.sum(cos_score.norm(dim=3, p=2), dim=2)
        else:
            for i in range(self.housd_num):
                k_head_i = k_head[:, :, :, i].unsqueeze(dim=3)
                head = head - (0 + k_head_i) * (r_list[self.house_num-1-i] * head).sum(dim=-1, keepdim=True) * r_list[self.house_num-1-i]
            head = head + re_weight
            
            for i in range(self.housd_num, self.house_num-self.housd_num):
                j = self.house_num - 1 - i
                head = head - 2 * (r_list[j] * head).sum(dim=-1, keepdim=True) * r_list[j]
            
            for i in range(self.housd_num):
                k_tail_i = k_tail[:, :, :, i].unsqueeze(dim=3)
                tail = tail - (0 + k_tail_i) * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]
            
            cos_score = head - tail
            cos_score = torch.sum(cos_score.norm(dim=3, p=2), dim=2)
        score = self.gamma.item() - (cos_score)
        return score
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        if mode == 'head-batch':
            pos_part = positive_sample[:, 0].unsqueeze(dim=1)
        else:
            pos_part = positive_sample[:, 2].unsqueeze(dim=1)
        positive_score = model((positive_sample, pos_part), mode=mode)

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            regularization = args.regularization * (
                model.entity_embedding.norm(dim=2, p=2).norm(dim=1, p=2).mean()
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
