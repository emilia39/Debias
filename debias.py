import argparse
import logging
import os
import pickle
import random
from typing import List, Tuple,Dict
from utils import *

import numpy as np
import torch
import sys
import time
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset,Subset
from tqdm import tqdm, trange
from arguments import get_args
from distance import calculate_group_to_one_relative_distance_asymmetric, JS_divergence

from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForPreTraining,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
timestr = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))


class LineByLineTextDataset(Dataset):
    def __init__(self, examples: list):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])



def create_dataset(data, dataset):
    d = dict()
    for key in data['example'].keys():
        d[key] = dataset(data['example'][key])
    return d

def fload_and_cache_examples(data, args, tokenizer):
    train_dataset = create_dataset(data['train'], LineByLineTextDataset)
    dev_dataset = create_dataset(data['dev'], LineByLineTextDataset)
    return {'train': train_dataset, 'dev': dev_dataset}

def split_data(attributes_examples_, neutral_examples, args):
    attributes_examples=attributes_examples_['sent']
    attributes_examples_neighbor=attributes_examples_['neighbor_sent']    
    data={'train':{'example':{'attribute0':[],'attribute1':[],'weight':[],'neutral':[]}},'dev':{'example':{'attribute0':[],'attribute1':[],'weight':[]},'neutral':[]}}
    if args.debias_type=='religion':
        data={'train':{'example':{'attribute0':[],'attribute1':[],'attribute2':[],'weight':[],'neutral':[]}},'dev':{'example':{'attribute0':[],'attribute1':[],'attribute2':[],'weight':[]},'neutral':[]}}
    for i, examples in enumerate(attributes_examples):
        idx_l = list(range(len(examples)))
        examples_neighbors=attributes_examples_neighbor[i]
        examples = [examples[idx] for idx in idx_l]
        examples_neighbors=[examples_neighbors[idx] for idx in idx_l]
        knn_neighbors_train=[]
        knn_weight_train=[]
        knn_neighbors_eval=[]
        knn_weight_eval=[]
        anchor_weight=[0.7]*len(examples)
        train_data_size=int(len(examples)*0.8)
        for neighbors in examples_neighbors[:train_data_size]:
            real_neighbor = [neighbor for neighbor in neighbors if neighbor is not None]
            weight = [(1 - 0.7) / len(real_neighbor)] * len(real_neighbor)
            knn_neighbors_train.extend(real_neighbor)
            knn_weight_train.extend(weight)
        for neighbors in examples_neighbors[train_data_size:]:
            real_neighbor = [neighbor for neighbor in neighbors if neighbor is not None]
            weight = [(1 - 0.7) / len(real_neighbor)] * len(real_neighbor)
            knn_neighbors_eval.extend(real_neighbor)
            knn_weight_eval.extend(weight)
        data['train']['example'][f'attribute{i}']= examples[:train_data_size]+knn_neighbors_train
        data['dev']['example'][f'attribute{i}']= examples[train_data_size:]+knn_neighbors_eval
    data['train']['example']['weight']=anchor_weight[:train_data_size]+knn_weight_train
    data['dev']['example']['weight']=anchor_weight[train_data_size:]+knn_weight_eval
    
    idx_l = list(range(len(neutral_examples)))
    neutral_examples = [neutral_examples[idx] for idx in idx_l]
    train_data_size=int(len(neutral_examples)*0.8)

    data['train']['example']['neutral'] = neutral_examples[:train_data_size]
    data['dev']['example']['neutral'] = neutral_examples[train_data_size:]
    return data

def split_data2(tar1_tokenized,tar2_tokenized,tar3_tokenized,args):
    data = {'train': {'example': {}}, 'dev': {'example': {}}}
    idx_l = list(range(len(tar1_tokenized)))
    tar1_tokenized_examples = [tar1_tokenized[idx] for idx in idx_l]
    tar2_tokenized_examples = [tar2_tokenized[idx] for idx in idx_l]
    train_data_size=int(len(idx_l)*0.8)
    data['train']['example']['token0'] = tar1_tokenized_examples[:train_data_size]
    data['dev']['example']['token0'] = tar1_tokenized_examples[train_data_size:]
    data['train']['example']['token1'] = tar2_tokenized_examples[:train_data_size]
    data['dev']['example']['token1'] = tar2_tokenized_examples[train_data_size:]
    if args.debias_type=='religion':
        tar3_tokenized_examples = [tar3_tokenized[idx] for idx in idx_l]
        data['train']['example']['token2'] = tar3_tokenized_examples[:train_data_size]
        data['dev']['example']['token2'] = tar3_tokenized_examples[train_data_size:]
    return data
def get_data1(datafile1,tokenizer,debias_type):
    data={"attributes_examples":{"sent":[[],[]],"neighbor_sent":[[],[]]},"neutral_examples":[]}
    if debias_type=='religion':
        data={"attributes_examples":{"sent":[[],[],[]],"neighbor_sent":[[],[],[]]},"neutral_examples":[]}
    with open(datafile1, 'rb') as file:
        XT_data = pickle.load(file)
        for index,l in tqdm(enumerate(XT_data['attribute0']['sent'][:])):
            orig_line=l.strip()
            embed=tokenizer.encode(orig_line, add_special_tokens=True)
            data["attributes_examples"]["sent"][0].append(embed)
            neighbor_embeds=[]
            for j,neighbor in enumerate(XT_data['attribute0']['neighbor_sent'][index]):
                if neighbor!=None:
                    neighbor=neighbor.strip()
                    neighbor_embed=tokenizer.encode(neighbor,add_special_tokens=True)
                    neighbor_embeds.append(neighbor_embed)
            data["attributes_examples"]["neighbor_sent"][0].append(neighbor_embeds)   
            data["neutral_examples"].append(embed)
        for index,l_ in tqdm(enumerate(XT_data['attribute1']['sent'][:])):
            orig_line_=l_.strip()
            embed_=tokenizer.encode(orig_line_, add_special_tokens=True)
            data["attributes_examples"]["sent"][1].append(embed_)
            neighbor_embeds=[]
            for neighbor in XT_data['attribute1']['neighbor_sent'][index]:
                if neighbor!=None:
                    neighbor=neighbor.strip()
                    neighbor_embed=tokenizer.encode(neighbor,add_special_tokens=True)
                    neighbor_embeds.append(neighbor_embed)
            data["attributes_examples"]["neighbor_sent"][1].append(neighbor_embeds)
            data["neutral_examples"].append(embed_)
        if debias_type=='religion':
            for index,l__ in tqdm(enumerate(XT_data['attribute2']['sent'][:])):
                orig_line__=l__.strip()
                embed__=tokenizer.encode(orig_line__, add_special_tokens=True)
                data["attributes_examples"]["sent"][2].append(embed__)
                neighbor_embeds=[]
                for neighbor in XT_data['attribute2']['neighbor_sent'][index]:
                    if neighbor!=None:
                        neighbor=neighbor.strip()
                        neighbor_embed=tokenizer.encode(neighbor,add_special_tokens=True)
                        neighbor_embeds.append(neighbor_embed)
                data["attributes_examples"]["neighbor_sent"][2].append(neighbor_embeds)
                data["neutral_examples"].append(embed__)
    return data

def get_data2(datafile2,tokenizer,debias_type):
    data={"token0":[],"token1":[]}
    if debias_type=='religion':
        data={"token0":[],"token1":[],'token2':[]}
    with open(datafile2, 'rb') as file:
        XNT_data = pickle.load(file)
        for l in tqdm(XNT_data['attribute0']['sent'][:]):
            orig_line=l.strip()
            assert orig_line.count("[MASK]")==1,orig_line
            embed=tokenizer.encode(orig_line, add_special_tokens=True)
            data["token0"].append(embed)
        for l_ in tqdm(XNT_data['attribute1']['sent'][:]):
            orig_line_=l_.strip()
            assert orig_line_.count("[MASK]")==1,orig_line_
            embed_=tokenizer.encode(orig_line_, add_special_tokens=True)
            data["token1"].append(embed_)
        if debias_type=='religion':
            for l__ in tqdm(XNT_data['attribute2']['sent'][:]):
                orig_line__=l__.strip()
                assert orig_line__.count("[MASK]")==1,orig_line__
                embed__=tokenizer.encode(orig_line__, add_special_tokens=True)
                data["token2"].append(embed__)
    return data

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def chunck_knn(datasets):
    new_knn_datasets = []
    new_datasets_weights = []
    for neighbors in datasets:
        real_neighbor = [neighbor for neighbor in neighbors if neighbor is not None]
        weight = [(1 - 0.7) / len(real_neighbor)] * len(real_neighbor)
        new_knn_datasets.extend(real_neighbor)
        new_datasets_weights.extend(weight)
    return new_knn_datasets, new_datasets_weights

def create_dataloader(args, datasets, tokenizer, train=False):
    def collate(batch: List[torch.Tensor]):
        mask_idxs=[]
        padded_examples = pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)
        for example in padded_examples:
            mask_idx=torch.where(example==tokenizer.mask_token_id)[0]
            mask_idxs.append(mask_idx)
        examples_mask_index=torch.stack(mask_idxs)
        examples_mask_index=torch.squeeze(examples_mask_index,dim=1)
        examples_attention_mask = torch.zeros_like(padded_examples, dtype=torch.int32)
        examples_attention_mask[torch.where(padded_examples != tokenizer.pad_token_id)] = 1
        token_type_ids = torch.zeros_like(padded_examples, dtype=torch.int32)
        # Find indices where padded_examples is not equal to pad_token_id
        non_pad_indices = torch.where(padded_examples != tokenizer.pad_token_id)
        # Set token_type_ids to 1 at non-pad indices
        token_type_ids[non_pad_indices] = 1
        return padded_examples,token_type_ids,examples_attention_mask,examples_mask_index

    dataloaders = {}
    example_num = 0
    data_distribution = []
    train_batch_size_={'attribute0':8,'token0':32}
    eval_batch_size_={'attribute0':8,'token0':32}

    assert len(datasets['attribute0'])==len(datasets['attribute1'])==len(datasets['weight'])
    assert len(datasets['token0'])==len(datasets['token1'])

    shuffle_order1 = torch.randperm(len(datasets['attribute0']))
    shuffle_order2 = torch.randperm(len(datasets['token0']))
    shuffle_order3 = torch.randperm(len(datasets['neutral']))

    for key, dataset in datasets.items():
        if key in ['token0', 'token1','token2']:
            shuffle_order = shuffle_order2
            train_batch_size = train_batch_size_['token0']
            eval_batch_size = eval_batch_size_['token0']
            min_size = len(datasets['token0'])
        elif key in ['weight','attribute0','attribute1','attribute2']:
            shuffle_order = shuffle_order1
            train_batch_size = train_batch_size_['attribute0']
            eval_batch_size = eval_batch_size_['attribute0']
            min_size = len(datasets['attribute0'])
        elif key == 'neutral':
            shuffle_order = shuffle_order3
            train_batch_size = train_batch_size_['attribute0']
            eval_batch_size = eval_batch_size_['attribute0']
            min_size = len(datasets['neutral'])

        example_num += len(dataset)
        dataset=Subset(dataset, shuffle_order)
        if key=='weight':
            if train:
                dataloaders[key] = iter(DataLoader(dataset, batch_size=train_batch_size, shuffle=False))
                data_distribution += [key for _ in range(int(min_size / train_batch_size))]
            else:
                dataloaders[key] = iter(DataLoader(dataset, batch_size=eval_batch_size, shuffle=False))
                data_distribution += [key for _ in range(int(min_size / eval_batch_size))]
        else:
            if train:
                dataloaders[key] = iter(DataLoader(dataset, batch_size=train_batch_size, collate_fn=collate, shuffle=False))
                data_distribution += [key for _ in range(int(min_size / train_batch_size))]
            else:
                dataloaders[key] = iter(DataLoader(dataset, batch_size=eval_batch_size, collate_fn=collate , shuffle=False))
                data_distribution += [key for _ in range(int(min_size / eval_batch_size))]

    return dataloaders, example_num, data_distribution

def train(args, data, datasets, model: PreTrainedModel,original_model,tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """Train the model"""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    train_datasets = datasets['train']
    dev_datasets = datasets['dev']

    train_dataloaders, train_example_num, train_distribution = create_dataloader(args, train_datasets, tokenizer, train=True)
    dev_dataloaders, dev_example_num, dev_distribution = create_dataloader(args, dev_datasets, tokenizer, train=False)

    train_iter_num = sum([len(dataloader) for dataloader in train_dataloaders.values()])

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (train_iter_num // args.gradient_accumulation_steps) + 1
    else:
        t_total = train_iter_num // args.gradient_accumulation_steps * args.num_train_epochs

    model = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model.resize_token_embeddings(len(tokenizer))

    original_model = original_model.module if hasattr(original_model, "module") else original_model  # Take care of distributed/parallel training
    original_model.resize_token_embeddings(len(tokenizer))

    # Prepare optimizer and scheduler (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    #paired sentences
    train_example_num=train_example_num//2
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        original_model = torch.nn.DataParallel(original_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        original_model = torch.nn.parallel.DistributedDataParallel(
            original_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
  
    # Train!
    logger.info("***** Running training *****")
    logger.info("Num examples = %d", train_example_num)
    logger.info("Num Epochs = %d", args.num_train_epochs)
    logger.info("Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    best_loss = float('inf')
    best_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (train_iter_num // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (train_iter_num // args.gradient_accumulation_steps)

            logger.info("Continuing training from checkpoint, will skip to saved global_step")
            logger.info("Continuing training from epoch %d", epochs_trained)
            logger.info("Continuing training from global step %d", global_step)
            logger.info("Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("Starting fine-tuning.")
    model.zero_grad()
    original_model.zero_grad()

    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])

    def inner_product(x, y):
        return torch.mean(torch.sum(y * x, 3))

    def mean_square(x, y, idx):
        return torch.mean(torch.mean((y - x) ** 2, idx))

    def save_best_model(best_loss, best_step, dev_dataloaders):
        if (args.local_rank == -1):  # Only evaluate when single GPU otherwise metrics may not average well
            eval_loss = evaluate(model, attributes_hiddens, dev_dataloaders)
            logger.info("global_step = %s, evaluate loss = %s", global_step, eval_loss)
            tb_writer.add_scalar("eval_loss", eval_loss, global_step)
        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)

        if eval_loss < best_loss:
            best_loss = eval_loss
            best_step = global_step
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, timestr, 'best_model_ckpt')
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)

            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", output_dir)
        logger.info("best_step = %s, best loss = %s", best_step, best_loss)

        return best_loss, best_step

    def get_hiddens_of_model(input, input_attention_mask):
        model.zero_grad()
        if args.model_type == 'bert':
            hiddens =model.bert(input).hidden_states
        elif args.model_type == 'roberta':
            hiddens = model.roberta(input).hidden_states
        elif args.model_type == 'albert':
            hiddens =model.albert(input).hidden_states
        return hiddens
    
    def attribute_vector_example(args):
        d=2
        attributes_hiddens = {f'attribute{i}': [] for i in range(d)}
        weights=[]

        dataloaders, _, distribution = create_dataloader(args, train_datasets, tokenizer, train=True)
        for key in distribution:
            if key == 'attribute0' or key=='attribute1' or key=='attribute2':
                inputs,token_type_ids,inputs_attention_mask,inputs_mask_ids= next(dataloaders[key])
                inputs = inputs.to(args.device)
                inputs_attention_mask = inputs_attention_mask.to(args.device)
                hiddens = get_hiddens_of_model(inputs, inputs_attention_mask)
                hiddens = torch.stack(hiddens, 2)
                attributes_hiddens[key].append(torch.mean(hiddens, 1))
            elif key =='weight':
                weights.append(next(dataloaders[key]))
        weights=torch.cat(weights)
        attribute_size = len(data['train']['example'])
        for i in range(d):
            expanded_weights = weights.unsqueeze(-1).unsqueeze(-1).to(args.device)
            weighted_attributes = torch.cat(attributes_hiddens[f'attribute{i}'],0) * expanded_weights
            weighted_sum = torch.sum(weighted_attributes, dim=0)
            sum_of_weights = torch.sum(expanded_weights, dim=0)

            weighted_average = weighted_sum / sum_of_weights   
            attributes_hiddens[f'attribute{i}'] = weighted_average.detach().unsqueeze(0)
        return attributes_hiddens


    def forward(attributes_hiddens,dataloaders,data_key,args):
        loss = 0
        if args.debias_type=='religion':
            d=3
        else:
            d=2   
        if 'neutral'==data_key:
            inputs = next(dataloaders["neutral"])
            inputs,token_type_ids, inputs_attention_mask,inputs_mask_ids = inputs
            inputs = inputs.to(args.device)
            inputs_attention_mask = inputs_attention_mask.to(args.device)
            all_layer_hiddens = model(inputs, inputs_attention_mask).hidden_states
            all_layer_hiddens = torch.stack(all_layer_hiddens, 2)
            if args.debias_layer == 'all':
                target_layer_hiddens = all_layer_hiddens
            else:
                if args.debias_layer == 'first':
                    idx = 0
                elif args.debias_layer == 'last':
                    idx = -1
                target_layer_hiddens = all_layer_hiddens[:,:,idx]
                target_layer_hiddens = target_layer_hiddens.unsqueeze(2)
                attributes_hiddens = {key: value[:,idx,:].unsqueeze(1) for key, value in attributes_hiddens.items()}
            if args.loss_target == 'sentence':
                attributes_hiddens = {key: value.unsqueeze(1) for key, value in attributes_hiddens.items()}
            elif args.loss_target == 'token':
                target_layer_hiddens = torch.mean(target_layer_hiddens, 1).unsqueeze(1)
  
            attributes_hiddens = torch.cat(list(attributes_hiddens.values()), dim=0)
            relative_distance = calculate_group_to_one_relative_distance_asymmetric(target_layer_hiddens, attributes_hiddens)
            relative_distance_shape0 = relative_distance.shape[0]
            for i in range(relative_distance_shape0):
                for j in range(i + 1, relative_distance_shape0):
                    loss += JS_divergence(relative_distance[i], relative_distance[j])
            loss /= relative_distance_shape0 * (relative_distance_shape0 - 1) / 2
            loss=loss*args.alpha
        elif 'token0'==data_key:
            tar_predictions_logits = {f'token{i}': [] for i in range(d)}
            for key in [f'token{i}' for i in range(d)]:
                inputs,token_type_ids,inputs_attention_mask,inputs_mask_id= next(dataloaders[key])
                is_empty = (inputs_mask_id.flatten().shape[0] == 0)
                assert is_empty==False,key
                inputs = inputs.to(args.device)
                token_type_ids=token_type_ids.long()
                token_type_ids=token_type_ids.to(args.device)
                inputs_attention_mask = inputs_attention_mask.to(args.device)
                inputs_ = {
                        'input_ids': inputs,
                        'token_type_ids':token_type_ids,
                        'attention_mask': inputs_attention_mask
                            }
                tar_predictions = model(**inputs_)
                tar_predictions_logits[key] = tar_predictions.prediction_logits[torch.arange(tar_predictions.prediction_logits.size(0)), inputs_mask_id]
            tar1_predictions_logits,tar2_predictions_logits=tar_predictions_logits["token0"],tar_predictions_logits["token1"]
            loss = jsd_model(tar1_predictions_logits,tar2_predictions_logits)
            if d==3:
                tar3_predictions_logits=tar_predictions_logits["token2"]
                loss = loss+jsd_model(tar1_predictions_logits,tar3_predictions_logits)
            loss=loss*args.beta

        else:
            inputs = next(dataloaders[data_key])
            inputs,token_type_ids, inputs_attention_mask,inputs_mask_ids = inputs
            inputs = inputs.to(args.device)
            inputs_attention_mask = inputs_attention_mask.to(args.device)
            all_layer_hiddens = model(inputs, inputs_attention_mask).hidden_states
            all_layer_hiddens = torch.stack(all_layer_hiddens, 2)
            with torch.no_grad():
                all_layer_original_hiddens = original_model(inputs, inputs_attention_mask).hidden_states
            all_original_hiddens =  torch.stack(all_layer_original_hiddens, 2)
            all_original_hiddens = all_original_hiddens.detach()    
            loss += criterion_ms(all_layer_hiddens, all_original_hiddens, 3)*0.7*10e-5
        return loss

    def evaluate(model, attributes_hiddens, dev_dataloaders, prefix=""):
        eval_output_dir = args.output_dir

        if args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir, exist_ok=True)
        # Note that DistributedSampler samples randomly
        # multi-gpu evaluate
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("Num examples = %d", dev_example_num)
        eval_loss = 0.0
        model.eval()
        #criterion.eval()

        for data_key in tqdm(dev_distribution):
            if data_key=='weight':
                continue
            if data_key=='token1':
                continue
            with torch.no_grad():
                loss = forward(attributes_hiddens, dev_dataloaders,data_key,args)
                eval_loss += loss.item()
                model.zero_grad()
        return eval_loss

    criterion_ms = mean_square
    #criterion_ip = inner_product
    train_loss = 0.0
    jsd_model = JSD()

    for _ in train_iterator:

        random.shuffle(train_distribution)
        epoch_iterator = tqdm(train_distribution, desc="Iteration", disable=args.local_rank not in [-1, 0])

        model.eval()
        with torch.no_grad():
            attributes_hiddens= attribute_vector_example(args)

        for step, data_key in enumerate(epoch_iterator):
            if data_key=="weight":
                continue
            if data_key=='token1':
                continue
            model.train()
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            loss = forward(attributes_hiddens,train_dataloaders,data_key,args)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            train_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logger.info("global_step = %s, train loss = %s", global_step, train_loss)
                    tb_writer.add_scalar("train_loss", train_loss, global_step)
                    train_loss = 0.0
                    # Log metrics
                    best_loss, best_step = save_best_model(best_loss, best_step, dev_dataloaders)
                    dev_dataloaders, dev_example_num, dev_distribution = create_dataloader(args, dev_datasets, tokenizer, train=False)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
            train_dataloaders, train_example_num, train_distribution = create_dataloader(args, train_datasets, tokenizer, train=True)
                    
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    dev_dataloaders, dev_example_num, dev_distribution = create_dataloader(args, dev_datasets, tokenizer, train=False)
    best_loss, best_step = save_best_model(best_loss, best_step, dev_dataloaders)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

def main():
    model_args, args = get_args()
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Set seed
    set_seed(args)

    # Setup logging
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(f'{args.log_dir}/{timestr}.log')],
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir, revision=model_args.model_revision)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir, revision=model_args.model_revision)
    else:
        # When we release a pip version exposing CONFIG_MAPPING,
        # we can do `config = CONFIG_MAPPING[args.model_type]()`.
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --config_name"
        )

    config.output_hidden_states = 'true'
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if args.block_size <= 0:
        args.block_size = tokenizer.model_max_length
        # Our input block size will be the max possible for the model
    else:
        try:
            args.block_size = min(args.block_size, tokenizer.model_max_length)
        except:
            args.block_size = min(args.block_size, tokenizer.max_len)

    if model_args.model_name_or_path:
        model =AutoModelForPreTraining.from_pretrained(args.model_name_or_path, output_hidden_states=True)
        original_model =AutoModelForPreTraining.from_pretrained(args.model_name_or_path, output_hidden_states=True)
    
    else:
        raise ValueError()
        
    # GPT-2 and GPT do not have pad.
    if tokenizer._pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model.resize_token_embeddings(len(tokenizer))
        original_model.resize_token_embeddings(len(tokenizer))

    model.to(args.device)
    original_model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)
    
    data=get_data1('data/train_data/XT_data.pk',tokenizer,args.debias_type)
    attributes_examples = data['attributes_examples']
    neutral_examples = data['neutral_examples']
    splited_data = split_data(attributes_examples, neutral_examples, args)

    data2=get_data2("data/train_data/bias_data/XNTmask_data.pk",tokenizer,args.debias_type)
    tar1_tokenized=data2["token0"]
    tar2_tokenized=data2["token1"]
    if args.debias_type=='religion':
        tar3_tokenized=data2["token2"]
    else:
        tar3_tokenized=[]
    splited_data2 = split_data2(tar1_tokenized,tar2_tokenized,tar3_tokenized,args)

    datasets1 = fload_and_cache_examples(splited_data, args, tokenizer)
    datasets2 = fload_and_cache_examples(splited_data2, args, tokenizer)
    datasets = {
    "train": {
        "attribute0": datasets1["train"]["attribute0"],
        "attribute1": datasets1["train"]["attribute1"],
        "neutral": datasets1["train"]["neutral"],
        "weight":datasets1['train']['weight'],
        "token0": datasets2["train"]["token0"],
        "token1": datasets2["train"]["token1"],

    },
    "dev": {
        "attribute0": datasets1["dev"]["attribute0"],
        "attribute1": datasets1["dev"]["attribute1"],
        "neutral": datasets1["dev"]["neutral"],
        "weight":datasets1['dev']['weight'],
        "token0": datasets2["dev"]["token0"],
        "token1": datasets2["dev"]["token1"],
    }
    }
    if args.debias_type=='religion':
        datasets['train']['attribute2']=datasets1["train"]["attribute2"]
        datasets['train']['token2']=datasets2["train"]["token2"]
        datasets['dev']['attribute2']=datasets1["dev"]["attribute2"]
        datasets['dev']['token2']=datasets2["dev"]["token2"]

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

    if args.local_rank == 0:
        torch.distributed.barrier()

    train(args, splited_data, datasets, model, original_model,tokenizer)


if __name__ == "__main__":
    main()