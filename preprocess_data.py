from transformers import BertTokenizer,BertModel
from transformers import RobertaTokenizer,RobertaModel
from transformers import AlbertTokenizer,AlbertModel
import torch
from tqdm import tqdm
import time
import argparse
import faiss
import numpy as np
import pickle
 
parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_type",
    default="bert",
    type=str,
    help="Choose from ['bert','roberta','albert']",
)

parser.add_argument(
    "--model_name_or_path",
    default="bert-base-uncased",
    type=str,
    help="Path to pretrained model or model identifier from huggingface.co/models",
)

parser.add_argument(
    "--debias_type",
    default='gender',
    type=str,
    choices=['gender'],
)

parser.add_argument(
    "--BS",
    default=8,
    type=int,
    help="batch size of the data fed into the model",
)

def KNN_split(embeddings_file):
    """
    Calculate similarity using Faiss.
    :return:
    """
    dim = 768
    # The method for calculating similarity METRIC_INNER_PRODUCT => inner product (cosine similarity)
    measure = faiss.METRIC_INNER_PRODUCT
    param = "HNSW64"
    # use a single GPU
    #res = faiss.StandardGpuResources()
    # index = faiss.index_factory(dim, param, measure)
    # gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    embeddings = []

    with open(embeddings_file, "r", encoding='utf8') as reader:
        lines=reader.readlines()
        count=0
        for line in lines[:100000]:
            line.strip()
            parts = line.split(' ')
            assert len(parts) == dim
            v = list(map(lambda x: float(x), parts[0:]))
            embeddings.append(v)

        # faiss index
        index = faiss.index_factory(dim, param, measure)
        #gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index=index
        embeddings = np.array(embeddings, dtype=np.float32)
        gpu_index.train(embeddings)
        print("Training completed")
        gpu_index.add(embeddings)
        #print(gpu_index.ntotal)
        # KNN_6
        D, I = gpu_index.search(embeddings, 6)
        threshold=227
        #ow_indices, col_indices = np.where(D > threshold)
        T_MASK = np.where(D > threshold,I,0)
        T_MASK[:, 0] = 0
        T_MASK_ = np.where(T_MASK > 0,1,0)
        #Count the number of elements in the row that satisfy the condition except themselves
        count_per_row = np.sum(T_MASK_, axis=1)
        #Basis for dividing X_T
        print(np.sum(count_per_row > 0, axis=0),len(lines))
        XT_idx=np.where(count_per_row>0)
        XNT_idx=np.where(count_per_row==0)
        return XT_idx,T_MASK,count_per_row,XNT_idx
#Modify these files(In the case of gender, to take only 100,000 pieces of data for the experiment.)
def save_pk(embeddings_file,keys):
    biastype = embeddings_file.split("/")[2]
    XT_idx,T_MASK,count_per_row,XNT_idx=KNN_split(embeddings_file)
    XT_idx,T_MASK,count_per_row,XNT_idx=XT_idx[0].tolist(),T_MASK.tolist(),count_per_row.tolist(),XNT_idx[0].tolist()
    XT_data = {}
    XTmask_data = {}
    XNT_data = {}
    XNTmask_data = {}
    for key in keys:
        XT_data[key] = {}
        XTmask_data[key] = {}
        XNT_data[key] = {}
        XNTmask_data[key] = {}
    T_attribute0_sent=[]
    T_attribute1_sent=[]
    T_attribute2_sent=[]
    if biastype=='gender':
        with open("data/gpt_data/gender/generate_male.txt", "r", encoding='utf8') as f1:
            attribute0_lines=f1.readlines()[:100000]
        with open("data/gpt_data/gender/male.txt", "r", encoding='utf8') as f1_:
            attribute0mask_lines=f1_.readlines()[:100000]
        with open("data/gpt_data/gender/generate_female.txt", "r", encoding='utf8') as f2:
            attribute1_lines=f2.readlines()[:100000]
        with open("data/gpt_data/gender/female.txt", "r", encoding='utf8') as f2_:
            attribute1mask_lines=f2_.readlines()[:100000]
    for i in XT_idx:
        # print(i,NT_MASK[i])
        T_attribute0_sent.append([attribute0_lines[int(index)] if index != 0 else None for index in T_MASK[i]])
        T_attribute1_sent.append([attribute1_lines[int(index)] if index != 0 else None for index in T_MASK[i]])
        if biastype=='religion':
            T_attribute2_sent.append([attribute2_lines[int(index)] if index != 0 else None for index in T_MASK[i]])

    neighbor_num=count_per_row
    XT_data['attribute0']['sent']=[attribute0_lines[int(i)] for i in XT_idx]
    XTmask_data['attribute0']['sent']=[attribute0mask_lines[int(i)] for i in XT_idx]
    XNT_data['attribute0']['sent']=[attribute0_lines[int(i)] for i in XNT_idx]
    XNTmask_data['attribute0']['sent']=[attribute0mask_lines[int(i)] for i in XNT_idx]
    XT_data['attribute1']['sent']=[attribute1_lines[int(i)] for i in XT_idx]
    XTmask_data['attribute1']['sent']=[attribute1mask_lines[int(i)] for i in XT_idx]
    XNT_data['attribute1']['sent']=[attribute1_lines[int(i)] for i in XNT_idx]
    XNTmask_data['attribute1']['sent']=[attribute1mask_lines[int(i)] for i in XNT_idx]

    XT_data['attribute0']['neighbor_sent']=[]
    XT_data['attribute1']['neighbor_sent']=[]

    for i in range(len(XT_idx)):
        XT_data['attribute0']['neighbor_sent'].append(T_attribute0_sent[i])
        XT_data['attribute1']['neighbor_sent'].append(T_attribute1_sent[i])
    XT_data['attribute0']['neighbor_num']=neighbor_num
    XT_data['attribute1']['neighbor_num']=neighbor_num

    if biastype=='religion':
        XT_data['attribute2']['sent']=[attribute2_lines[int(i)] for i in XT_idx]
        XTmask_data['attribute2']['sent']=[attribute2mask_lines[int(i)] for i in XT_idx]
        XNT_data['attribute2']['sent']=[attribute2_lines[int(i)] for i in XNT_idx]
        XNTmask_data['attribute2']['sent']=[attribute2mask_lines[int(i)] for i in XNT_idx]
        XT_data['attribute2']['neighbor_sent']=[]
        for i in range(len(XT_idx)):
            XT_data['attribute2']['neighbor_sent'].append(T_attribute2_sent[i])
        XT_data['attribute2']['neighbor_num']=neighbor_num

    with open('data/train_data/XT_data.pk', 'wb') as file1:
        pickle.dump(XT_data, file1)
    with open('data/train_data/XTmask_data.pk', 'wb') as file4:
        pickle.dump(XTmask_data, file4)
    with open('data/train_data/XNT_data.pk', 'wb') as file2:
        pickle.dump(XNT_data, file2)
    with open('data/train_data/XNTmask_data.pk', 'wb') as file3:
        pickle.dump(XNTmask_data, file3)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        model = BertModel.from_pretrained(args.model_name_or_path)
    elif args.model_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        model = RobertaModel.from_pretrained(args.model_name_or_path)
    elif args.model_type == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained(args.model_name_or_path)
        model = AlbertModel.from_pretrained(args.model_name_or_path)
    else:
        raise NotImplementedError("not implemented!")
    if args.debias_type=='gender':
        gpt_sentences_file='data/gpt_data/gender/generate_male.txt'
        keys=['attribute0','attribute1']
    #It is recommended to only process small-scale data. 
    #If you want to process all the sentences (270k), it is suggested to divide them into several small files.
    with open(gpt_sentences_file, 'r', encoding='utf-8') as file:
        sentences = file.readlines()[:]
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=512)
    embeddings = []
    start_time=time.time()
    embeddings_file=gpt_sentences_file.split('.txt')[0]+'_embedding.txt'
    # with open(embeddings_file, 'a', encoding='utf-8') as emb_file:
    #     for i in tqdm(range(0, len(sentences), args.BS),desc="Converting data", dynamic_ncols=True):
    #         batch_inputs = {k: v[i:i+args.BS] for k, v in inputs.items()}
    #         with torch.no_grad():
    #             outputs = model(**batch_inputs)
    #         batch_embeddings = outputs.last_hidden_state[:, 0, :]
    #         for i in range(len(batch_embeddings)):
    #             sentence_embedding = outputs.last_hidden_state[i, 0, :].numpy()  
    #             emb_file.write(' '.join(str(x) for x in sentence_embedding) + '\n')
    #         embeddings.append(batch_embeddings)
    save_pk(embeddings_file,keys)
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time consumed for data dividing:", total_time, "seconds")
