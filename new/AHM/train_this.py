
import torch
import time
import numpy as np

from torch import nn
from logging import getLogger
from data import Vocab, NLP, S2SDataset
from utils import build_optimizer, init_seed, init_logger, init_device, read_configuration, collate_fn_graph_text, \
    format_time
from module import GraphEncoder,GraphPointer,MyPooler
from transformers import BartTokenizer, BartForConditionalGeneration, BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader



def compute_kd_loss(node_embeddings, desc_embeddings):#Teacher与Student对齐的loss
    assert node_embeddings.size() == desc_embeddings.size()
    mse_loss = nn.MSELoss(reduction='none')
    loss = mse_loss(node_embeddings, desc_embeddings)
    loss = loss.mean(dim=-1)
    loss = loss.mean()
    return loss

def run_train_batch(config, batch, teacher, student,pooler, external_optimizer, device ):#teacher, student,
    nodes, edges, types,node_lists,node_lens , relations,positions = batch
    new_nodes=list()
    #np.save("2020nodes.npy", nodes[0])
    for every_node in node_lists[0]:#为每一个节点进行嵌入    一共循环节点数量次
        input_ids = torch.LongTensor([every_node])  # 需要变成张量喂给模型
        input_ids=input_ids.to(device)
        output_dict = teacher(input_ids,output_hidden_states=True,return_dict=True)
        e=output_dict[3]#'encoder_last_hidden_state'  [1, 序列长度, 768]  =======f[6]
        pooled_output = pooler(e[0]) #这是长度为嵌入大小（768）的张量
        new_nodes.append(pooled_output)
    teacher_embeddings = torch.stack(new_nodes)  #2010-499*768的张量     节点数*嵌入维度
    teacher_embeddings_three = torch.unsqueeze(teacher_embeddings, dim=0)  #1*1040*768
    nodes=torch.as_tensor(nodes)
    nodes = nodes.to(device)   #1*1040
    student_embeddings = student(nodes, edges, types)    #1*当年节点数*768
    kd_loss = compute_kd_loss(student_embeddings,teacher_embeddings_three)

    external_optimizer.zero_grad()
    kd_loss.backward()
    external_optimizer.step()
    myres=  torch.squeeze(student_embeddings, dim=0)  #节点数*768
    myres= myres.cuda().data.cpu().numpy()
    np.save("2020newest_embedding.npy", myres)
    return kd_loss.item()

def train(config):
    init_logger(config)
    logger = getLogger()

    logger.info(config)
    init_seed(config["seed"], config["reproducibility"])
    device = init_device(config)

    logger.info("Build node and relation vocabularies.")
    vocabs = dict()
    vocabs["node"] = Vocab(config["node_vocab"])

    vocabs["relation"] = Vocab(config["relation_vocab"])

    logger.info("Build Teacher Model.")
    teacher = BartForConditionalGeneration.from_pretrained(config["teacher_dir"])
    teacher.requires_grad = False
    for para in teacher.parameters():
        para.requires_grad = False
    teacher.to(device)

    logger.info("Pooling node embedding(Teacher).")
    pooler = MyPooler(config["hidden_size"])
    pooler.requires_grad = False
    for para in pooler.parameters():
        para.requires_grad = False
    pooler.to(device)

    logger.info("Build Student Model.")
    student = GraphEncoder(vocabs["node"].size(), vocabs["relation"].size(),
                           config["gnn_layers"], config["embedding_size"], config["node_embedding"])
    student.to(device)

    external_parameters = []
    for p in student.parameters():
        if p.requires_grad:
            external_parameters.append(p)
    bart_tokenizer = BartTokenizer.from_pretrained(config["plm_dir"])
    external_optimizer = build_optimizer(external_parameters, config["external_learner"], config["external_lr"], config)
    scheduler1 = torch.optim.lr_scheduler.StepLR(external_optimizer, step_size=50, gamma=0.1) #学习率衰减
    kd_losses = []
    logger.info("Create training dataset.")
    for epoch_idx in range(config["start_epoch"], config["epochs"]):#epoch的迭代次数

        train_dataloader = DataLoader(
            S2SDataset(data_dir=config["data_dir"], dataset=config["dataset"],
                       tokenizer=bart_tokenizer, node_vocab=vocabs["node"], relation_vocab=vocabs["relation"],
                       num_samples=config["num_samples"], usage="2020"),
            batch_size=1,  # 默认为1
            shuffle=True,  # 每一轮数据会随机选   默认为false
            num_workers=0,  # num_workers=0,表示数据将在主进程中加载     默认为0
            drop_last=True,  # 告诉如何处理数据集长度除于batch_size余下的数据。True就抛弃，否则保留
            collate_fn=collate_fn_graph_text,  # 如何取样本的   这里使用的是自定义函数
            pin_memory=True)  # 会影响速度   可以不断尝试一下
        logger.info('Load train data done.')
        torch.cuda.empty_cache()
        teacher.train()
        student.train()
        pooler.train()
        train_gen_loss = 0
        t0 = time.time()
        for batch_idx, batch in enumerate(train_dataloader):
            kd_loss  = run_train_batch( config, batch, teacher, student, pooler, external_optimizer, device )
            logger.info(
                "Based one gragh ro train,so batch=1.Epoch {} batch {}: KD loss {} .".format(
                    epoch_idx, 1, kd_loss ))

            kd_losses.append(kd_loss)
            train_gen_loss=kd_losses[epoch_idx]
        train_gen_loss /= len(train_dataloader)  #train_gen_loss是float类型
        train_ppl = np.exp(train_gen_loss)  #也是一个float型
        training_time = format_time(time.time() - t0)
        logger.info("Epoch {}: training loss {}, perplexity {} ,time {}.\n".format(epoch_idx,
                                                                                   train_gen_loss,
                                                                                                train_ppl,
                                                                                                training_time))
        scheduler1.step()
        if epoch_idx%30==0:
            from matplotlib import pyplot as plt
            plt.plot(kd_losses)
            plt.show()
    print("ok")

def main():
    config = read_configuration("config.yaml")
    if config["mode"] == "train":
        train(config)



if __name__ == '__main__':
    main()
