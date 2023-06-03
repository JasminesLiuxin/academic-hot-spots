import os
import torch
import xlrd2
from xlutils.copy import copy;
import time
import numpy as np
from torch import nn
from logging import getLogger
from data_2023 import Vocab, S2SDataset
from utils import build_optimizer, init_seed, init_logger, init_device, read_configuration, collate_fn_graph_text, \
    format_time
from module import GraphEncoder,MyPooler
from transformers import BartTokenizer, BartForConditionalGeneration, BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from De_RNN_CNN_2023 import CNN_decoder
def oushi_simi1(x, y):
    sum_XYSimlar = 0
    for i in range(0, 768):
        # 两个数的欧几里得距离
        XYdistiance = np.sqrt(np.sum(np.square(x[i] - y[i])))
        # 欧氏距离定义的相似度,距离越小相似度越大

        # 获取相似度和
        sum_XYSimlar = sum_XYSimlar + XYdistiance
    return sum_XYSimlar / 768
def oushi_simi(x, y):
    sum_XYSimlar = 0
    for i in range(0, 768):
        # 两个数的欧几里得距离
        XYdistiance = np.sqrt(np.sum(np.square(x[i] - y[i])))
        # 欧氏距离定义的相似度,距离越小相似度越大
        XYSimlar = 1 / (1 + XYdistiance)
        # 获取相似度和
        sum_XYSimlar = sum_XYSimlar + XYSimlar
    return sum_XYSimlar / 768
def compute_kd_loss(node_embeddings, desc_embeddings):#Teacher与Student对齐的loss
    assert node_embeddings.size() == desc_embeddings.size()
    mse_loss = nn.MSELoss(reduction='none')
    loss = mse_loss(node_embeddings, desc_embeddings)
    loss = loss.mean(dim=-1)
    loss = loss.mean()
    return loss
from sklearn.metrics.pairwise import cosine_similarity
def similarity_loss(pre_e, auc_e):  # candidates表示预测的标签, references表示真实的标签     5*20
    sim = 1
    num=0
    for every in pre_e:
        max = 0
        for every_a in auc_e:
            tmp=abs(cosine_similarity([every], [every_a])[0][0])
            if(tmp>max):
                max = tmp
        if (max > 0):
            sim = sim + max;
        if (max > 0.9):
            num=num+1;
    return 1/sim,num
def similarity_evaluate(pre_e, auc_e):  # candidates表示预测的标签, references表示真实的标签     5*20
    cs=0
    len = 0
    sim = 0
    for every in pre_e:
        max = 0
        for every_a in auc_e:
            tmp = abs(cosine_similarity([every], [every_a])[0][0])
            if (tmp > max):
                max = tmp  # 这个词与集合的最大相似度
        if (max > 0):
            sim = sim + max;
            len = len + 1;

    if (len > 0):
        cs=sim / len

    es=0
    len = 0
    sim = 0

    for every in pre_e:
        max=0
        for every_a in auc_e:
            if (oushi_simi1(every, every_a) > max):
                max = oushi_simi1(every, every_a)  # 这个词与集合的最大相似度

        if (max > 0):
            len = len + 1;
            sim = sim + max;
    if (len > 0):
        es=sim / len

    ass=0
    len = 0
    sim = 0
    for every in pre_e:
        max=0
        for every_a in auc_e:
            if (abs(cosine_similarity([every], [every_a]))[0][0] > max):
                max = abs(cosine_similarity([every], [every_a]))[0][0]  # 这个词与集合的最大相似度
        sim = sim + max;
        if (max > 0):
            len = len + 1;
    if (len > 0):
        ass=sim / len

    acs=0
    len = 0
    sim = 0
    for every in pre_e:
        max=0
        for every_a in auc_e:
            if (abs(cosine_similarity([every], [every_a]))[0][0] > max):
                max = abs(cosine_similarity([every], [every_a]))[0][0]  # 这个词与集合的最大相似度

        if (max > 0.5):
            len = len + 1;
            sim = sim + max;
    if (len > 0):
        acs=sim / len
    return cs,es,ass,acs

def run_train_batch(config, batch, teacher, student,pooler, external_optimizer, device,plm_de,plm_optimizer):#teacher, student,
    nodes, edges, types,node_lists,output_ids,output_ids_masks,target_sets,target_set_mask= batch
    new_nodes=list()  #480*768
    for every_node in node_lists[0]: 
        input_ids = torch.LongTensor([every_node])   
        input_ids=input_ids.to(device)
        output_dict = teacher(input_ids,output_hidden_states=True,return_dict=True)
        e=output_dict[3]#'encoder_last_hidden_state'  [1, 序列长度, 768]  =======f[6]
        pooled_output = pooler(e[0])  
        new_nodes.append(pooled_output)
    teacher_embeddings = torch.stack(new_nodes)   
    teacher_embeddings_three = torch.unsqueeze(teacher_embeddings, dim=0)  #1*480*768
    nodes=torch.as_tensor(nodes) #1*480
    nodes = nodes.to(device)
    student_embeddings = student(nodes, edges, types)    #1*480*768
    kd_loss = compute_kd_loss(student_embeddings,teacher_embeddings_three) #tensor 3.4987
    target_T=[]
    for every_node in target_sets[0]:
        trg_ids = torch.LongTensor([every_node])  # 
        trg_ids = trg_ids.to(device)
        trg_embedding= teacher(trg_ids, output_hidden_states=True, return_dict=True)
        pool_emb =trg_embedding[3]
        pooled_output = pooler(pool_emb[0])
        target_T.append(pooled_output) #10*768  list*tensor
    teacher_embeddings1 = torch.stack(target_T)  # 

    plm_de.to(device)  #这是CNN解码
    final_emb, final_id = plm_de(teacher_embeddings_three)  # 1 10 768  得到了预测结果的10个id     用于损失和评估
    #final_id是该年中的节点对应的nodes的值就是最终得到的结果
    final_emb = torch.squeeze(final_emb, dim=0)  # 可以转为10*768的向量
    teacher_embeddings1 = teacher_embeddings1.cuda().data.cpu().numpy().tolist()
    final_emb = final_emb.cuda().data.cpu().numpy().tolist()
    loss_smila, _num = similarity_loss(final_emb, teacher_embeddings1)  # 相似度
    loss1 = 0  # 数量
    loss_all = kd_loss * config["kd_weight"] #+ loss1 + loss_smila
    plm_optimizer.zero_grad()
    external_optimizer.zero_grad()
    loss_all.backward()
    external_optimizer.step()
    plm_optimizer.step()
    cs,es,ass,acs=similarity_evaluate(final_emb,teacher_embeddings1)
    return loss_all ,cs,es,ass,acs,_num

def run_eval_batch(config, batch, teacher, student, pooler,  device,  plm_de):
    nodes, edges, types, node_lists, output_ids, output_ids_masks, target_sets, target_set_mask = batch
    new_nodes = list()  # 480*768
    for every_node in node_lists[0]:  # 为每一个节点进行嵌入    一共循环节点数量次
        input_ids = torch.LongTensor([every_node])  # 需要变成张量喂给模型
        input_ids = input_ids.to(device)
        output_dict = teacher(input_ids, output_hidden_states=True, return_dict=True)
        e = output_dict[3]  # 'encoder_last_hidden_state'  [1, 序列长度, 768]  =======f[6]
        pooled_output = pooler(e[0])  # 这是长度为嵌入大小（768）的张量
        new_nodes.append(pooled_output)
    teacher_embeddings = torch.stack(new_nodes)  # 480*768的张量     节点数*嵌入维度
    teacher_embeddings_three = torch.unsqueeze(teacher_embeddings, dim=0)  # 1*480*768
    nodes = torch.as_tensor(nodes)  # 1*480
    nodes = nodes.to(device)
    student_embeddings = student(nodes, edges, types)  # 1*480*768
    kd_loss = compute_kd_loss(student_embeddings, teacher_embeddings_three)  # tensor 3.4987
    target_T = []
    for every_node in target_sets[0]:
        trg_ids = torch.LongTensor([every_node])  # 需要变成张量喂给模型
        trg_ids = trg_ids.to(device)
        trg_embedding = teacher(trg_ids, output_hidden_states=True, return_dict=True)
        pool_emb = trg_embedding[3]
        pooled_output = pooler(pool_emb[0])
        target_T.append(pooled_output)  # 10*768  list*tensor
    teacher_embeddings1 = torch.stack(target_T)  # 10*768  list*list*tensor的张量     节点数*嵌入维度
    plm_de.to(device)
    final_emb, final_id = plm_de(teacher_embeddings_three)  # 1 10 768  得到了预测结果的10个id     用于损失和评估
  
    final_emb = torch.squeeze(final_emb, dim=0)  # 可以转为10*768的向量
    teacher_embeddings1 = teacher_embeddings1.cuda().data.cpu().numpy().tolist()
    final_emb = final_emb.cuda().data.cpu().numpy().tolist()
    loss_smila, _num = similarity_loss(final_emb, teacher_embeddings1)  # 相似度
    loss1 = 1 / (1 + _num)  # 数量
    loss_all = kd_loss * config["kd_weight"] #+ loss1 + loss_smila
    cs,es,ass,acs=similarity_evaluate(final_emb,teacher_embeddings1)
    return loss_all ,cs,es,ass,acs,_num

def train(config):
    init_logger(config)
    logger = getLogger()
    init_seed(config["seed"], config["reproducibility"])
    device = init_device(config)
    logger.info("Build node and relation vocabularies.")
    vocabs = dict()
    vocabs["node"] = Vocab(config["node_vocab"]) #37137
    vocabs["relation"] = Vocab(config["relation_vocab"]) #8
    logger.info("Build Teacher Model.")
    teacher = BartForConditionalGeneration.from_pretrained(config["teacher_dir"])
    teacher.requires_grad = False
    for para in teacher.parameters():
        para.requires_grad = False
    teacher.to(device)
    logger.info("Pooling node embedding(Teacher).")
    pooler = MyPooler(config["hidden_size"]) #768
    pooler.requires_grad = False
    for para in pooler.parameters():
        para.requires_grad = False
    pooler.to(device)
    logger.info("Build Student Model.")
    student = GraphEncoder(vocabs["node"].size(), vocabs["relation"].size(),config["gnn_layers"], config["embedding_size"], config["node_embedding"])
    student.to(device)# 37133 8 2 768 37133*768
    external_parameters = []
    for p in student.parameters():
        if p.requires_grad:
            external_parameters.append(p)
   
    plm_de = CNN_decoder(10)
    external_parameters1=[]
    for pde in plm_de.parameters():
        if pde.requires_grad:
            external_parameters1.append(pde)
    plm_optimizer = build_optimizer(external_parameters1, config["plm_learner"], config["plm_lr"], config)  # 解码层优化器
    external_optimizer = build_optimizer(external_parameters, config["external_learner"], config["external_lr"], config)
    scheduler1 = torch.optim.lr_scheduler.StepLR(external_optimizer, step_size=50, gamma=0.1) #学习率衰减
    scheduler2 = torch.optim.lr_scheduler.StepLR(plm_optimizer, step_size=50, gamma=0.1)  # 学习率衰减
    best_idx=0
    best_gen_loss = None
    train_dataloader = DataLoader(
        S2SDataset(data_dir=config["data_dir"], dataset=config["dataset"],
                   tokenizer=bart_tokenizer, node_vocab=vocabs["node"], relation_vocab=vocabs["relation"],
                   num_samples=config["num_samples"], usage="train"),

        batch_size=config["train_batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn_graph_text,
        pin_memory=True)
    logger.info('Load train data done.')
    logger.info("Create validation dataset.")
    valid_dataloader = DataLoader(
        S2SDataset(data_dir=config["data_dir"], dataset=config["dataset"],
                   tokenizer=bart_tokenizer, node_vocab=vocabs["node"], relation_vocab=vocabs["relation"],
                   num_samples="1", usage="valid"),
        batch_size=config["eval_batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn_graph_text,
        pin_memory=True)
    logger.info('Load valid data done.')
    for epoch_idx in range(config["start_epoch"], config["epochs"]):#epoch的迭代次数
        torch.cuda.empty_cache()
        teacher.train()
        student.train()
        pooler.train()
        #plm.train()
        plm_de.train()
        t0 = time.time() #秒数为单位进行计数
        css = []
        ass = []
        ess = []
        acss = []
        ps = 0
        batch_loss=[];
        for batch_idx, batch in enumerate(train_dataloader):
            loss_all ,cs,es,as_,acs,_num  = run_train_batch(
                config, batch, teacher, student, pooler, external_optimizer, device, plm_de, plm_optimizer)
            batch_loss.append(loss_all.item())
            css.append(cs)
            ass.append(as_)
            ess.append(es)
            acss.append(acs)
            ps=ps+_num
#        1轮1次评估结果，将所有的都平均一下
        training_time = time.time() - t0  #format_time(time.time() - t0)
       
      
        batch_loss1 = [];
        with torch.no_grad():
            teacher.eval()
            student.eval()
            plm_de.eval()
            t0 = time.time()
            for batch_idx1, batch1 in enumerate(valid_dataloader):
                loss_all1 ,cs1,es1,as_1,acs1,_num1 = run_eval_batch(config, batch1, teacher, student, pooler,  device,  plm_de)
                batch_loss1.append(loss_all1.item())
                css1.append(cs1)
                ass1.append(as_1)
                ess1.append(es1)
                acss1.append(acs1)
                ps1=ps1+_num1
            valid_time = time.time() - t0
           
        if best_gen_loss is None or np.mean(batch_loss) <= best_gen_loss:
            output_dir = '{}-{}-{}-2023'.format(config["dataset"], config["num_samples"], str(epoch_idx))
            saved_path = os.path.join("./ckpt-2023", output_dir)
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)
            torch.save(config, os.path.join(saved_path, '2023-training_configurations.bin'))
            torch.save({"student": student.state_dict(),"plm_de": plm_de.state_dict()},os.path.join(saved_path, '2023-external.bin'))
            logger.info("Save AHM model into {}.".format(saved_path))
            best_gen_loss = np.mean(batch_loss)
            best_idx=epoch_idx + 1
        scheduler1.step()
        scheduler2.step()

def mytest(config):
    init_logger(config)
    logger = getLogger()
    init_seed(config["seed"], config["reproducibility"])
    device = init_device(config)
    logger.info("Build node and relation vocabularies.")
    vocabs = dict()
    vocabs["node"] = Vocab(config["node_vocab"])
    vocabs["relation"] = Vocab(config["relation_vocab"])
    logger.info("Build Student Model.")
    student = GraphEncoder(vocabs["node"].size(), vocabs["relation"].size(),
                           config["gnn_layers"], config["embedding_size"], config["node_embedding"])
    student.load_state_dict(torch.load(config["external_model"])["student"])
    student.to(device)
    bart_tokenizer = BartTokenizer.from_pretrained(config["plm_dir"])
    logger.info("Create testing dataset.")
    test_dataloader = DataLoader(
        S2SDataset(data_dir=config["data_dir"], dataset=config["dataset"],
                   tokenizer=bart_tokenizer, node_vocab=vocabs["node"], relation_vocab=vocabs["relation"],
                   num_samples=0, usage="test"),
        batch_size=config["test_batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn_graph_text,
        pin_memory=True)
    logger.info('Load test data done.')
    student.eval()
    # 建立解码层模型
    logger.info("Build Decoder Model.")
    plm_de = CNN_decoder(7726)
    plm_de.load_state_dict(torch.load(config["external_model"])["plm_de"])
    plm_de.to(device)
    plm_de.eval()

    idx = 0
    generated_text = []
    reference_text = []
    teacher = BartForConditionalGeneration.from_pretrained(config["teacher_dir"])
    teacher.requires_grad = False
    for para in teacher.parameters():
        para.requires_grad = False
    teacher.to(device)
    logger.info("Pooling node embedding(Teacher).")
    pooler = MyPooler(config["hidden_size"])  # 768
    pooler.requires_grad = False
    for para in pooler.parameters():
        para.requires_grad = False
    pooler.to(device)
    with torch.no_grad():
        for batch in test_dataloader:
            nodes, edges, types, node_lists, output_ids, output_ids_masks, target_sets, target_set_mask = batch
            suoyin = nodes[0]
            nodes = torch.as_tensor(nodes)
            nodes = nodes.to(device)
            student_embeddings = student(nodes, edges, types)  # torch.Size([1, 14362, 768])
            new_nodes = list()

            for every_node in node_lists[0]:  # 为每一个节点进行嵌入    一共循环节点数量次
                input_ids = torch.LongTensor([every_node])  # 需要变成张量喂给模型
                input_ids = input_ids.to(device)
                output_dict = teacher(input_ids, output_hidden_states=True, return_dict=True)
                e = output_dict[3]  # 'encoder_last_hidden_state'  [1, 序列长度, 768]  =======f[6]
                pooled_output = pooler(e[0])  # 这是长度为嵌入大小（768）的张量
                new_nodes.append(pooled_output)
            teacher_embeddings = torch.stack(new_nodes)  # torch.Size([14362, 768])
            teacher_embeddings_three = torch.unsqueeze(teacher_embeddings, dim=0)  # torch.Size([1, 14362, 768])

            # kd_loss = compute_kd_loss(student_embeddings, teacher_embeddings_three)  # tensor 3.4987
            target_T = []
            for every_node in target_sets[0]:
                trg_ids = torch.LongTensor([every_node])  # 需要变成张量喂给模型
                trg_ids = trg_ids.to(device)
                trg_embedding = teacher(trg_ids, output_hidden_states=True, return_dict=True)
                pool_emb = trg_embedding[3]
                pooled_output = pooler(pool_emb[0])
                target_T.append(pooled_output)  # 10*768  list*tensor
            teacher_embeddings1 = torch.stack(target_T)  # 10*768  list*list*tensor的张量     节点数*嵌入维度   teacher对标签的表示嵌入
            plm_de.to(device)  # 这是CNN解码
            final_emb, final_id = plm_de(teacher_embeddings_three)  # 1 10 768  得到了预测结果的10个id     用于损失和评估
            # final_id是该年中的节点对应的nodes的值就是最终得到的结果
            final_emb = torch.squeeze(final_emb, dim=0)  # 可以转为10*768的向量
            teacher_embeddings1 = teacher_embeddings1.cuda().data.cpu().numpy().tolist()
            final_emb = final_emb.cuda().data.cpu().numpy().tolist()
            loss_smila, _num = similarity_loss(final_emb, teacher_embeddings1)  # 相似度
            loss1 = 1 / (1 + _num)  # 数量
            loss_all = loss1 + loss_smila
            cs, es, ass, acs = similarity_evaluate(final_emb, teacher_embeddings1)
            generated_text = []
            reference_text = []

            for generated_id in final_id:
                lalalala = suoyin[generated_id.item()]
                generated = vocabs["node"].convert_ids_to_tokens(lalalala)
                generated_text.append(generated)
            for label_id in output_ids[0]:
                reference = vocabs["node"].convert_ids_to_tokens(label_id)
                reference_text.append(reference)

            idx += 1
            logger.info("2023-Finish {}-th example.".format(idx))

    assert len(generated_text) == len(reference_text)
    saved_file = "2023-{}-{}.res".format(config["dataset"], config["num_samples"])
    saved_file_path = os.path.join(config["output_dir"], saved_file)
    fout = open(saved_file_path, "w")
    fout.write("LOSS: " + str(loss_all) + "\n")
    fout.write("评价指标: cs:" + str(cs) + ",es:" + str(es) + ",as:" + str(ass) + ",acs:" + str(acs) + ",P:" + str(
        _num / 10) + "\n")
    fout.write("Generated text: " + ";".join(generated_text) + "\n")
    fout.write("Reference text: " + ";".join(reference_text) + "\n")
    fout.close()

def main():
    config = read_configuration("config_2023.yaml")
    if config["mode"] == "train":

        train(config)
    else:
        mytest(config)

if __name__ == '__main__':
    main()
