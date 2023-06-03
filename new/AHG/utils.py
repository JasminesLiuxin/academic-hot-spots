import json
import torch
import random
import logging
import os
import datetime
import yaml
import re
import numpy as np
from torch import optim
from optim import CosineSchedule, TransformerSchedule


def build_optimizer(parameters, learner, learning_rate, config):
    if learner.lower() == 'adam':
        optimizer = optim.Adam(parameters, lr=learning_rate)
    elif learner.lower() == 'sgd':
        optimizer = optim.SGD(parameters, lr=learning_rate)
    elif learner.lower() == 'adagrad':
        optimizer = optim.Adagrad(parameters, lr=learning_rate)
    elif learner.lower() == 'rmsprop':
        optimizer = optim.RMSprop(parameters, lr=learning_rate)
    elif learner.lower() == 'adamw':
        optimizer = optim.AdamW(parameters, lr=learning_rate)
    elif learner.lower() == 'cosine_warmup':
        optimizer = CosineSchedule(
            optim.AdamW(parameters, betas=(0.9, 0.98), eps=1e-08, weight_decay=0.01),
            learning_rate, config["warmup_steps"], config["training_steps"]
        )
    elif learner.lower() == 'transformer_warmup':
        optimizer = TransformerSchedule(
            optim.AdamW(parameters, betas=(0.9, 0.98), eps=1e-08, weight_decay=0.01),
            learning_rate, config["embedding_size"], config["warmup_steps"]
        )
    else:
        raise ValueError('Received unrecognized optimizer {}.'.format(learner))
    return optimizer


def init_seed(seed, reproducibility):
    '''
    设置随机种子的作用：随机数种子确定时，模型的训练结果将保持一致
    在数据预处理阶段使用了随机初始化的nn.Embedding，并将其通过持久化方式pickle、cpickle保存了下来。再次使用时，通过pickle.load()读取，即使固定了随机数种子，
    此时读取到的nn.Embedding()中的weight与当初保存下来的weight是不同的

    '''

    torch.manual_seed(seed)  #为cpu设置种子
    torch.cuda.manual_seed(seed)  #为gpu设置种子
    torch.cuda.manual_seed_all(seed)
    if reproducibility:  #这个对于效率的影响其实是小数点后几位  如果对于精度的要求不高   不需要修改
        torch.backends.cudnn.benchmark = False
        #设置为True可以对模型里的卷积层及逆行预先优化，即在每一个卷积层中测试 cuDNN 提供的所有卷积实现算法，然后选择最快的那个。　但是这样会增加时间，减少运行效率
        torch.backends.cudnn.deterministic = True
    else:  #更快   重复性更低
        torch.backends.cudnn.benchmark = True

        #固定ｃｕｄａ的种子　　　将这个 flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法。如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的
        torch.backends.cudnn.deterministic = False


def init_device(config):
    use_gpu = config["use_gpu"]
    device = torch.device("cuda:" + str(config["gpu_id"]) if torch.cuda.is_available() and use_gpu else "cpu")
    return device


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur


def init_logger(config):
    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])

    logfilename = '{}-{}-{}.log'.format(config["dataset"], config["num_samples"], get_local_time())
    logfilepath = os.path.join(config["log_dir"], logfilename)

    filefmt = "%(asctime)-15s %(levelname)s %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(asctime)-15s %(levelname)s %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = logging.Formatter(sfmt, sdatefmt)
    if config["state"] is None or config["state"].lower() == 'info':
        level = logging.INFO
    elif config["state"].lower() == 'debug':
        level = logging.DEBUG
    elif config["state"].lower() == 'error':
        level = logging.ERROR
    elif config["state"].lower() == 'warning':
        level = logging.WARNING
    elif config["state"].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO
    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(
        level=level,
        handlers=[fh, sh]
    )


def read_configuration(config_file):
    yaml_loader = yaml.FullLoader
    yaml_loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
                 [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                |[-+]?\\.(?:inf|Inf|INF)
                |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open(config_file, 'r') as f:
        config_dict = yaml.load(f.read(), Loader=yaml_loader)

    return config_dict


def collate_fn_graph_text(batch):

    output_ids_masks,nodes, edges, types,node_lists,  output_ids,target_sets,target_set_mask=[],[], [],[],[], [],[],[]
    for b in batch:
        nodes.append(b[0])
        edges.append(b[1])
        types.append(b[2])
        node_lists.append(b[3])
        output_ids.append(b[4])
        output_ids_masks.append(b[4])
        target_sets.append(b[5])
        target_set_mask.append(b[5])
    return nodes, edges, types,node_lists,output_ids,output_ids_masks,target_sets,target_set_mask

