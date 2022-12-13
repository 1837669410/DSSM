def set_soft_gpu(soft_gpu):
    import tensorflow as tf
    if soft_gpu:
        # 返回电脑上可用的GPU列表
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

def one_hot(y, depth=10):
    import torch
    y_onehot = torch.zeros(y.shape[0], depth)
    idx = torch.LongTensor(y).view(-1,1)
    y_onehot.scatter_(dim=1, index=idx, value=1)
    return y_onehot