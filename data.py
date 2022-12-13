import os
import numpy as np
from wordhash import get_ori_data, make_wordhash
from tqdm import tqdm

def add_sign(content, N):
    result = []
    js_window = [i for i in range(N)]
    for c in content:
        temp = []
        temp1 = []
        for w in c.split(" "):
            temp.append("#"+w+"#")
        temp = "".join(temp)
        for i in range(0, len(temp)-N+1):
            temp_content = []
            for w in js_window:
                temp_content.append(temp[i+w])
            temp1.append("".join(temp_content))
        result.append(temp1)
    return result

def pad_content(corpus, v2i, max_text_length=100):
    for i in tqdm(range(len(corpus))):
        if len(corpus[i]) < max_text_length:
            corpus[i] = corpus[i] + ["PAD"] * (max_text_length - len(corpus[i]))
        elif len(corpus[i]) >= max_text_length:
            corpus[i] = corpus[i][:max_text_length]
        for j in range(len(corpus[i])):
            corpus[i][j] = v2i.get(corpus[i][j], 0)
    return corpus

def process(data, v2i, N, max_text_length):
    string1 = []
    string2 = []
    label = []
    for i in range(1, len(data)):
        temp = data[i].strip("\n").replace(",", "").replace(".", "").split("\t")
        label.append(int(temp[0]))
        string1.append(temp[3])
        string2.append(temp[4])
    string1, string2 = add_sign(string1, N), add_sign(string2, N)
    string1, string2 = pad_content(string1, v2i, max_text_length), pad_content(string2, v2i, max_text_length)
    string1, string2 = np.array(string1), np.array(string2)
    return string1, string2, label

def load_data(train_path, val_path, N, max_text_length=100, method="tf"):
    # 1、建立wordhash.txt
    if os.path.exists("wordhash.txt") == False:
        make_wordhash(train_path, N)
    # 2、获得i2v和v2i
    words = []
    with open("wordhash.txt", "r", encoding="utf-8") as fp:
        temp = fp.readlines()
        for i in range(len(temp)):
            words.append(temp[i].strip("\n"))
    i2v = {i+2: v for i, v in enumerate(words)}
    i2v[0] = "PAD"
    i2v[1] = "UNK"
    v2i = {v: i for i, v in i2v.items()}
    # 3、给原始string数据加上#标识
    train_data = get_ori_data(train_path)
    val_data = get_ori_data(val_path)
    string1_train, string2_train, label_train = process(train_data, v2i, N, max_text_length)
    string1_val, string2_val, label_val = process(val_data, v2i, N, max_text_length)
    if method == "tf":
        import tensorflow as tf
        db_train = tf.data.Dataset.from_tensor_slices((string1_train, string2_train, label_train))
        db_train = db_train.shuffle(4000).batch(32, drop_remainder=True)
        db_val = tf.data.Dataset.from_tensor_slices((string1_val, string2_val, label_val))
        db_val = db_val.shuffle(4000).batch(32, drop_remainder=True)
        return db_train, db_val, len(v2i)


if __name__ == "__main__":
    load_data("msr_paraphrase_train.txt", "msr_paraphrase_test.txt", 3, 100)