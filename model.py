import tensorflow as tf
from tensorflow import keras
from utils import set_soft_gpu
from data import load_data

class DSSM(keras.Model):

    def __init__(self, vocab_num, out_dim, dropout_rate, lr, decay):
        super(DSSM, self).__init__()

        # embedding
        self.embedding1 = keras.layers.Embedding(input_dim=vocab_num, output_dim=out_dim)
        self.embedding2 = keras.layers.Embedding(input_dim=vocab_num, output_dim=out_dim)
        # dense
        self.dense1 = keras.layers.Dense(256, activation="tanh")
        self.dense2 = keras.layers.Dense(128, activation="tanh")
        self.dense3 = keras.layers.Dense(64, activation="tanh")
        # dropout
        self.dropout = keras.layers.Dropout(dropout_rate)

        # cosine
        self.cosine = keras.losses.CosineSimilarity(axis=1, reduction="none")
        # loss_func
        self.loss_func = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # opt
        self.opt = keras.optimizers.Adam(learning_rate=lr, decay=decay)

    def call(self, inputs1, inputs2, training=None, mask=None):
        # embedding [None 100] -> [None 100 300]
        emb1 = self.embedding1(inputs1)
        emb2 = self.embedding2(inputs2)
        # emb_sum [None 100 300] -> [None 300]
        emb1 = tf.reduce_sum(emb1, axis=1)
        emb2 = tf.reduce_sum(emb2, axis=1)
        # dense [None 300] -> [None 256] -> [None 128] -> [None 64]
        out11 = self.dense1(emb1)
        out11 = self.dropout(out11)
        out12 = self.dense2(out11)
        out12 = self.dropout(out12)
        out13 = self.dense3(out12)
        out13 = self.dropout(out13)

        out21 = self.dense1(emb2)
        out21 = self.dropout(out21)
        out22 = self.dense2(out21)
        out22 = self.dropout(out22)
        out23 = self.dense3(out22)
        out23 = self.dropout(out23)
        # cosine
        pos = self.cosine(out13, out23)
        neg = 1 - pos
        out = tf.stack([pos, neg], axis=1)

        return out

def train():
    epoch = 50
    set_soft_gpu(True)
    db_train, db_val, vocab_num = load_data(train_path="msr_paraphrase_train.txt", val_path="msr_paraphrase_test.txt", N=3
                                            , max_text_length=100, method="tf")
    print(next(iter(db_train))[0].shape, next(iter(db_train))[1].shape, next(iter(db_train))[2].shape,
          next(iter(db_val))[0].shape, next(iter(db_val))[1].shape, next(iter(db_val))[2].shape)
    model = DSSM(vocab_num=vocab_num, out_dim=300, dropout_rate=0.3, lr=0.00001, decay=0.00001)

    for e in range(epoch):
        for step, (x1, x2, y) in enumerate(db_train):
            with tf.GradientTape() as tape:
                out = model.call(x1, x2)
                loss = model.loss_func(y, out)
                grads = tape.gradient(loss, model.trainable_variables)
            model.opt.apply_gradients(zip(grads, model.trainable_variables))
            if step % 50 == 0:
                print("epoch:%d | step:%d | loss:%.3f"%(e, step, loss))

        total_num = 0
        total_acc = 0
        for step, (x1, x2, y) in enumerate(db_val):
            out = model.call(x1, x2)
            pred = tf.nn.softmax(out, axis=1)
            pred = tf.cast(tf.argmax(pred, axis=1), dtype=tf.int32)
            pred = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            acc = tf.reduce_sum(pred)
            total_acc += acc
            total_num += x1.shape[0]
        print("epoch:%d | acc:%.3f"%(e, total_acc / total_num))

if __name__ == "__main__":
    train()