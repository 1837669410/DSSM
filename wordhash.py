from tqdm import tqdm

def get_ori_data(path):
    with open(path, "r", encoding="utf-8") as fp:
        return fp.readlines()

def pad_word(string1, string2):
    for i in range(len(string1)):
        temp1 = string1[i].split(" ")
        temp2 = string2[i].split(" ")
        for j in range(len(temp1)):
            temp1[j] = "#" + temp1[j] + "#"
        for j in range(len(temp2)):
            temp2[j] = "#" + temp2[j] + "#"
        string1[i] = "".join(temp1)
        string2[i] = "".join(temp2)
    return string1, string2

def n_gram_wordhash(N, corpus):
    js_window = [i for i in range(N)]
    word_hash = []
    for c in tqdm(corpus):
        for i in range(0, len(c)-N+1):
            temp = []
            for j in js_window:
                temp.extend(c[i+j])
            word_hash.append("".join(temp))
    word_hash = list(set(word_hash))
    with open("wordhash.txt", "w", encoding="utf-8") as fp:
        for whash in word_hash:
            fp.write(whash + "\n")

def make_wordhash(path, N):
    ori_data = get_ori_data(path)
    string1 = []
    string2 = []
    for i in range(1, len(ori_data)):
        temp = ori_data[i].strip("\n").replace(",", "").replace(".", "").split("\t")
        string1.append(temp[3])
        string2.append(temp[4])
    string1, string2 = pad_word(string1, string2)
    n_gram_wordhash(N, string1 + string2)

if __name__ == "__main__":
    make_wordhash("msr_paraphrase_train.txt", 3)