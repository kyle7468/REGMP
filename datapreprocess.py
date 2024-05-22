from ordered_set import OrderedSet

def data_preprocess(dataset):
    ent_set, rel_set = OrderedSet(), OrderedSet()
    new_ent_set = OrderedSet()
    for split in ['train', 'test', 'valid']:
        for line in open('./data/{}/{}.txt'.format(dataset, split)):
            sub, rel, obj = map(str.lower, line.strip().split('\t'))
            ent_set.add(sub)
            rel_set.add(rel)
            ent_set.add(obj)
    for split in ['train']:
        for line in open('./data/{}/{}.txt'.format(dataset, split)):
            sub, rel, obj = map(str.lower, line.strip().split('\t'))
            lis = sub+rel+obj
            new_ent_set.add(lis)

    # print(len(liss))

    print(len(new_ent_set))

    # self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
    # self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
    # self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(rel_set)})
    #
    # self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
    # self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}
    #
    # self.p.num_ent		= len(self.ent2id)
    # self.p.num_rel		= len(self.rel2id) // 2
    # self.p.embed_dim	= self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim
    #
    # self.data = ddict(list)
    # sr2o = ddict(set)

def main():
    data_preprocess('WN18RR')


if __name__ == '__main__':
    main()