import argparse
from collections import defaultdict
import numpy as np
from scipy.spatial import distance


class CorpusSimilarity:
    def __init__(self, dict_file, vocab1, vocab2):
        self.dico = self.read_dict(dict_file)
        self.vocab1 = self.read_vocab(vocab1)
        self.vocab2 = self.read_vocab(vocab2)

    @staticmethod
    def read_dict(file):
        dico = {}
        f = open(file)
        for line in f:
            key, value = tuple(line.split())
            if key not in dico:
                dico[key] = value
        f.close()
        return dico

    @staticmethod
    def read_vocab(file):
        dico = defaultdict(lambda: 0)
        f = open(file)
        for line in f:
            key, value = tuple(line.split())
            if key not in dico:
                dico[key] = int(value)
        f.close()
        return dico

    @staticmethod
    def get_dist_vec(word_list, vocab):
        vec = np.array([vocab[word] for word in word_list])
        vec = vec / sum(vec)
        return vec

    @staticmethod
    def bhattacharyya(vec1, vec2):
        return -np.log(sum(np.sqrt(vec1 * vec2)))

    @property
    def similarity_bhattacharyya(self):
        dist1 = self.get_dist_vec(list(self.dico.keys()), self.vocab1)
        dist2 = self.get_dist_vec(list(self.dico.values()), self.vocab2)
        return self.bhattacharyya(dist1, dist2)

    @property
    def similarity_jsd(self):
        dist1 = self.get_dist_vec(list(self.dico.keys()), self.vocab1)
        dist2 = self.get_dist_vec(list(self.dico.values()), self.vocab2)
        return distance.jensenshannon(dist1, dist2)

    @property
    def token_overlap(self):
        words1 = set(self.vocab1.keys())
        words2 = set(self.vocab2.keys())
        return len(words1.intersection(words2)) / len(words1.union(words2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute similarity of two corpora")
    parser.add_argument("dict_file",
                        help="dictionary file from source to target language, one translation pair in each row")
    parser.add_argument("src_vocab", help="file containing word counts of the source language corpus")
    parser.add_argument("trg_vocab", help="file containing word counts of the target language corpus")
    args = parser.parse_args()

    my_corps = CorpusSimilarity(args.dict_file, args.src_vocab, args.trg_vocab)
    sim1 = my_corps.similarity_bhattacharyya
    sim2 = my_corps.similarity_jsd
    tok = my_corps.token_overlap

    print(f"{args.src_vocab},{args.trg_vocab},{sim1},{sim2},{tok}")
