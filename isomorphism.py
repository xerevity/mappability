import scipy
import random
import argparse
import numpy as np
from sklearn.preprocessing import normalize
from gensim.models.keyedvectors import KeyedVectors


class Isomorphism:
    def __init__(self, emb_file1, emb_file2, dict_file):
        self.emb1 = KeyedVectors.load_word2vec_format(emb_file1, limit=200000)
        self.emb2 = KeyedVectors.load_word2vec_format(emb_file2, limit=200000)
        self.words1 = list(self.emb1.vocab)
        self.words2 = list(self.emb2.vocab)
        self.dico = self.read_dict(dict_file)

    def read_dict(self, dict_file):
        pairs_dict = {}
        f = open(dict_file)
        for line in f:
            lang1, lang2 = tuple(line.split())
            if lang1 in self.words1 and lang2 in self.words2:
                if lang1 in pairs_dict:
                    pairs_dict[lang1].append(lang2)
                else:
                    pairs_dict[lang1] = [lang2]
        f.close()
        return pairs_dict

    def relational_similarity(self):
        # choose 10000 random word pairs
        word_pairs = [random.sample(self.dico.items(), k=2) for _ in range(10000)]

        word_pairs1 = [(k[0], v[0]) for k, v in word_pairs]
        word_pairs2 = [(k[1][0], v[1][0]) for k, v in word_pairs]

        # compute similarities
        similarities1 = [self.emb1.similarity(k, v) for k, v in word_pairs1]
        similarities2 = [self.emb2.similarity(k, v) for k, v in word_pairs2]

        # pearson, spearman correlation
        pear = scipy.stats.pearsonr(similarities1, similarities2)
        spear = scipy.stats.spearmanr(similarities1, similarities2)
        print("Pearson correlation coefficient between similarity lists:", np.around(pear[0], decimals=4))
        print("Spearman rank correlation between similarity lists:", np.around(spear[0], decimals=4))
        return pear, spear

    def neighbor_similarity(self):
        lang1_dict_words = set()
        lang2_dict_words = set()
        for w, l in self.dico.items():
            for trans in l:
                lang1_dict_words.add(w)
                lang2_dict_words.add(trans)

        lang1_dict_words = list(lang1_dict_words)
        mat_lang1 = np.stack([self.emb1[w] for w in lang1_dict_words])
        sim_mat_lang1 = mat_lang1 @ mat_lang1.T  # a matrix with the normalized vectors of the dict words, squared

        lang2_dict_words = list(lang2_dict_words)
        mat_lang2 = np.stack([self.emb2[w] for w in lang2_dict_words])
        sim_mat_lang2 = mat_lang2 @ mat_lang2.T

        # get the 10 most similar words in the dictionary for 1000 random words
        src_index = random.sample(list(range(len(lang1_dict_words))), 1000)
        intersects = []
        for i in src_index:
            word = lang1_dict_words[i]
            positions = sim_mat_lang1[i].argsort()[-10:]
            neighbors = set([lang1_dict_words[p] for p in positions])

            translation = self.dico[word][0]
            translations = set([t for neighbor in neighbors for t in self.dico[neighbor]])
            positions2 = sim_mat_lang2[lang2_dict_words.index(translation)].argsort()[-10:]
            neighbors2 = set([lang2_dict_words[p] for p in positions2])

            intersects.append(len(translations.intersection(neighbors2)) / 10)

        score = np.average(intersects)
        print("Average neighbor intersection ratio:", np.around(score, decimals=4))
        return score

    def spectral_similarity(self):

        def entropy(numbers):
            numbers = np.array(numbers)
            dist = numbers / sum(numbers)
            parts = [x * np.log(x) for x in dist if x != 0]
            return -sum(parts)

        centered1 = normalize((self.emb1.vectors - np.mean(normalize(self.emb1.vectors), axis=0)).T)
        centered2 = normalize((self.emb2.vectors - np.mean(normalize(self.emb2.vectors), axis=0)).T)

        s1 = scipy.linalg.svd(centered1, compute_uv=False)
        s2 = scipy.linalg.svd(centered2, compute_uv=False)

        s1.sort()
        s2.sort()

        condition_num1 = s1[-1] / s1[0] if s1[0] > 0 else np.inf
        condition_num2 = s2[-1] / s2[0] if s2[0] > 0 else np.inf

        effective_rank1 = int(np.floor(np.exp(entropy(s1))))
        effective_rank2 = int(np.floor(np.exp(entropy(s2))))

        effcond_num1 = s1[-1] / s1[-effective_rank1]
        effcond_num2 = s2[-1] / s2[-effective_rank2]

        cond_hm = 2 * condition_num1 * condition_num2 / (condition_num1 + condition_num2)
        effcond_hm = 2 * effcond_num1 * effcond_num2 / (effcond_num1 + effcond_num2)
        singular_value_gap = sum((np.log(s1) - np.log(s2)) ** 2)

        print("Spectral condition number HM:", np.around(cond_hm, decimals=4))
        print("Effective spectral condition number HM:", np.around(effcond_hm, decimals=4))
        print("Singular value gap:", np.around(singular_value_gap, decimals=4))

        return cond_hm, effcond_hm, singular_value_gap

    def get_nn_graphs(self, word_list1, word_list2):
        # first create a similarity matrix
        mat_lang1 = np.stack([self.emb1[w] for w in word_list1])
        normalized1 = normalize(mat_lang1)
        centered1 = normalize((normalized1 - np.mean(normalized1, axis=0)).T)
        sim_mat_lang1 = centered1.T @ centered1

        mat_lang2 = np.stack([self.emb2[w] for w in word_list2])
        normalized2 = normalize(mat_lang2)
        centered2 = normalize((normalized2 - np.mean(normalized2, axis=0)).T)
        sim_mat_lang2 = centered2.T @ centered2

        # create a nearest neighbor matrix
        nn_graph1 = np.zeros((len(word_list1), len(word_list1)), dtype=int)
        for i, row in enumerate(sim_mat_lang1):
            ind_a, ind_b = np.argsort(row)[-2], np.argsort(row)[-3]
            nn_graph1[i][ind_a] = 1
            nn_graph1[i][ind_b] = 1

        nn_graph2 = np.zeros((len(word_list2), len(word_list2)), dtype=int)
        for i, row in enumerate(sim_mat_lang2):
            ind_a, ind_b = np.argsort(row)[-2], np.argsort(row)[-3]
            nn_graph1[i][ind_a] = 1
            nn_graph1[i][ind_b] = 1

        # make them symmetric
        nn1 = np.bitwise_or(nn_graph1, nn_graph1.T)
        nn2 = np.bitwise_or(nn_graph2, nn_graph2.T)

        return nn1, nn2

    def isospectral_similarity(self):

        def laplacian_similarity(nn_graph1, nn_graph2):
            lapl1 = np.diag(np.sum(nn_graph1, axis=0)) - nn_graph1
            lapl2 = np.diag(np.sum(nn_graph2, axis=0)) - nn_graph2

            e1 = np.linalg.eigvals(lapl1)
            e1.sort()
            e1 = e1[::-1]
            sum1 = sum(e1)

            e2 = np.linalg.eigvals(lapl2)
            e2.sort()
            e2 = e2[::-1]
            sum2 = sum(e2)

            k1, s1 = 0, 0
            while s1 < 0.9 * sum1:
                s1 += e1[k1]
                k1 += 1

            k2, s2 = 0, 0
            while s2 < 0.9 * sum2:
                s2 += e2[k2]
                k2 += 1

            k = min(k1, k2) - 1
            delta = sum((e1[:k] - e2[:k]) ** 2)

            return delta

        results = []
        for _ in range(10):
            words1 = random.sample(list(self.dico.keys()), k=50)
            words2 = [self.dico[w][0] for w in words1]
            nn1, nn2 = self.get_nn_graphs(words1, words2)
            results.append(laplacian_similarity(nn1, nn2))

        print("Laplacian eigenvalue similarity:", np.around(float(np.mean(results)), decimals=4))

        return float(np.mean(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute similarity of two word embeddings with several measures")
    parser.add_argument("src_emb", help="one embedding model, source language in translation")
    parser.add_argument("trg_emb", help="other embedding model, target in translation")
    parser.add_argument("dict", help="dictionary file from source to target language, one translation pair in each row")
    args = parser.parse_args()

    iso = Isomorphism(args.src_emb, args.trg_emb, args.dict)
    iso.relational_similarity()
    iso.neighbor_similarity()
    iso.spectral_similarity()
    iso.isospectral_similarity()

