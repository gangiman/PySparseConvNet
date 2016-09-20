import numpy as np
from functools import partial


def normilize_matrix_of_vectors(_all_feature_vectors):
    sam, veclen = _all_feature_vectors.shape
    return _all_feature_vectors / (
        np.sqrt(
            (_all_feature_vectors**2).sum(axis=1)
        ).reshape(sam, 1).dot(
            np.ones((1, veclen)))
    )


def eval_query(query_class, retrieved_class, C):
    G = (query_class == retrieved_class)
    G_sum = G.cumsum(axis=0).astype(float)
    R_points = np.zeros((C,), dtype=int)
    for rec in range(1, C + 1):
        R_points[rec - 1] = (G_sum == rec).nonzero()[0][0]
    P_points = G_sum[R_points] / (R_points + 1).astype(float)
    Av_Precision = P_points.mean()
    NN = int(G[0])
    FT = G_sum[C - 1] / C
    ST = G_sum[2 * C - 1] / C
    P_32 = G_sum[32 - 1] / 32
    R_32 = G_sum[32 - 1] / C

    if P_32 == 0 and R_32 == 0:
        E = 0
    else:
        E = 2 * P_32 * R_32 / (P_32 + R_32)

    NORM_VALUE = 1 + sum(1.0 / np.log2(np.arange(2, C + 1)))
    dcg_i = (1.0 / np.log2(np.arange(2, len(retrieved_class) + 1))) * G[1:]
    dcg_i = np.hstack((float(G[0]), dcg_i[:]))
    dcg = dcg_i.sum() / NORM_VALUE
    return P_points, NN, FT, ST, dcg, E, Av_Precision


class RetrievalDataset(object):
    # lists or sets of unique strings

    num_of_relevant_samples = None

    def query_to_file(self, query, **kwargs):
        raise NotImplementedError()

    def get_search_set_for_query_sample(self, query_sample):
        raise NotImplementedError()

    def search_to_file(self, search):
        raise NotImplementedError()

    def search_to_label(self, search):
        raise NotImplementedError()

    def query_to_label(self, query):
        raise NotImplementedError()

    def evaluate_ranking_for_query(self, _query, metric, **kwargs):
        # for _query in self.query_set:
        potential = partial(metric, self.query_to_file(_query, **kwargs))
        search_set = self.get_search_set_for_query_sample(_query)
        C = self.num_of_relevant_samples

        resulting_ranking = np.array(
            map(potential, map(self.search_to_file, search_set))).argsort()
        return eval_query(
            self.query_to_label(_query),
            np.array(map(
                self.search_to_label,
                map(search_set.__getitem__, resulting_ranking))), C)

    def evaluate_ranking_for_query_set(self, query_set, metric, **kwargs):
        number_of_queries = len(query_set)
        P_points=np.zeros((number_of_queries, self.num_of_relevant_samples))
        Av_Precision = np.zeros((number_of_queries,))
        NN = np.zeros((number_of_queries,))
        FT = np.zeros((number_of_queries,))
        ST = np.zeros((number_of_queries,))
        dcg = np.zeros((number_of_queries,))
        E = np.zeros((number_of_queries,))
        C = self.num_of_relevant_samples

        for qqq, _query in enumerate(query_set):
            P_points[qqq, :], NN[qqq], FT[qqq], ST[qqq], dcg[qqq], E[qqq],\
            Av_Precision[qqq] = self.evaluate_ranking_for_query(
                _query, metric, **kwargs)

        NN_av = NN.mean()
        FT_av = FT.mean()
        ST_av = ST.mean()
        dcg_av = dcg.mean()
        E_av = E.mean()
        Mean_Av_Precision = Av_Precision.mean()

        Precision=P_points.mean(axis=0)
        Recall=np.arange(1, C + 1, dtype=float) / C

        return Precision, Recall, NN_av, FT_av, ST_av, dcg_av,\
               E_av, Mean_Av_Precision
