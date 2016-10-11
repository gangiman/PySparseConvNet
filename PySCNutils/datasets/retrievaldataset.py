import numpy as np
from functools import partial
from tqdm import tqdm


def normilize_matrix_of_vectors(_all_feature_vectors):
    sam, veclen = _all_feature_vectors.shape
    return _all_feature_vectors / (
        np.sqrt(
            (_all_feature_vectors**2).sum(axis=1)
        ).reshape(sam, 1).dot(
            np.ones((1, veclen)))
    )


def eval_query(query_class, retrieved_class, num_of_relevant):
    G = (query_class == retrieved_class)
    G_sum = G.cumsum(axis=0).astype(float)
    R_points = np.zeros((num_of_relevant,), dtype=int)
    for rec in range(1, num_of_relevant + 1):
        R_points[rec - 1] = (G_sum == rec).nonzero()[0][0]
    P_points = G_sum[R_points] / (R_points + 1).astype(float)
    Av_Precision = P_points.mean()
    NN = int(G[0])
    FT = G_sum[num_of_relevant - 1] / num_of_relevant
    ST = G_sum[2 * num_of_relevant - 1] / num_of_relevant
    P_32 = G_sum[32 - 1] / 32
    R_32 = G_sum[32 - 1] / num_of_relevant

    if P_32 == 0 and R_32 == 0:
        E = 0
    else:
        E = 2 * P_32 * R_32 / (P_32 + R_32)

    NORM_VALUE = 1 + sum(1.0 / np.log2(np.arange(2, num_of_relevant + 1)))
    dcg_i = (1.0 / np.log2(np.arange(2, len(retrieved_class) + 1))) * G[1:]
    dcg_i = np.hstack((float(G[0]), dcg_i[:]))
    dcg = dcg_i.sum() / NORM_VALUE
    return P_points, NN, FT, ST, dcg, E, Av_Precision


class RetrievalDataset(object):
    # lists or sets of unique strings

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

    def num_relevant_samples_for_query(self, query):
        raise NotImplementedError()

    def evaluate_ranking_for_query(self, _query, metric, **kwargs):
        """

        :param _query:
        :param metric:
        :param n_relevant_samples: Number of relevant samples for query
        :param kwargs:
        :return:
        """
        # for _query in self.query_set:
        potential = partial(metric, self.query_to_file(_query, **kwargs))
        search_set = self.get_search_set_for_query_sample(_query)
        n_relevant_samples = self.num_relevant_samples_for_query(_query)

        resulting_ranking = np.array(
            map(potential, map(self.search_to_file, search_set))
        ).argsort()
        return eval_query(
            self.query_to_label(_query),
            np.array(map(
                self.search_to_label,
                map(search_set.__getitem__, resulting_ranking))),
            n_relevant_samples)

    def evaluate_ranking_for_query_set(self, query_set, metric, **kwargs):
        number_of_queries = len(query_set)

        # P_points=np.zeros((number_of_queries, self.num_of_relevant_samples))
        p_points = [0] * number_of_queries

        av_precision = np.zeros((number_of_queries,))
        nearest_neighbour = np.zeros((number_of_queries,))
        first_tier = np.zeros((number_of_queries,))
        second_tier = np.zeros((number_of_queries,))
        dcg = np.zeros((number_of_queries,))
        e_measure = np.zeros((number_of_queries,))

        for qqq, _query in enumerate(tqdm(
                query_set, leave=False, unit='sample',
                desc='Iter val. queries')):
            (p_points[qqq], nearest_neighbour[qqq], first_tier[qqq],
             second_tier[qqq], dcg[qqq], e_measure[qqq],
             av_precision[qqq]) = self.evaluate_ranking_for_query(
                _query, metric, **kwargs)

        nearest_neighbour_av = nearest_neighbour.mean()
        first_tier_av = first_tier.mean()
        second_tier_av = second_tier.mean()
        dcg_av = dcg.mean()
        e_measure_av = e_measure.mean()
        mean_av_precision = av_precision.mean()

        def recall_fun(n):
            return np.arange(1, n + 1, dtype=float) / n

        lengths_of_ppoints = map(len, p_points)

        if kwargs.get('average_pr', True):
            max_num_of_relevant = max(lengths_of_ppoints)
            interp_p_points = np.zeros(
                (number_of_queries, max_num_of_relevant))

            for _i, arr in enumerate(p_points):
                if len(arr) < max_num_of_relevant:
                    interp_p_points[_i] = np.interp(
                        recall_fun(max_num_of_relevant),
                        recall_fun(len(arr)), arr)
                else:
                    interp_p_points[_i] = arr
            precision = interp_p_points.mean(axis=0)
            recall = recall_fun(max_num_of_relevant)
        else:
            precision = p_points
            recall = map(recall_fun, lengths_of_ppoints)

        return (precision, recall, nearest_neighbour_av, first_tier_av,
                second_tier_av, dcg_av, e_measure_av, mean_av_precision)
