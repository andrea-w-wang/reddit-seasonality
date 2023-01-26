import math
from scipy import stats


class JSD:
    def __init__(self, ngrams_counter_1, ngrams_counter_2, weight_1, weight_2, base):
        self.base = base
        self.weight_1 = weight_1
        self.weight_2 = weight_2
        self.p_1 = self.make_prob_distribution(ngrams_counter_1)  # probability distribution 1
        self.p_2 = self.make_prob_distribution(ngrams_counter_2)  # probability distribution 2
        self.p_joint = self.weighted(self.p_1, weight_1) + self.weighted(self.p_2, weight_2)
        self.total_diff = self._get_total_diff()
        self.item_contribution = self._get_item_contribution()

    @staticmethod
    def make_prob_distribution(ngrams_counter):
        x = ngrams_counter.copy()
        total = sum(x.values(), 0.0)
        for key in x:
            x[key] /= total
        return x

    @staticmethod
    def weighted(prob_dist, weight):
        weighted_prob_dist = prob_dist.copy()
        for k in prob_dist:
            weighted_prob_dist[k] = prob_dist[k] * weight
        return weighted_prob_dist

    def kl_div(self, dist_1, dist_2):
        union_items = set(dist_1.keys()).union(dist_2.keys())
        dist_1_k = list()
        dist_2_k = list()
        for item in union_items:
            dist_1_k.append(dist_1[item])
            dist_2_k.append(dist_2[item])
        return stats.entropy(dist_1_k, dist_2_k, base=self.base)

    def _get_total_diff(self):
        return self.weight_1 * self.kl_div(self.p_1, self.p_joint) + self.weight_2 * self.kl_div(self.p_2, self.p_joint)

    def item_relative_entropy(self, item, dist_1, dist_2):
        if dist_1[item] > 0:
            return dist_1[item] * math.log(dist_1[item] / dist_2[item], self.base)
        elif dist_1[item] == 0:
            return 0
        else:
            raise Exception("item probability could not be < 0.")

    def _get_item_contribution(self):
        stash = dict()
        for ngram in self.p_joint:
            contrib_from_dist_1 = self.item_relative_entropy(ngram, self.p_1, self.p_joint)
            contrib_from_dist_2 = self.item_relative_entropy(ngram, self.p_2, self.p_joint)
            contribution = self.weight_1 * contrib_from_dist_1 + self.weight_2 * contrib_from_dist_2
            stash[ngram] = {
                "contribution": contribution / self.total_diff,
                "from_distribution": "1" if
                contrib_from_dist_1 > contrib_from_dist_2 else "2"  # todo: does it needs to times weight?
            }

        return stash
