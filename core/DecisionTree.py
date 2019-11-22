from core.common import *

class TreeNode:
    def __init__(self,is_leaf,feature,threshold,left,right,score):
        self.is_leaf = is_leaf
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.score = score


def depth(tree):
    if tree.is_leaf:
        return 0
    else:
        return 1 + max(depth(tree.left), depth(tree.right))


def predict_single(tree, example):
    if tree.is_leaf:
        return tree.score
    else:
        if example[tree.feature] < tree.threshold:
            return predict_single(tree.left, example)
        else:
            return predict_single(tree.right, example)


def predict_mp(tree, examples, batch_size = 10, worker_num = 5):

    total_num = len(examples)
    ret = []
    for i in range(math.ceil(total_num / batch_size)):
        with Pool(processes=worker_num) as pool:
            ret.append(
                pool.map_async(
                    partial(predict_single, tree=tree),
                    examples[i*batch_size, (i+1)*batch_size]
                )
            )
    return ret



class Tree(object):
    def __init__(self, estimator=None, max_depth=3, min_sample_split=10, gamma=0):
        self.estimator = estimator
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.gamma = gamma

    def predict_single(self, treenode, test):
        if treenode.is_leaf:
            return treenode.score
        else:
            if test[treenode.feature] < treenode.threshold:
                return self.predict_single(treenode.left, test)
            else:
                return self.predict_single(treenode.right, test)

    def predict(self, test):
        result = []
        for i in test:
            result.append(self.predict_single(self.estimator, i))
        return result

    def calculate_score(self, label):
        return np.mean(label)

    def mse(self, label):
        return np.square(label - np.mean(label))

    # need to impl
    def find_best_feature_threshold_and_gain(self, train, label):
        # 找到增益最大的划分
        best_feature = -1
        best_threshold = -1
        best_gain = -1
        n_row, n_feature = train.shape[0], train.shpe[1]
        print(train.shape)
        for feature in n_feature:
            bins = np.array(list(set(train[:,feature])), dtype=float)
            thresholds = (bins[1:] +bins[:-1]) / 2
            for threshold in thresholds:
                split_index = train[:,feature] < threshold
                # gain = self.mse(train[:,feature][])
                gain = math.fabs(
                    self.mse(label=label[split_index]) - self.mse(label=label[~split_index])
                )
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = threshold
                    best_feature = feature

        return best_feature, best_threshold, best_gain

    def construct_tree(self, train, label, depth_left):
        # 如果train数量很小，则停止划分
        if len(train) < self.min_sample_split or depth_left == 0:
            return TreeNode(is_leaf=True, score=self.calculate_score(label))

        best_feature, best_threshold, best_gain = self.find_best_feature_threshold_and_gain(train, label)
        # 如果增益已经足够小
        if best_gain < self.gamma:
            return TreeNode(is_leaf=True, score=self.calculate_score(label))

        index = train[:,best_feature] < best_threshold

        left = self.construct_tree(train=train[index], label=label[index], depth_left=depth_left - 1)
        right = self.construct_tree(train=train[~index], label=label[~index], depth_left=depth_left - 1)
        return TreeNode(feature=best_feature, threshold=best_threshold, left=left, right=right)