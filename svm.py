"""
Author: Raj Patel
Class: CS 5360
"""
import numpy as np
DATA_DIR = 'data/'


class SVM:
    np.random.seed(911)

    def parse_data(self, data):
        """
        Parses the data from files.
        :param data: data to parse
        :return: array of parsed data
        """
        data_to_return = []
        for line in data:
            line = line.split()
            array_to_add = np.zeros(220)
            label = int(line[0])
            if label == 0:
                label = -1
            array_to_add[0] = label
            line.remove(line[0])
            for val in line:
                val = val.split(':')
                array_to_add[int(val[0])] = int(val[1])
            data_to_return.append(array_to_add)
        return data_to_return

    def split(self, a, n):
        """
        Splits an array in to n parts.
        :param n: Number of parts to split.
        :return: Arrays of n size array.
        """
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    def train_and_eval(self, weights, train_data, debug=False):
        print(self.bordered("SUB-GRADIENT SVM"))

        test_data = self.parse_data(np.loadtxt(DATA_DIR + "test.liblinear", delimiter=",", dtype=str))
        k_fold_splits = [self.parse_data(np.loadtxt(DATA_DIR + "CVSplits/training00.data", delimiter=",", dtype=str)),
                         self.parse_data(np.loadtxt(DATA_DIR + "CVSplits/training01.data", delimiter=",", dtype=str)),
                         self.parse_data(np.loadtxt(DATA_DIR + "CVSplits/training02.data", delimiter=",", dtype=str)),
                         self.parse_data(np.loadtxt(DATA_DIR + "CVSplits/training03.data", delimiter=",", dtype=str)),
                         self.parse_data(np.loadtxt(DATA_DIR + "CVSplits/training04.data", delimiter=",", dtype=str))]

        hyper_params = {'learning_rates': [10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3, 10 ** -4],
                        'trade_offs': [10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3, 10 ** -4]}

        return self.analyze(train_data, test_data, k_fold_splits, hyper_params, 0, weights=weights, debug=debug)

    def get_train_data(self):
        return self.parse_data(np.loadtxt(DATA_DIR + "train.liblinear", delimiter=",", dtype=str))

    def get_test_data(self):
        return self.parse_data(np.loadtxt(DATA_DIR + "test.liblinear", delimiter=",", dtype=str))

    def analyze(self, train_data, test_data, k_fold_splits, hyper_params, t, weights=None, debug=False):
        best = {'learning_rate': 0, 'trade_off': 0, 'avg_precision': 0, 'recall': 0, 'F_1': 0, 'accuracy': 0}
        for learning_rate in hyper_params['learning_rates']:
            for trade_off in hyper_params['trade_offs']:
                k_fold_results = self.k_fold(k_fold_splits, learning_rate, trade_off, t,
                                             weights=weights)
                if best['F_1'] <= k_fold_results['F_1']:
                    best['F_1'] = k_fold_results['F_1']
                    best['avg_precision'] = k_fold_results['precision']
                    best['recall'] = k_fold_results['recall']
                    best['learning_rate'] = learning_rate
                    best['trade_off'] = trade_off
                    best['accuracy'] = k_fold_results['accuracy']

        if debug:
            print("\n")
            print("Best K-Fold results")
            print("Learning Rate:", best['learning_rate'])
            print("Trade Off:", best['trade_off'])
            print("Average Precision:", best['avg_precision'])
            print("Recall:", best['recall'])
            print("F_1:", best['F_1'])
            print("Accuracy:", best['accuracy'])
            print("\n")

        epoch_results = self.epochs(10, train_data, test_data, best['learning_rate'], best['trade_off'], t, debug=False,
                                    weights=weights)

        if debug:
            print("Best Test Data Results")
            print("Precision:", epoch_results['best_precision'])
            print("Recall:", epoch_results['best_recall'])
            print("F_1:", epoch_results['best_F_1'])
            print("Accuracy", epoch_results['best_accuracy'])

        return epoch_results['avg_w']
        # self.predict_to_csv(best_results["w"], eval_data, csv_name, margin=best_hyper_param["margin"],
        #                     debug=True)

    def k_fold(self, k_fold_splits, learning_rate, trade_off, t, weights=None):
        # print("K-Fold -> [learning rate: " + str(learning_rate) + ", trade off: " + str(trade_off) + "]")
        avg = {'precision': 0, 'recall': 0, 'F_1': 0, 'accuracy': 0}
        for i in range(5):
            k_fold_test_data = k_fold_splits[i]
            k_fold_train_data = []

            for j in range(5):
                if i == j:
                    continue
                k_fold_train_data = k_fold_train_data + k_fold_splits[j]

            epoch_results = self.epochs(10, k_fold_train_data, k_fold_test_data, learning_rate, trade_off, t, weights=weights)
            avg['precision'] += epoch_results['avg_precision']
            avg['recall'] += epoch_results['avg_recall']
            avg['F_1'] += epoch_results['avg_F_1']
            avg['accuracy'] += epoch_results['avg_accuracy']

        avg['precision'] = avg['precision'] / 5
        avg['recall'] = avg['recall'] / 5
        avg['F_1'] = avg['F_1'] / 5
        avg['accuracy'] = avg['accuracy'] / 5

        # print("Average Precision:", avg['precision'])
        # print("Recall:", avg['recall'])
        # print("F_1:", avg['F_1'])
        # print("Accuracy:", avg['accuracy'])
        return avg

    def epochs(self, repeat, train_data, test_data, learning_rate, trade_off, t, debug=False, weights=None):
        if debug:
            print("Running Epochs [%d Times]" % repeat)
        w = weights
        if weights is None:
            w = np.full(219, np.random.uniform(low=-0.01, high=0.01))
        t_temp = t

        results = {'best_precision': 0, 'best_recall': 0, 'best_F_1': 0, 'best_accuracy': 0, 'avg_precision': 0,
                   'avg_recall': 0, 'avg_F_1': 0, "w": w, "avg_accuracy": 0}
        for i in range(repeat):
            if debug:
                print("Epoch ID:", i + 1)
            np.random.shuffle(train_data)
            w, t_temp = self.sub_gradient_svm(train_data, learning_rate, trade_off, t_temp, w, debug=debug)
            current_results = self.predict(w, test_data, debug=False)
            # if debug:
            #     print("Accuracy:", current_accuracy)
            results['avg_precision'] += current_results['precision']
            results['avg_recall'] += current_results['recall']
            results['avg_F_1'] += current_results['F_1']
            results['avg_accuracy'] += current_results['accuracy']
            if current_results['F_1'] >= results['best_F_1']:
                results['best_F_1'] = current_results['F_1']
                results['best_precision'] = current_results['precision']
                results['best_recall'] = current_results['recall']
                results['w'] = w
                results['best_accuracy'] = current_results['accuracy']
        results['avg_precision'] /= repeat
        results['avg_recall'] /= repeat
        results['avg_F_1'] /= repeat
        results['avg_accuracy'] /= repeat
        results['avg_w'] = w
        return results

    @staticmethod
    def bordered(text):
        lines = text.splitlines()
        width = max(len(s) for s in lines)
        res = ['┌' + '─' * width + '┐']
        for s in lines:
            res.append('│' + (s + ' ' * width)[:width] + '│')
        res.append('└' + '─' * width + '┘')
        return '\n'.join(res)

    def sub_gradient_svm(self, data_arr, init_learning_rate, trade_off, t, w, debug=False):
        for x_i in data_arr:
            learning_rate_t = self.calculate_learning_rate(init_learning_rate, t)
            t += 1
            y_i = x_i[0]
            x_i = np.delete(x_i, 0)
            y_prime = self.sign(w, x_i, y_i)
            # print(y_prime)
            if y_prime <= 1:
                w = ((1 - learning_rate_t) * w) + (learning_rate_t * trade_off * y_i * x_i)
            else:
                w = (1 - learning_rate_t) * w
        return w, t

    @staticmethod
    def calculate_learning_rate(init_learning_rate, t):
        return init_learning_rate / (1 + t)

    @staticmethod
    def sign(w, x_i, y_i):
        dot_product = y_i * np.dot(w.transpose(), x_i)
        # return np.sign(dot_product)
        return dot_product

    @staticmethod
    def predict_sign(w, x_i):
        dot_product = np.dot(w.transpose(), x_i)
        if dot_product <= 0:
            return -1
        return 1
        # return dot_product

    def predict(self, w, test_data, debug=False):
        tp = 0
        fp = 0
        fn = 0
        total = 0
        correct = 0
        for x_i in test_data:
            total += 1
            y_i = x_i[0]
            x_i = np.delete(x_i, 0)
            y_prime = self.predict_sign(w, x_i)
            if y_i == y_prime:
                correct += 1
            # print(y_prime, y_i)
            if (y_i == 1) and (y_prime == 1):
                tp += 1
            if (y_i == -1) and (y_prime == 1):
                fp += 1
            if (y_i == 1) and (y_prime == -1):
                fn += 1

        if debug:
            print(tp, fp, fn)

        results = {'precision': 1, 'recall': 1, 'F_1': 1, 'accuracy': correct / total}

        if tp == 0:
            tp = 1
        if fp == 0:
            fp = 1
        if fn == 0:
            fn = 1

        results['precision'] = tp / (tp + fp)
        results['recall'] = tp / (tp + fn)
        results['F_1'] = 2 * ((results['precision'] * results['recall']) / (results['precision'] + results['recall']))

        return results
