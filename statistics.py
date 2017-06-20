import numpy as np
import matplotlib.pyplot as plt


class Analyzer:

    def __init__(self, class_num):
        self.class_num = class_num

        self.graphs = dict()

    def get_confusion_matrix(self, predicted_classes, true_classes):
        conf = np.zeros([self.class_num, self.class_num])

        assert len(predicted_classes) == len(true_classes)

        for i in range(len(predicted_classes)):
            conf[predicted_classes[i]][true_classes[i]] += 1
        return conf

    def plot_precision_recall(self, conf_matrix, y):
        assert self.class_num == conf_matrix.shape[1]

        n = conf_matrix.shape[0]
        y = np.argmax(y, axis=1)

        rr = [0] * (conf_matrix.shape[0] + 1)
        pp = [0] * (conf_matrix.shape[0] + 1)

        conf_matrix = conf_matrix / np.sum(conf_matrix, axis=1, keepdims=True)
        for i in range(self.class_num):
            column = list(conf_matrix[:, i])

            x_r = []
            y_p = []
            x_r.append(1)
            y_p.append(0)

            for p in sorted(column):
                tp = 0
                tn = 0
                fp = 0
                fn = 0

                for j in range(n):
                    if column[j] >= p and y[j] != i:
                        fp += 1
                    elif column[j] >= p and y[j] == i:
                        tp += 1
                    elif column[j] < p and y[j] != i:
                        tn += 1
                    else:
                        fn += 1

                if tp == 0:
                    pr = 0
                    re = 0
                else:
                    pr = tp / (tp + fp)
                    re = tp / (tp + fn)

                x_r.append(re)
                y_p.append(pr)

            for ind in range(conf_matrix.shape[0] + 1):
                rr[ind] += x_r[ind]
                pp[ind] += y_p[ind]

            plt.plot(x_r, y_p, color='green')
            plt.axis([-0.1, 1.1, -0.1, 1.1])
            plt.show()

        for ind in range(conf_matrix.shape[0] + 1):
            rr[ind] /= self.class_num
            pp[ind] /= self.class_num

        plt.plot(rr, pp, color='green')
        plt.axis([-0.1, 1.1, -0.1, 1.1])
        plt.show()

    def add_point(self, step, value, name):
        if name not in self.graphs:
            self.graphs[name] = []
        self.graphs[name].append((step, value))

    def plot(self):
        for name in self.graphs:
            x = [t[0] for t in self.graphs[name]]
            y = [t[1] for t in self.graphs[name]]
            plt.plot(x, y, label=name)
        plt.show()

if __name__ == '__main__':
    a = np.array([[1, 2, 3], [3, 2, 1]])
    ana = Analyzer(3)
    ana.plot_precision_recall(a, [0,1])
