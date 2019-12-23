from abc import ABC, abstractmethod

class A(ABC):

    @abstractmethod
    def fit(self, X, y):
        X = X.append(['caio matto'])
        y = y.append([2,3,4,5,])

class B(A):

    def __init__(self):
        pass

    def fit(self, X, y):
        super().fit(X, y)

        print('X: \n', X)
        print('y: \n', y)


if __name__ == "__main__":
    # m = Model.load('/home/jovyan/work/models/model.hal')
    # pred = m.predict_single('Otherllo', 'The evil Iago pretends to be friend of Othello \
    #     in order to manipulate him to serve his own end in the film version of \
    #     this Shakespeare classic.')

    # print(pred)
    
    # --t 'Otherllo' --d 'The evil Iago pretends to be friend of Othello in order to manipulate him to serve his own end in the film version of this Shakespeare classic.'
    

    l = [12, 15, 25, 45, 218, 648]

    print(l)
    for i, e in reversed(list(enumerate(l))):
        l[i] = e - l[i-1]

    l = l[1:]
    print(l)