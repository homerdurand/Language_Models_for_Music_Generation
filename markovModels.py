import numpy as np
import scipy.stats as scp
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import music21 as m21
from tqdm import tqdm


def transitionMatrix(sequence, n_gram=1) :
    """Compute the transition matrix of a sequence regarding on the following state after a sequence of length ngram

    Parameters
    ----------
    sequence : array-like of states
        Sequence of state observed
    n : int
        Length of the sequence that precede the predicted next state

    Returns
    -------
    """

    transitionDict = {}
    for i in tqdm(range(len(sequence)-n_gram)):
        currentNgram = []
        for j in range(n_gram) :
            currentNgram.append(sequence[i+j])
        currentNgram = tuple(currentNgram)
        if currentNgram in transitionDict :
            if sequence[i+n_gram] in transitionDict[currentNgram] :
                transitionDict[currentNgram][sequence[i+n_gram]]+=1
            else :
                transitionDict[currentNgram][sequence[i+n_gram]]=1
        else :
            transitionDict[currentNgram] = {sequence[i+n_gram]:1}

    transitionDf = pd.DataFrame(transitionDict).T

    return(transitionDf.div(transitionDf.sum(axis=1), axis=0).fillna(1))

def sequencify(transition_matrix, startingSeq, seqLength=1000, n=1) :
    if len(startingSeq) < n :
        print("The starting sequence should be longer than the length of the ngram...")
    sequence = startingSeq
    state =""
    for i in range(seqLength):
        currentNgram = tuple(sequence[-n:lennltk.lm.preprocessing.padded_everygram_pipeline(sequence)])
        prob=transition_matrix.loc[currentNgram]
        val=list(transition_matrix.columns)
        state = np.random.choice(val, p = prob)
        sequence.append(state)
    return(sequence)



class MMarkov :
    def __init__(self, order = 1) :
        self.order = order

    def fit(self, X) :
        self.sequence = X
        self.vocab_size = len(set(X))
        self.vocab_frequency = dict(Counter(X))
        self.transition_matrix = transitionMatrix(X, self.order)
        self.n_ngrams = len(self.transition_matrix)
        return self

    def sequencify(self, startingSeq, seqLength) :
        return sequencify(self.transition_matrix, startingSeq, seqLength, self.order)

    def interest(self, log=True) :
        overfitted_val = 0 #Une valeur est overfit si elle n'a qu'un etat suivant possible
        supZtot=0
        for ind in self.transition_matrix :
            supZ = 0
            for val in self.transition_matrix[ind] :
                supZ+=val>0
            if log :
                supZtot+=np.log(supZ)
            else :
                supZtot+=supZ
        self.interest = supZtot/len(self.transition_matrix.index)
        return self.interest

    def coherence(self, testSeq) :
        coherence = 0
        new = 0
        for i in range(len(testSeq)-self.order):
            currentNgram = tuple([testSeq[i+j] for j in range(self.order)])
            try :
                coherence+=self.transition_matrix.T[currentNgram][testSeq[i+self.order]]
            except :
                new+=1
        return coherence/(len(testSeq)-self.order-new), new

    def topKAccurracy(self, testSeq, k=5) :
            acc = 0
            unseen = 0
            for i in range(len(testSeq)-self.order):
                currentNgram = tuple([testSeq[i+j] for j in range(self.order)])
                if currentNgram in self.transition_matrix.T :
                    d = dict(self.transition_matrix.T[currentNgram])
                    c = Counter(d)
                    topk = c.most_common(k)
                    if testSeq[i+self.order] in list(dict(topk).keys()) and dict(topk)[testSeq[i+self.order]]!=0:
                        acc+=1
                else :
                    unseen+=1
            return acc/(len(testSeq)-self.order), unseen/(len(testSeq)-self.order)

    def perplexity(self, testSeq, add_alpha=True) :
        perplexity = []
        for i in range(len(testSeq)-self.order):
            currentNgram = tuple([testSeq[i+j] for j in range(self.order)])
            try :
                perplexity.append(np.log(self.transition_matrix.T[currentNgram][testSeq[i+self.order]]))
            except :
                perplexity.append(add_alpha/len(self.transition_matrix.T))
        return(2**np.mean(perplexity))

    def vocabRepartition(self) :
        #consider only the frequencies
        myDict = self.vocab_frequency
        plt.bar(myDict.keys(), myDict.values(), color='g')

    def langageCoverage(self, plot = True, topN = False) :
        dictOccurenceNgram={}
        for i in range(len(self.sequence)-self.order+1):
                currentNgram = []
                for j in range(self.order) :
                    currentNgram.append(self.sequence[i+j])
                currentNgram = tuple(currentNgram)
                if currentNgram in dictOccurenceNgram :
                    dictOccurenceNgram[currentNgram]+=1
                else :
                    dictOccurenceNgram[currentNgram] = 1
        if plot :
            plt.plot(np.cumsum(sorted(dictOccurenceNgram.values(), reverse=True))/len(self.sequence), label = str(self.order)+"Gram")
            plt.ylabel('Langage Coverage')
            plt.xlabel('Numer of N_Gram')
        if topN :
            k = Counter(dictOccurenceNgram)
            high = k.most_common(topN)
            return dict(high)
        return np.cumsum(sorted(dictOccurenceNgram.values(), reverse=True))/len(self.sequence)

    def averageEntropy(self) :
        entropy=0
        for ngram in self.transition_matrix.T :
            p = self.transition_matrix.T[ngram].values
            entropy += scp.entropy(p)
        return entropy/len(self.transition_matrix )

    def intraEntropy(self) :
        entropy=[]
        for i in range(len(testSeq)-self.order):
            currentNgram = tuple([testSeq[i+j] for j in range(self.order)])
            entropy += scp.entropy(p)
        return entropy/len(self.transition_matrix )

    def KLDivergence(self, sequence, alpha = 10**-10) :
        count = dict((key, alpha) for key in self.vocab_frequency.keys()) #Initialise to alpha to avoid /0
        for element in sequence :
            count[element]+=1
        return scp.entropy(list(count.values()), list(self.vocab_frequency.values()))
