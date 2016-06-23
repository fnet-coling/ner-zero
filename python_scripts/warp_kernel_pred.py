
from gensim.models import Word2Vec
from gensim.matutils import argsort
from numpy import array,asarray
import numpy as np
import sys
from fast_read import *

def select(A,B,k=10):
    C=np.zeros([A.shape[0],B.shape[0]])
    D = np.dot(A,B.T)
    for i in range(D.shape[1]):
        X = argsort(D[:,i],reverse=True)[:k]
        C[X,i] = 1.0
    return C
class WARPModel:
    def __init__(self,label_vec_f,feature_vec_f,k=800000,binary=False,threshold=0):
        #label2vec = Word2Vec.load_word2vec_format(label_vec_f,binary=binary)
        #self.label_embed =  label2vec.syn0
        #self.dictionary = label2vec.index2word
        #self.vocab = label2vec.vocab
	self.vocab,self.dictionary,self.label_embed = readEmbed(label_vec_f)
        _,_,self.feat_embed = readEmbed(feature_vec_f)
        self.threshold = threshold
        self.C = select(self.feat_embed,self.label_embed,k=k)
    def scores(self,X):

        scores =  [np.dot(np.sum(self.feat_embed[[f  for f in X if self.C[f][l] ==1. ]],axis=0),self.label_embed[l].T)for l in range(self.label_embed.shape[0])]
        return scores
    def label_rank(self,X):
	scores = self.scores(X)
        return scores,argsort(scores,reverse=True)
    def batch_predict(self,Xs,topk=2,use_text=False):
        if use_text:
	    labels = []
	    for X in Xs:
		label = []
	        scores,ranks = self.label_rank(X[1])
		if scores[ranks[0]]<self.threshold:
		    label.append(self.dictionary[ranks[0]])
		else:
		    label = [self.dictionary[l] for l in ranks[:topk] if scores[l] > self.threshold]	
		labels.append(label)

            return labels#[map(lambda x:self.dictionary[x],self.label_rank(X[1])[:topk]) for X in Xs]
        else:
            return [refine(self.label_rank(X[1])[:topk]) for X in Xs]
        
def read_data(fn):
    data = []
    with open(fn,'r') as f:
        for ln in f:
            splited = ln.rstrip().split()
            data.append([splited[0],map(int,splited[1].split(','))])
    return data
def build_mention_map(fn):
    mmap = dict()
    with open(fn,'r') as f:
        for ln in f:
            splited= ln.rstrip().split()
            mmap[splited[0]] = splited[1]
    return mmap
def refine(labels,maxDepth=2,delim='/'):
    keep = [""]*maxDepth
    for l in labels:
        path = getPath(l,delim)
        for i in range(len(path)):
            if keep[i] =="" :
                keep[i] = path[i]
            elif keep[i] != path[i]:
                break
    results = []
    tmp= ''
    for l in keep:
        if l!="":
            tmp+=delim
            tmp +=l
            results.append(tmp)
        
    return results        
def getPath(label,delim='/'):
    return label.split(delim)[1:] 
def main():
    label_embed_fn = sys.argv[1]
    feature_embed_fn = sys.argv[2]
    data_fn = sys.argv[3]
    mention_map = build_mention_map(sys.argv[4])
    threshold = float(sys.argv[8])
    model = WARPModel(label_embed_fn,feature_embed_fn,threshold=threshold,k=int(sys.argv[9]))
    data = read_data(data_fn)
    delimer=sys.argv[7]
    out = open(sys.argv[5],'wb')
    depth = int(sys.argv[6])
#    threshold = float(sys.argv[8])
    predictions = model.batch_predict(data,use_text=True,topk=depth)
    for i in xrange(len(predictions)):
        for l in refine(predictions[i],delim=delimer,maxDepth=depth):
            out.write("%s\t%s\t1\n" % (mention_map[data[i][0]],str(model.vocab[l])))
    out.close()


if __name__ == "__main__":
	    main()      
