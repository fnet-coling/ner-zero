
from gensim.models import Word2Vec
from gensim.matutils import argsort
from numpy import array,asarray
import numpy as np
import sys
from fast_read import *
from TypeHierarchy import TypeHierarchy
class WARPModel:
    def __init__(self,label_vec_f,feature_vec_f,binary=False,threshold=0):
        self.vocab,self.dictionary,self.label_embed = readEmbed(label_vec_f)
        _,_,self.feat_embed = readEmbed(feature_vec_f)
	self.threshold = threshold
    def scores(self,X):
        scores =  np.dot(np.sum(self.feat_embed[X],axis=0),self.label_embed.T)
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
                    label.append(ranks[0])
                else:
                    label = [l for l in ranks[:topk] if scores[l] > self.threshold] 
                labels.append(label)

            return labels#[map(lambda x:self.dictionary[x],self.label_rank(X[1])[:topk]) for X in Xs]
        else:
            return None
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
def refine(labels,tier,maxDepth=2,delim='/'):
    keep = [""]*maxDepth
    for l in labels:
        path = tier.get_type_path(l)
        for i in range(len(path)):
            if keep[i] =="" :
                keep[i] = path[i]
            elif keep[i] != path[i]:break

    return [l for l in keep if l != ""]       
def getPath(label,delim='/'):
    return label.split(delim)[1:] 
def main():
    label_embed_fn = sys.argv[1]
    feature_embed_fn = sys.argv[2]
    data_fn = sys.argv[3]
    mention_map = build_mention_map(sys.argv[4])
    threshold = float(sys.argv[8])
    model = WARPModel(label_embed_fn,feature_embed_fn,threshold=threshold)
    _label_size =  len(model.label_embed)
    tier = TypeHierarchy(sys.argv[9]+"/supertype.txt",_label_size)
    data = read_data(data_fn)
    delimer=sys.argv[7]
    out = open(sys.argv[5],'wb')
    depth = int(sys.argv[6])
#    threshold = float(sys.argv[8])
    predictions = model.batch_predict(data,use_text=True,topk=depth)
    for i in xrange(len(predictions)):
        for l in refine(predictions[i],tier,delim=delimer,maxDepth=depth):
            out.write("%s\t%s\t1\n" % (mention_map[data[i][0]],str(l)))
    out.close()


if __name__ == "__main__":
	    main()      
