
from gensim.models import Word2Vec
from gensim.matutils import argsort
from numpy import array,asarray
import numpy as np
import sys
from fast_read import *
from TypeHierarchy import TypeHierarchy
import cPickle as pickle
def max_vec_and_score(x,vecs):
    max_score=  -float('inf')
    max_vec = vecs[0]
    for v in vecs:
        score = np.dot(x,v)
        if score >max_score:
            max_vec = v
            max_score= score
    return max_score,max_vec
def top_vec_and_score(x,vecs,topk):
    max_score=  -float('inf')
    max_vec = vecs[0]
    scores= [np.dot(x,v) for v in vecs]
    ranks = argsort(scores,reverse=True)
    avg = np.mean([vecs[r] for r in ranks[:topk]],axis=0)
    return np.mean([scores[r] for r in ranks[:topk]]),avg
class WARPModel:
    def __init__(self,label_vec_f,label_vec_f_add,feature_vec_f,binary=False,threshold=0):
	    self.label_embed = pickle.load(open(label_vec_f,'r'))
            _,_,self.feat_embed = readEmbed(feature_vec_f)
	    _,_,self.label_embed_add=readEmbed(label_vec_f_add)
	    self.threshold = threshold
    def scores(self,X):
        x= np.sum(self.feat_embed[X],axis=0)
#        scores =  [max_vec_and_score(x,vecs)[0]+np.dot(x,self.label_embed_add[i]) for i,vecs in enumerate(self.label_embed)]
	#scores =  [max_vec_and_score(x,vecs)[0] for i,vecs in enumerate(self.label_embed)]
	scores =  [top_vec_and_score(x,vecs,10)[0] for i,vecs in enumerate(self.label_embed)]
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
    label_embed_fn_add = sys.argv[10]
    model = WARPModel(label_embed_fn,label_embed_fn_add,feature_embed_fn,threshold=threshold)
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
