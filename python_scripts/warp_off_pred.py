
from gensim.models import Word2Vec
from gensim.matutils import argsort
from numpy import array,asarray
import numpy as np
import sys
from TypeHierarchy import TypeHierarchy
from collections import Counter
class WARPModel:
    def __init__(self,label_vec_f,feature_vec_f,binary=False):
        label2vec = Word2Vec.load_word2vec_format(label_vec_f,binary=binary)
        self.label_embed =  label2vec.syn0
        self.dictionary = label2vec.index2word
        self.vocab = label2vec.vocab
        self.feat_embed = Word2Vec.load_word2vec_format(feature_vec_f,binary=binary).syn0
    def scores(self,X):
        return np.dot(np.sum(self.feat_embed[X],axis=0),self.label_embed.T)
    def label_rank(self,X):
	scores = self.scores(X)
        return scores,argsort(scores,reverse=True)
    def batch_predict(self,Xs,topk=2,use_text=False):
        if use_text:
	    labels = []
	    scores = []
	    for X in Xs:
	       	score,ranks = self.label_rank(X[1])
		#print map(lambda x:self.dictionary[x],ranks[:topk]),[ scores[l] for l in ranks[:topk]]
		    
		labels.append(ranks[:topk])
		scores.append([score[l] for l in ranks[:topk]])
            return scores,labels
        else:return None
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
def refine(model,scores,labels,type_hierarchy,maxDepth=2,delim='/'):
    results = []
    labels = [labels[i] for i in range(len(labels)) if scores[0] - scores[i] <0.55]
 #   print [tuple([model.dictionary[labels[i]],scores[i]]) for i in range(len(scores)) if scores[0]-scores[i]<0.35]
    pathes = [ type_hierarchy.get_type_path(labels[i]) for i in range(len(labels))]
    #results = list(set([l for p in pathes for l in p]))
    for i in range(maxDepth):
        ids = [ k for k,p in enumerate(pathes) if len(p)>i]
	if len(ids) == 0:continue
        cnts = Counter([pathes[idx][i] for idx in ids])
        max_l = cnts.most_common()[0]
        for j in range(len(ids)):
                if results!=[] and pathes[ids[j]][i-1]!=results[-1]:
                    continue
                    
                if cnts[pathes[ids[j]][i]] == max_l[1]:
                    results.append(pathes[ids[j]][i])
                    break
        
#    results = [""]*maxDepth 
      
 #   for l in labels:
  #      p = type_hierarchy.get_type_path(l)
#	for i in range(len(p)):
#		if results[i] == "":
#			results[i] = model.dictionary[p[i]]
#		else:break
    return map(lambda x:model.dictionary[x],results)#results 
def getPath(label,delim='/'):
    return labelx.split(delim)[1:] 
def main():
    label_embed_fn = sys.argv[1]
    feature_embed_fn = sys.argv[2]
    data_fn = sys.argv[3]
    mention_map = build_mention_map(sys.argv[4])
    hierarchy_file = sys.argv[8] + '/supertype.txt'
    model = WARPModel(label_embed_fn,feature_embed_fn)
    _label_size = len(model.vocab)
    type_hierarchy = TypeHierarchy(hierarchy_file, _label_size)

    data = read_data(data_fn)
    max_depth=int(sys.argv[6])
    delimer=sys.argv[7]
    out = open(sys.argv[5],'wb')
    scores,predictions = model.batch_predict(data,use_text=True,topk=max_depth)
    for i in xrange(len(predictions)):
        for l in refine(model,scores[i],predictions[i],type_hierarchy,maxDepth=max_depth,delim=delimer):
	    if l in model.vocab:
            	out.write("%s\t%s\t1\n" % (mention_map[data[i][0]],str(model.vocab[l].index)))
    out.close()


if __name__ == "__main__":
	    main()      
