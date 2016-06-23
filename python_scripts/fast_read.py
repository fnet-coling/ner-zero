import numpy as np
import sys
def readEmbed(fn):
    with open(fn,'r') as dat:
        size,dim = map(int,dat.readline().rstrip().split())
        index2word = [""]*size
        embedding = np.zeros([size,dim])
        step = size/10000+1
        dictionary = dict()
        lcnt= 0 
        for ln in dat:
            if lcnt%step==0:
                sys.stdout.write("\rloaded %f%%" % (lcnt*100.0/size)) 
                sys.stdout.flush()
            splited = ln.rstrip().split()
            index2word[lcnt] = splited[0]
            embedding[lcnt] = np.asarray(map(float,splited[1:]))
            dictionary[splited[0]] =lcnt
            lcnt+=1
    return dictionary,index2word,embedding