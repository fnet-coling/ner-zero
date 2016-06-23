from collections import defaultdict
import sys
def convert(test_y_fn,idmap,out_fn):
    out = open(out_fn,'wb')
    with open(test_y_fn,'r') as Ys:
        for ln in Ys:
            mid,labels = ln.rstrip().split('\t')
            mid = idmap[mid]
            for l in labels.split(','):
                out.write('%s\t%s\t1\n' % (mid,l))
    out.close()
def main():
    data = sys.argv[1]
    mention2id= defaultdict()
    with open(data+'mention.txt','rb') as mf:
        for ln in mf:
            mid,iid = ln.rstrip().split()
            mention2id[mid]=iid   
    convert(sys.argv[2],mention2id,sys.argv[3])
if __name__ == "__main__":
	    main()      
