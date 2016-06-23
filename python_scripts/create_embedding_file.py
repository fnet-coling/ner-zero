import sys
def convertEmbedding(vecf,df,out):
	output =open(out,'wb')
	dictionary =[ln.split()[0] for ln in open(df,'r')]
	with open(vecf,'r') as embeddings:
		head = embeddings.readline()
		vecs = [ln.rstrip() for ln in embeddings]
		assert(len(dictionary) == len(vecs))
		output.write(head)
		for i in range(len(vecs)):
			output.write("%s %s\n" % (dictionary[i],vecs[i]))
	output.close()
def main():
	vecf = sys.argv[1] #embedding file outputed by warp
	df = sys.argv[2] #dictionary file, e.g., feature.txt
	out = sys.argv[3] #output filename
	convertEmbedding(vecf,df,out)
if __name__ == "__main__":
	    main()

