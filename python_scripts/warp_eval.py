from evaluation import *
import sys
def main():
    gold = load_labels(sys.argv[1])
    pred = load_labels(sys.argv[2])
    acc,macro_p,macro_r,macro_f,micro_p,micro_r,micro_f= evaluate(pred,gold)
    print "Accuracy:",acc
    print "macro Precision:",macro_p
    print "macro Recall:",macro_r
    print "macro F-score:", macro_f
    print "micro Precision:",micro_p
    print "micro Recall:",micro_r
    print "micro F-score:", micro_f
if __name__ == "__main__":
	    main() 
