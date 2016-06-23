from DataIO import *
import json
import sys
def compare(item1, item2):
    if item1['start'] != item2['start']:
        return item1['start'] - item2['start']
    else:
        return item2['end'] - item1['end']
def putback(sent_json, mention_mapping, label_mapping, clean_mentions):
    fileid = sent_json['fileid']
    senid = sent_json['senid']
    tokens = sent_json['tokens']
    pivot = 0
    result = []
    mentions = sent_json['mentions']
    sorted_m = sorted(mentions, cmp=compare)
    for m in sorted_m:
        start = m['start']
        end = m['end']
        if end - start == 1:
            mention_name = '[%s]' % (tokens[start])
        else:
            mention_name = '[%s]' % (' '.join(tokens[start:end]))
        if pivot <= start:
            result.extend(tokens[pivot:start])
            result.append(mention_name)
            # find predicted labels if any
            m_name = '%s_%d_%d_%d'%(fileid, senid, start, end)
            if m_name in mention_mapping:
                m_id = mention_mapping[m_name]
                if m_id in clean_mentions:
                    clean_labels = [label_mapping[l] for l in clean_mentions[m_id]]
                    result.append(':'+'['+','.join(clean_labels)+']')
        pivot = end
    if pivot < len(tokens):
        result.extend(tokens[pivot:])
    result = ' '.join([x for x in result if x is not None])
    return fileid+':'+str(senid)+'\t'+result + '\n'
def casestudy(filename, output, mention_mapping, label_mapping, clean_mentions):
    with open(filename) as f, open(output, 'w') as g:
        for line in f:
            sent = json.loads(line.strip('\r\n'))
            result = putback(sent, mention_mapping, label_mapping, clean_mentions)
            if result is not '':
                g.write(result+'\n')
def main():
    raw_data= sys.argv[1]#'../../Data/BBN/'
    data=sys.argv[3]#'../../Intermediate/BBN/'
    output = sys.argv[2]
    type_file = data + '/type.txt'
    mention_file = data + '/mention.txt'
    json_file = raw_data + '/test.json'
    test_y_file = sys.argv[4]#outdir+ 'warp_predictions'
    mention_mapping = load_map(mention_file, 'mention')
    label_mapping = load_map(type_file, 'label')
    clean_mentions = load_mention_type(test_y_file)
    casestudy(json_file, output, mention_mapping, label_mapping, clean_mentions)
if __name__ == "__main__":
	    main()      
