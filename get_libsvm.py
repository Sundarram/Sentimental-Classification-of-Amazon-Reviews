import argparse
import sys

import numpy as np

from sklearn.externals import joblib


def read_raw_reviews(file_name):
    ret_dict = {}
    ret_dict['data'] = []
    ret_dict['target'] = []
    ff = open(file_name, 'r')
    for fl in ff:
        toks = fl.strip().split(' ')
        ret_dict['target'].append(toks[0])
        ret_dict['data'].append(' '.join(toks[1:]))
    ff.close()
    
    return ret_dict


if __name__ == '__main__':
    
    try:
        parser = argparse.ArgumentParser(description='lib-svm representation for text')
        parser.add_argument('fname',
                            help='Path to file with text data')

        parser.add_argument('-tfidf',
                            required=True,
                            help='Path to tf-idf vectorizer')

        parser.add_argument('-lsa',
                            default=None,
                            required=False,
                            help='Path to LSA embedding model')

        parser.add_argument('-pca',
                            default=None,
                            required=False,
                            help='Path to PCA embedding model')
        
        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(0)
            
        args = parser.parse_args()
        print("Reading from %s " % (args.fname,))
        out_fname = "%s.svm" % (args.fname,)
        
        text_data=read_raw_reviews(args.fname)
        
        print ("Loaded %d samples from %s" % (len(text_data['target']), args.fname))
        
        ### load tf-idf vectorizer from text
        from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer, TfidfTransformer
        vectorizer = joblib.load(args.tfidf)
        print("transforming tf-idf")
        
        trn_tfidf = vectorizer.transform(text_data['data'])

        print("tf-idf shape=%s" % (trn_tfidf.shape,))
        
        trn_lsa = None
        trn_pca = None
        
        if args.lsa is not None:
            ### perform dimensionality reduction (LSA) for sparse tf-idf representation
            from sklearn.decomposition import TruncatedSVD
            lsa_obj=joblib.load(args.lsa)
            
            print("transforming lsa")
            trn_lsa = lsa_obj.transform(trn_tfidf)  ### embed tf-idf vectors into 200 dimensions
            print("LSA shape=%s" % (trn_lsa.shape,))
            
            if args.pca is None:
                from sklearn.datasets import dump_svmlight_file, load_svmlight_file
                dump_svmlight_file(trn_lsa, np.array(text_data['target'], dtype='float'), out_fname)

        if args.pca is not None:
            
            ### perform dimensionality reduction (PCA) for dense vector representation for text documents
            ### that was obtained using LSA dimensionality reduction applied to tf-idf representation
            from sklearn.decomposition import PCA
            pca_obj=joblib.load(args.pca)

            print("transforming pca")
            trn_pca = pca_obj.transform(trn_lsa)  ### embed 200-dim LSA vectors into 100 dimensions
            
            print("PCA shape=%s" % (trn_pca.shape,))
            
            from sklearn.datasets import dump_svmlight_file, load_svmlight_file
            dump_svmlight_file(trn_pca, np.array(text_data['target'], dtype='float'), out_fname)
            
        if args.pca is None and args.lsa is None:
            from sklearn.datasets import dump_svmlight_file, load_svmlight_file
            dump_svmlight_file(trn_tfidf, np.array(text_data['target'], dtype='float'), out_fname)
            
    except Exception, exc:
        import traceback
        print('Exception was raised in %s of %s: %s \n %s ' % (__name__, __file__, str(exc), ''.join(traceback.format_exc())))

