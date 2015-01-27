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
        parser = argparse.ArgumentParser(description='tf-idf representation for text')
        parser.add_argument('fname',
                            help='Path to file with text data')
        
        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(0)
            
        args = parser.parse_args()
        print("Reading from %s " % (args.fname,))
        tfidf_fname="%s.tfidf" % (args.fname,)
        lsa_fname="%s.lsa" % (args.fname,)
        pca_fname="%s.pca" % (args.fname,)
        
        text_data=read_raw_reviews(args.fname)
        print(text_data['target'][10:15])
        print(text_data['data'][10:15])
        
        trn_size = int(float(len(text_data['target']))*0.6) ### will train classifier on 60% of text corpus
        
        ### construct tf-idf features for text corpus
        from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer, TfidfTransformer
        vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=10, decode_error='replace', ngram_range=(1,5), analyzer='word') #use_idf=True, smooth_idf=True)
        trn_tfidf = vectorizer.fit_transform(text_data['data'])

        ### save tf-idf vectorizer object to file
        joblib.dump(vectorizer, tfidf_fname)

        print("tf-idf shape=%s" % (trn_tfidf.shape,))

        ### perform dimensionality reduction (LSA) for sparse tf-idf representation
        from sklearn.decomposition import TruncatedSVD
        lsa_obj=TruncatedSVD(200, n_iterations=5)
        lsa_obj.fit(trn_tfidf)

        ### save "trained" LSA object to file
        joblib.dump(lsa_obj, lsa_fname)
       
        trn_lsa = lsa_obj.transform(trn_tfidf)  ### embed tf-idf vectors into 200 dimensions

        print("LSA shape=%s" % (trn_lsa.shape,))
        
        ### perform dimensionality reduction (PCA) for dense vector representation for text documents
        ### that was obtained using LSA dimensionality reduction applied to tf-idf representation
        from sklearn.decomposition import PCA
        pca_obj=PCA(100)
        pca_obj.fit(trn_lsa)

        ### save "trained" PCA object to file
        joblib.dump(pca_obj, pca_fname)

        trn_pca = pca_obj.transform(trn_lsa)  ### embed 200-dim LSA vectors into 100 dimensions

        print("PCA shape=%s" % (trn_pca.shape,))
       
        ### training Gaussian Naive Bayes classifier on three representations of text corpus
        from sklearn.naive_bayes import GaussianNB, MultinomialNB 
        tfidf_clf = MultinomialNB()
        tfidf_clf.fit(trn_tfidf[0:trn_size,:], text_data['target'][0:trn_size])
        tfidf_acc = tfidf_clf.score(trn_tfidf[trn_size:], text_data['target'][trn_size:])

        lsa_clf = GaussianNB()
        lsa_clf.fit(trn_lsa[0:trn_size,:], text_data['target'][0:trn_size])
        lsa_acc = lsa_clf.score(trn_lsa[trn_size:], text_data['target'][trn_size:])

        pca_clf = GaussianNB()
        pca_clf.fit(trn_pca[0:trn_size,:], text_data['target'][0:trn_size])
        pca_acc = pca_clf.score(trn_pca[trn_size:], text_data['target'][trn_size:])

        print ("Accuracies. tf-idf: %.4f, lsa: %.4f, pca: %.4f" % (tfidf_acc, lsa_acc, pca_acc))
    except Exception, exc:
        import traceback
        print('Exception was raised in %s of %s: %s \n %s ' % (__name__, __file__, str(exc), ''.join(traceback.format_exc())))

