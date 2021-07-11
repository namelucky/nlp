import pandas as pd
from gensim.models import word2vec
import pickle
import numpy as np

def load( nfile):
    with open(nfile, 'rb') as file:
        return pickle.load(file)
def save(nfile, object):
    with open(nfile, 'wb') as file:
        pickle.dump(object, file)

def merge(t):
    '''
    :t：表示train or test
    '''
    java_y_se=load("./java/java_{}_idf.pickle".format(t))
    print("java 行数 {}".format(np.array(java_y_se).shape))

    python_y_se=load("./python/python_{}_idf.pickle".format(t))

    print("python 行数 {}".format(np.array(python_y_se).shape))

    c_y_se=load("./c++/c++_{}_idf.pickle".format(t))

    print("c++ 行数 {}".format(np.array(c_y_se).shape))


    ruby_y_se=load("./ruby/ruby_{}_idf.pickle".format(t))

    print("ruby 行数 {}".format(np.array(ruby_y_se).shape))

    html_y_se = load("./html/html_{}_idf.pickle".format(t))
    print("html 行数 {}".format(np.array(html_y_se).shape))

    objc_y_se = load("./c/c_{}_idf.pickle".format(t))
    print("c 行数 {}".format(np.array(objc_y_se).shape))

    java_y_se.extend(python_y_se)
    java_y_se.extend(c_y_se)
    java_y_se.extend(ruby_y_se)
    java_y_se.extend(html_y_se)
    java_y_se.extend(objc_y_se)
    save('./{}_idf.pickle'.format(t), java_y_se)
    print(np.array(java_y_se).shape)

merge("train")
merge("test")