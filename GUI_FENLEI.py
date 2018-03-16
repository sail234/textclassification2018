# -*- coding: utf-8 -*- 

###########################################################################
## Python code generated with wxFormBuilder (version Jun 17 2015)
## http://www.wxformbuilder.org/
##
## PLEASE DO "NOT" EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc

###########################################################################
## Class MyFrame3
###########################################################################

# -*- coding: utf-8 -*-  
import jieba  
import os  
import re  
import time  
import string 
import pickle 
from sklearn.naive_bayes import MultinomialNB  
from sklearn.cross_validation import train_test_split   
from sklearn.cross_validation import StratifiedKFold  
from sklearn.cross_validation import KFold  
from sklearn.metrics import precision_recall_curve    
from sklearn.metrics import  classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC  
 
#from Tkinter import *
#import FileDialog
import tkFileDialog

diction = ""
def train_mo(rootpath):
    os.chdir(rootpath)  
    # stopword  
    words_list = []                                      
    filename_list = []  
    category_list = []  
    all_words = {}                                # 全词库 {'key':value }  
    stopwords = {}.fromkeys([line.rstrip() for line in open('G:/machine learning/textclassification2018/stopwords.txt')])  
    category = os.listdir(rootpath)               # 类别列表  
    delEStr = string.punctuation + ' ' + string.digits  
    identify = string.maketrans('', '')     
    #########################  
    #       分词，创建词库    #  
    #########################  
    def fileWordProcess(contents):  
        wordsList = []
        alist = []  
        contents = re.sub(r'\s+',' ',contents) # trans 多空格 to 空格  
        contents = re.sub(r'\n',' ',contents)  # trans 换行 to 空格  
        contents = re.sub(r'\t',' ',contents)  # trans Tab to 空格  
        contents = contents.translate(identify, delEStr)   
        for seg in jieba.cut(contents):  
            seg = seg.encode('utf8')
            alist.append(seg)  
            if seg not in stopwords:           # remove 停用词  
                if seg!=' ':                   # remove 空格  
                    wordsList.append(seg)      # create 文件词列表  
        file_string = ' '.join(wordsList)              
        return file_string,set(alist)  
    aword = set([])  
    for categoryName in category:             # 循环类别文件，OSX系统默认第一个是系统文件  
        if(categoryName=='.DS_Store'):continue  
        categoryPath = os.path.join(rootpath,categoryName) # 这个类别的路径  
        filesList = os.listdir(categoryPath)      # 这个类别内所有文件列表  
        # 循环对每个文件分词  
        k = 0 #计算每类文件个数
        for filename in filesList:  
            if(filename=='.DS_Store'):continue  
            starttime = time.clock()  
            contents = open(os.path.join(categoryPath,filename)).read()  
            wordProcessed,alist = fileWordProcess(contents)       # 内容分词成列表
            aword = aword | alist 
    #暂时不做#filenameWordProcessed = fileWordProcess(filename) # 文件名分词，单独做特征  
    #         words_list.append((wordProcessed,categoryName,filename)) # 训练集格式：[(当前文件内词列表，类别，文件名)]  
            words_list.append(wordProcessed)  
            filename_list.append(filename)  
            category_list.append(categoryName)  
            endtime = time.clock();   
            k += 1
            if(k >= 100):#取前100个文件
                break
            print 'category:%s >>>>file:%s >>>>time: %.3f' % (categoryName,filename,endtime-starttime)  
#    print len(aword)
    # 创建词向量矩阵，创建tfidf值矩阵  
      
    from sklearn.feature_extraction.text import TfidfTransformer  
    from sklearn.feature_extraction.text import CountVectorizer  
    freWord = CountVectorizer(stop_words='english')  
    transformer = TfidfTransformer()  
    fre_matrix = freWord.fit_transform(words_list)  
    tfidf = transformer.fit_transform(fre_matrix)  
      
    import pandas as pd  
    feature_names = freWord.get_feature_names()           # 特征名  
    freWordVector_df = pd.DataFrame(fre_matrix.toarray()) # 全词库 词频 向量矩阵  
    tfidf_df = pd.DataFrame(tfidf.toarray())              # tfidf值矩阵  
    # print freWordVector_df  
    tfidf_df.shape  
    
    # tf-idf 筛选  
    tfidf_sx_featuresindex = tfidf_df.sum(axis=0).sort_values(ascending=False)[:10000].index  
#    print len(tfidf_sx_featuresindex)  
    freWord_tfsx_df = freWordVector_df.ix[:,tfidf_sx_featuresindex] # tfidf法筛选后的词向量矩阵  
    df_columns = pd.Series(feature_names)[tfidf_sx_featuresindex]  
#    print df_columns.shape  
    def guiyi(x):  
        x[x>1]=1  
        return x  
    import numpy as np  
    tfidf_df_1 = freWord_tfsx_df.apply(guiyi)  
    tfidf_df_1.columns = df_columns  
    from sklearn import preprocessing  
    le = preprocessing.LabelEncoder()  
    tfidf_df_1['label'] = le.fit_transform(category_list)  
    tfidf_df_1.index = filename_list  
    
    # 卡方检验  
    from sklearn.feature_selection import SelectKBest, chi2  
    ch2 = SelectKBest(chi2, k=7000)  
    nolabel_feature = [x for x in tfidf_df_1.columns if x not in ['label']]  
    ch2_sx_np = ch2.fit_transform(tfidf_df_1[nolabel_feature],tfidf_df_1['label'])  
    label_np = np.array(tfidf_df_1['label'])  
    
    # 朴素贝叶斯  
    X = ch2_sx_np  
    y = label_np 
    
    output2 = open('G:/machine learning/textclassification2018/X.pkl', 'wb')
    pickle.dump(X,output2, -1)
    output2.close()
    
    output3 = open('G:/machine learning/textclassification2018/y.pkl', 'wb')
    pickle.dump(y,output3, -1)
    output3.close()
    print "preprocessing finished"


class MyFrame3 ( wx.Frame ):

    def __init__( self, parent ):
        wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 623,378 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
        
        self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
        
        bSizer8 = wx.BoxSizer( wx.VERTICAL )
        
        self.m_button28 = wx.Button( self, wx.ID_ANY, u"选择数据", wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer8.Add( self.m_button28, 0, wx.ALL, 5 )
        
        self.m_button27 = wx.Button( self, wx.ID_ANY, u"预处理", wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer8.Add( self.m_button27, 0, wx.ALL, 5 )
        
        gSizer12 = wx.GridSizer( 0, 2, 0, 0 )
        
        gSizer13 = wx.GridSizer( 0, 2, 0, 0 )
        
        self.m_button20 = wx.Button( self, wx.ID_ANY, u"朴素贝叶斯", wx.DefaultPosition, wx.DefaultSize, 0 )
        gSizer13.Add( self.m_button20, 0, wx.ALL, 5 )
        
        self.m_button21 = wx.Button( self, wx.ID_ANY, u"KNN", wx.DefaultPosition, wx.DefaultSize, 0 )
        gSizer13.Add( self.m_button21, 0, wx.ALL, 5 )
        
        
        gSizer12.Add( gSizer13, 1, wx.EXPAND, 5 )
        
        self.m_button22 = wx.Button( self, wx.ID_ANY, u"SVM", wx.DefaultPosition, wx.DefaultSize, 0 )
        gSizer12.Add( self.m_button22, 0, wx.ALL, 5 )
        
        
        bSizer8.Add( gSizer12, 1, wx.EXPAND, 5 )
        
        
        self.SetSizer( bSizer8 )
        self.Layout()
        
        self.Centre( wx.BOTH )
        
        # Connect Events
        self.m_button28.Bind( wx.EVT_BUTTON, self.select_data )
        self.m_button27.Bind( wx.EVT_BUTTON, self.train_model )
        self.m_button20.Bind( wx.EVT_BUTTON, self.bayes_model )
        self.m_button21.Bind( wx.EVT_BUTTON, self.knn_model )
        self.m_button22.Bind( wx.EVT_BUTTON, self.SVM_model )

    
    def __del__( self ):
        pass
    
    
    # Virtual event handlers, overide them in your derived class
    def select_data( self, event):
#        ff = FileDialog.askdirectory()
        ff = tkFileDialog.askdirectory()
        ff = ff.encode("utf-8")
        print(ff)
        global diction
        diction = ff
        event.Skip()
    
    def train_model( self, event ):
        train_mo(diction)
        event.Skip()
    
    
    def bayes_model( self, event):
        pkl_file = open('G:/machine learning/textclassification2018/X.pkl', 'rb')
        X = pickle.load(pkl_file)
        pkl_file.close()
        pkl_file = open('G:/machine learning/textclassification2018/y.pkl', 'rb')
        y = pickle.load(pkl_file)
        pkl_file.close() 
        y_pre = y.copy()  
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3) 
        clf = MultinomialNB().fit(X_train, y_train)    
        y_pre = clf.predict(X_test) 
        classification_report(y_test,y_pre) 
        print "bayes"
        print(classification_report(y_test,y_pre))
        event.Skip()
    
    def knn_model( self, event ):
        pkl_file = open('G:/machine learning/textclassification2018/X.pkl', 'rb')
        X = pickle.load(pkl_file)
        pkl_file.close()
        pkl_file = open('G:/machine learning/textclassification2018/y.pkl', 'rb')
        y = pickle.load(pkl_file)
        pkl_file.close()
        neigh = KNeighborsClassifier(n_neighbors = 3)
        y_pre = y.copy()  
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)  
        clf = neigh.fit(X_train, y_train)    
        y_pre = clf.predict(X_test)
        classification_report(y_test,y_pre) 
        print "knn"
        print(classification_report(y_test,y_pre))
        event.Skip()
    
    def SVM_model( self, event ):
        pkl_file = open('G:/machine learning/textclassification2018/X.pkl', 'rb')
        X = pickle.load(pkl_file)
        pkl_file.close()
        pkl_file = open('G:/machine learning/textclassification2018/y.pkl', 'rb')
        y = pickle.load(pkl_file)
        pkl_file.close()  
        y_pre = y.copy() 
        svclf = LinearSVC() 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)  
        clf = svclf.fit(X_train, y_train)    
        y_pre = clf.predict(X_test)  
        classification_report(y_test,y_pre)  
        print "SVM"
        print classification_report(y_test,y_pre)
        event.Skip()


	

