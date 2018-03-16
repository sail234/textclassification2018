# -*- coding: utf-8 -*-  
import jieba  
import os  
import re  
import time  
import string  
rootpath="G:/machine learning/textclassification2018/answer1"  
os.chdir(rootpath)  
# stopword  
words_list = []                                      
filename_list = []  
category_list = []  
all_words = {}                                # 全词库 {'key':value }  
stopwords = {}.fromkeys([line.rstrip() for line in open('../stopwords.txt')])  
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
        print '类别:%s >>>>文件:%s >>>>导入用时: %.3f' % (categoryName,filename,endtime-starttime)  
print len(aword)
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
print len(tfidf_sx_featuresindex)  
freWord_tfsx_df = freWordVector_df.ix[:,tfidf_sx_featuresindex] # tfidf法筛选后的词向量矩阵  
df_columns = pd.Series(feature_names)[tfidf_sx_featuresindex]  
print df_columns.shape  
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
from sklearn.naive_bayes import MultinomialNB  
from sklearn.cross_validation import train_test_split   
from sklearn.cross_validation import StratifiedKFold  
from sklearn.cross_validation import KFold  
from sklearn.metrics import precision_recall_curve    
from sklearn.metrics import classification_report  
# nolabel_feature = [x for x in tfidf_df_1.columns if x not in ['label']]  
# x_train, x_test, y_train, y_test = train_test_split(ch2_sx_np, tfidf_df_1['label'], test_size = 0.2)  

X = ch2_sx_np  
y = label_np  
skf = StratifiedKFold(y,n_folds=5)  
y_pre = y.copy()  
for train_index,test_index in skf:  
    X_train,X_test = X[train_index],X[test_index]  
    y_train,y_test = y[train_index],y[test_index]  
    clf = MultinomialNB().fit(X_train, y_train)    
    y_pre[test_index] = clf.predict(X_test)  

print '准确率为 %.6f' %(np.mean(y_pre == y))  
# 精准率 召回率 F1score  
from sklearn.metrics import confusion_matrix,classification_report  
print 'precision,recall,F1-score如下：》》》》》》》》'  
print classification_report(y,y_pre)  


# KNN  
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors = 3)
y_pre = y.copy()  
for train_index,test_index in skf:  
    X_train,X_test = X[train_index],X[test_index]  
    y_train,y_test = y[train_index],y[test_index]  
    clf = neigh.fit(X_train, y_train)    
    y_pre[test_index] = clf.predict(X_test)  

print '准确率为 %.6f' %(np.mean(y_pre == y))  
# 精准率 召回率 F1score  
print 'precision,recall,F1-score如下：》》》》》》》》'  
print classification_report(y,y_pre)  


#SVM 
from sklearn.svm import LinearSVC   
y_pre = y.copy() 
svclf = LinearSVC() 

for train_index,test_index in skf:  
    X_train,X_test = X[train_index],X[test_index]  
    y_train,y_test = y[train_index],y[test_index]  
    clf = svclf.fit(X_train, y_train)    
    y_pre[test_index] = clf.predict(X_test)  

print '准确率为 %.6f' %(np.mean(y_pre == y))  
# 精准率 召回率 F1score  
print 'precision,recall,F1-score如下：》》》》》》》》'  
print classification_report(y,y_pre)  