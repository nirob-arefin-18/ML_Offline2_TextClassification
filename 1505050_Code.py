#!/usr/bin/env python
# coding: utf-8

# In[161]:


import numpy as np
import pandas as pd
import string
import nltk
import re
import csv
import math
from scipy import spatial
from scipy import stats
from itertools import islice
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from bs4 import BeautifulSoup as bs


# In[2]:


def strip_html(text):
    soup = bs(text, "html.parser")
    return soup.get_text()


# In[3]:


def preprocess_data(text):
    text = strip_html(text)
    
    #Lowercase the text
    text = text.lower()
    #Number Removal
    text = re.sub(r'[-+]?\d+', '', text)
    #Removing all slashes so that each work can be considered separately 
    text= text.replace("/"," ")
    #Remove punctuations
    text=text.translate((str.maketrans('','',string.punctuation)))
    #Tokenize
    text = word_tokenize(text)
    #Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if not word in stop_words]
    #Lemmatize tokens
    lemmatizer=WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]
    #Stemming tokens
    stemmer= PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    
    return text
    
    


# In[4]:


def find_union(list1, list2):

    set_1 = set(list1)
    set_2 = set(list2)

    result_set = set_1.union(set_2)
    result = list(result_set)
    
    return result



# In[5]:


# Dictionary d stores all the datas, where key is the topic name and value is a 2D list for preprocessed 
#words of each document
# e.g. List[i] = list of strings of preprocessed words for row i
d = {}

# This list stores unique words from all training documents documents
unique_words = []


topic_file = open('Data/topics.txt')

for each_line in topic_file:
    #print(each_line)
    topic_name = each_line.strip()
    List = []

    with open('Data/Training/'+topic_name+'.xml','r',encoding='utf-8') as file:
        content = file.read()
        soup = bs(content)
        
        for id_no in range(1,1201):
            
            for items in soup.findAll("row",id=id_no):
                body = items.get('body')
                body = preprocess_data(body)

            List.append(body)
            
            if(id_no<=500):
                unique_words = find_union(unique_words,body)

            
    d[topic_name]=List
    


# In[6]:



bool_representations = []   # boolean representations of all documents 
numeric_representations = [] # numeric representations of all documents

for (x,y) in d.items():
    topic_name = x

    
    for doc_no in range(0,len(y)):
        document = y[doc_no]
        
        #csv_one_row = [0 for v in range(len(unique_words))]
        bool_one_row = [0 for v in range(len(unique_words))]
        numeric_one_row = [0 for v in range(len(unique_words))]
        
        for i in range(0,len(unique_words)):
            temp1 = unique_words[i]

            for j in range(0,len(document)):
                temp2 = document[j]

                if(temp1 == temp2):
                    bool_one_row[i]=1
                    numeric_one_row[i]=numeric_one_row[i]+1
                    
        bool_one_row.append(topic_name)
        numeric_one_row.append(topic_name)
        
        bool_representations.append(bool_one_row)
        numeric_representations.append(numeric_one_row)

    


# In[7]:


# splitting into train, test and validation set
training_set_bool = []
validation_set_bool = []
test_set_bool = []

for i in range(0,len(bool_representations),1200):
    #print(i)
    for j in range(i,i+500):
        training_set_bool.append(bool_representations[j])
    for k in range(i+500,i+700):
        validation_set_bool.append(bool_representations[k])
    for l in range(i+700,i+1200):
        test_set_bool.append(bool_representations[l])
        

training_set_numeric = []
validation_set_numeric = []
test_set_numeric = []

for i in range(0,len(numeric_representations),1200):
    #print(i)
    for j in range(i,i+500):
        training_set_numeric.append(numeric_representations[j])
    for k in range(i+500,i+700):
        validation_set_numeric.append(numeric_representations[k])
    for l in range(i+700,i+1200):
        test_set_numeric.append(numeric_representations[l])
        
print(training_set_numeric[1000][-1])


# In[9]:


# Here instance is 1D list, a row 
def HammingDistance(instance1, instance2):
    distance = 0
    for i in range(0,len(instance1)):
        if(instance1[i]!=instance2[i]):
            distance+=1
            
    return distance
        
        


# In[10]:


def EuclideanDistance(instance1, instance2):
    distance = 0.0
    for i in range(0,len(instance1)):
        distance += (int(instance1[i]) - int(instance2[i]))**2
    return np.sqrt(distance)


# In[11]:


def FindAccuracy(actual, predicted):
    total_count = len(predicted)
    correct_count=0
    
    for i in range(0,len(predicted)):
        if(actual[i]==predicted[i]):
            correct_count+=1
            
    accuracy = (correct_count/total_count)*100
    return accuracy


# In[88]:


# Trainig set is the total training set, test document is only one document
#containing the features only
def Predict_document(train_X,train_Y,test_document_X,k):

    #this dictionary keys are row index, values are distnace and class types 
    dist_type = {} 
    
    for i in range(0,len(train_X)):
        #distance = HammingDistance(train_X[i],test_document_X)
        distance = EuclideanDistance(train_X[i],test_document_X)
        
        #distance = spatial.distance.cosine(train_X[i],test_document_X)
        #distance = spatial.distance.hamming(train_X[i],test_document_X)
        
        #distance = spatial.distance.euclidean(train_X[i],test_document_X)
        #print(distance)
        dist_type[i]=(distance,train_Y[i])
    
    
    #sort this dictionary according to distance values
    dist_type = dict(sorted(dist_type.items(), key=lambda item: item[1][0]))
    
    #find the first k neighbors from the sorted dict
    neighbors = dict(islice(dist_type.items(), k))
    
    neighbor_types=[]
    for (x,y) in neighbors.items():
        neighbor_types.append(y[1])

    #find the class type by finding the class with maximum occurances
    result = max(set(neighbor_types), key = neighbor_types.count)
    return result


# In[89]:


# Here paramter should be a row from the training_set and test or validation
def KNN(train_X,train_Y,test_X,test_Y,k):
    

    
    #stores predicted results for all documents
    all_predicted_outputs = []
    
    #epoch = 0
    for testInput in test_X:
        #print(epoch)
        predicted_output = Predict_document(train_X, train_Y,testInput,k)
        all_predicted_outputs.append(predicted_output)
        
        #epoch+=1
        
    
   
    acc = FindAccuracy(test_Y,all_predicted_outputs)
    return acc
    


# In[51]:



# k_val = 1

# KNN(training_set_numeric,validation_set_numeric,k_val)


# In[52]:


#### TF -IDF calculation starting

def X_Y_split(dataset):
    #index of the column containing class type
    y_ind = len(dataset[0])-1
    
    X = np.delete(dataset,y_ind,axis=1)
    Y = [row[y_ind] for row in dataset]
    
    return X,Y
    


# In[63]:


# splitting feature and class
train_numeric_X,train_Y = X_Y_split(training_set_numeric)
train_numeric_X = [list( map(int,i) ) for i in train_numeric_X]

validation_numeric_X,validation_Y = X_Y_split(validation_set_numeric)
validation_numeric_X = [list( map(int,i) ) for i in validation_numeric_X]

test_numeric_X,test_Y = X_Y_split(test_set_numeric)
test_numeric_X = [list( map(int,i) ) for i in test_numeric_X]



train_bool_X,train_Y = X_Y_split(training_set_bool)
train_bool_X = [list( map(int,i) ) for i in train_bool_X]

validation_bool_X,validation_Y = X_Y_split(validation_set_bool)
validation_bool_X = [list( map(int,i) ) for i in validation_bool_X]

test_bool_X,test_Y = X_Y_split(test_set_bool)
test_bool_X = [list( map(int,i) ) for i in test_bool_X]


# In[26]:


def finding_tf(dataset_X):
    dataset_tf = []
    length = len(dataset_X[0])  #since all documents have same number of columns
    for i in range(0,len(dataset_X)):
        doc = dataset_X[i]
        doc_tf = []
        total_w = sum(doc)
        
        if(total_w==0):
            total_w = 1
            
        for j in range(0,length):
            val = doc[j]
            tf= val/total_w
            doc_tf.append(tf)
            
        dataset_tf.append(doc_tf)
    
    return dataset_tf
    


# In[27]:


def finding_idf(dataset_X):
    IDF = []
    D = len(dataset_X)
    column_as_row = list(zip(*dataset_X))
    
    for i in range(0,len(column_as_row)):
        d = np.count_nonzero(column_as_row[i])
        val = math.log(D/d)
        
        IDF.append(val)
        
    return IDF

    


# In[28]:


# tf_values is 2d list, idf_values is 1D list
def finding_tf_idf(tf_values,idf_values):
    tf_idf = []
    for row in tf_values:
        temp = np.multiply(row,idf_values)
        tf_idf.append(temp)
        
    return tf_idf
    


# In[29]:


### Getting tf and idf separately
training_tf = finding_tf(train_numeric_X)
validation_tf = finding_tf(validation_numeric_X)
test_tf = finding_tf(test_numeric_X)

IDF_for_unique_words= finding_idf(train_numeric_X) #calculated on training set


# In[30]:


training_tf_idf = finding_tf_idf(training_tf,IDF_for_unique_words)
validation_tf_idf = finding_tf_idf(validation_tf,IDF_for_unique_words)
test_tf_idf = finding_tf_idf(test_tf,IDF_for_unique_words)


# In[67]:


### KNN function is used for all of the three distances cases.
### Just change the distance calculation in function Predict_document


#print(KNN(training_tf_idf,train_Y, validation_tf_idf,validation_Y,1))
print(KNN(train_bool_X,train_Y, validation_bool_X,validation_Y,1))


# In[92]:


### From the report , best KNN is for Euclidean distance and k=1
### This is for KNN
## this one iteration takes 10 documents form test set of each topic

def RunOneItrn(train_X,train_Y,test_X,test_Y,k,ind):
    temp_test_X = []
    temp_test_Y =[]
    
    for i in range(0,len(test_X),500):
        for j in range(i+ind,i+ind+10):
            temp_test_X.append(test_X[j])
            temp_test_Y.append(test_Y[j])
            
    result=KNN(train_X,train_Y,temp_test_X,temp_test_Y,k)
    return result


# In[80]:


#RunOneItrn(train_numeric_X,train_Y,test_numeric_X,test_Y,1,0)


# In[95]:



KNN_file = open("KNN_result.txt","w")
KNN_file.write('KNN accuracy result for 50 iterations over test set.\nEuclidean Distance and k=1\n\n')

for i in range(0,50):
    index = i*10
    accuracy = RunOneItrn(train_numeric_X,train_Y,test_numeric_X,test_Y,1,index)
    line = 'Iteration '+str(i+1)+': '+str(accuracy)+'\n'
    print(line)
    KNN_file.write(line)

KNN_file.close()


# In[115]:


### Naive Bayes Implementation

## To simplify calculation, this dictionary is introduced.
## dictionary value is the class type 

class_value_dict = {}

j = 0
for i in range(0,len(training_set_numeric),500):
    class_value_dict[j]=training_set_numeric[i][-1]
    j+=1
    
#print(class_value_dict)


# In[103]:


#which is NCk on slide
def find_total_words(dataset):
    count = 0
    
    for i in range(0,len(dataset)):
        count+=sum(dataset[i])
        
    return count
        


# In[110]:


# which is Nwi,ck on slide. This counts each word occurances
def find_occurances_of_words(dataset):
    occurance_count=[]
    
    column_as_row = list(zip(*dataset))
    
    for i in range(0,len(column_as_row)):
        occurance_count.append(sum(column_as_row[i]))
        
    return occurance_count
 


# In[112]:


# which is V in slide
def total_diff_words(dataset):
    count=0
    column_as_row = list(zip(*dataset))
    
    for i in range(0,len(column_as_row)):
        temp = np.count_nonzero(column_as_row[i])
        if(temp!=0):
            count+=1
            
    return count
        


# In[114]:


class_wise_total_words=[]  #1D list 
class_wise_occurances_of_words= [] #2D list, row is the class number
class_wise_total_diff_words=[] #1D list


# In[128]:


### finding on all training documents
for i in range(0,len(train_numeric_X),500):
    one_class_documents=[]
    for j in range(i,i+500):
        one_class_documents.append(train_numeric_X[j])
        
    class_wise_total_words.append(find_total_words(one_class_documents))
    class_wise_occurances_of_words.append(find_occurances_of_words(one_class_documents))
    class_wise_total_diff_words.append(total_diff_words(one_class_documents))
    


# In[130]:



def predict_a_doc_NB(test_doc,total_words,occur_words,diff_words,class_val_dict,alpha):
    class_count= len(occur_words)
    
    prob_all_classes=[]
    
    for i in range(0,class_count):
        prob_this_class = 0
        total_w = total_words[i] #single value
        total_diff_w = diff_words[i] #single value
        occurances_w= occur_words[i]  #its a list
        
        for j in range(0,len(test_doc)):
            
            if(test_doc[j]!=0):
                p = (occurances_w[j]+alpha)/(total_w + alpha*total_diff_w)
                prob_this_class+=p
                
        prob_all_classes.append(prob_this_class)
        
    max_index = prob_all_classes.index(max(prob_all_classes))
    
    predicted_class = class_val_dict[max_index]
    
    return predicted_class
    
        


# In[141]:


def NB(test_X,test_Y,total_words,occur_words,diff_words,class_val_dict,alpha):

    
    #stores predicted results for all documents
    all_predicted_outputs = []
    
    epoch = 0
    for testInput in test_X:
        #print(epoch)
        predicted_output = predict_a_doc_NB(testInput,total_words,occur_words,diff_words,class_val_dict,alpha)
        all_predicted_outputs.append(predicted_output)
        
        epoch+=1

    acc = FindAccuracy(test_Y,all_predicted_outputs)
    return acc
    


# In[154]:


#NB(validation_numeric_X,validation_Y,class_wise_total_words,class_wise_occurances_of_words,class_wise_total_diff_words,class_value_dict,2)


# In[143]:


# NB on validation set for 10 diff alpha
alpha = 0.1
for i in range(0,10):
    res=NB(validation_numeric_X,validation_Y,class_wise_total_words,class_wise_occurances_of_words,class_wise_total_diff_words,class_value_dict,alpha)
    print('alpha ',alpha,' acc :',res)
    alpha+=0.1


# In[159]:


### From the report , best NB for smoothing factor 0.005
# this one iteration takes 10 documents form test set of each topic


def RunOneItrn_NB(test_X,test_Y,total_words,occur_words,diff_words,class_val_dict,alpha,ind):
    
    temp_test_X = []
    temp_test_Y =[]
    
    for i in range(0,len(test_X),500):
        
        for j in range(i+ind,i+ind+10):
            temp_test_X.append(test_X[j])
            temp_test_Y.append(test_Y[j])
            
    result = NB(temp_test_X,temp_test_Y,total_words,occur_words,diff_words,class_val_dict,alpha)
    return result
        
        


# In[160]:


NB_file = open("NB_result.txt","w")
NB_file.write('NB accuracy result for 50 iterations over test set.\nSmoothing factor is 0.005\n\n')

for i in range(0,50):
    ind = i*10
    
    accuracy = RunOneItrn_NB(test_numeric_X,test_Y,class_wise_total_words,class_wise_occurances_of_words,class_wise_total_diff_words,class_value_dict,alpha,ind)
    line = 'Iteration '+str(i+1)+': '+str(accuracy)+'\n'
    #print(line)
    NB_file.write(line)

NB_file.close()


# In[168]:


#### t- statistics calculation

file1 = open('KNN_result.txt','r')
KNN_result_for_test=[]

for each_line in file1 :
    each_line = each_line.strip()
    
    stream = each_line.split()
    KNN_result_for_test.append(float(stream[-1]))

#print(KNN_result_for_test)


file2 = open('NB_result.txt','r')
NB_result_for_test=[]

for each_line in file2 :
    each_line = each_line.strip()
    
    stream = each_line.split()
    NB_result_for_test.append(float(stream[-1]))

#print(NB_result_for_test)


# In[171]:


# Here ttest_rel is used, since test set is same for two impelementations

print(stats.ttest_rel(NB_result_for_test,KNN_result_for_test))


# In[ ]:




