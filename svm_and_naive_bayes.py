import csv
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.cross_validation import train_test_split
import seaborn as sns


"""X_train = []
Y_train = []
X_test = []
Y_actual = []"""
Y_predicted = []
Y_n_predicted = []
Y_actual_normal_list = []

def plot_cm(cm, classi,zero_diagonal=False):
   n = len(cm)
   fig = plt.figure(figsize=(8, 6), dpi=80, )
   plt.clf()
   ax = fig.add_subplot(111)
   ax.set_aspect(1)
   res = ax.imshow(np.array(cm), cmap=plt.cm.viridis,
                   interpolation='nearest')
   width, height = cm.shape
   fig.colorbar(res)
   if(classi == "SVM"):
      plt.savefig('SVM_confusion_Matrix.png', format='png')
   else:
      plt.savefig('NaiveBayes_confusion_Matrix.png', format='png')
   #plt.hold(False)
   #plt.show()

def plot(Y_predicted,Y_actual,classi):
   
    y_axis = [0,0,0,0,0]

    for i in range(len(Y_predicted)):
        p = int(Y_predicted[i-1]-1)
        y_axis[p] = y_axis[p]+1
    print("")
    if(classi == "naive"):
       print("Naive Bayes")
    else:
       print("SVM")
    for i in range(len(y_axis)):
       print("Number of ",i+1,"stars after prediction: ",y_axis[i])
    x_axis = [1,2,3,4,5]
    ind = np.arange(len(x_axis))
    #print(ind)
    
    plt.figure()
    """plt.subplot(211)"""

    plt.xlabel('Stars')
    plt.ylabel('Number')
    plt.title('Predicted Ratings')
    plt.xticks(ind,x_axis)
    plt.bar(ind,y_axis)
    if (classi =="naive"):
      plt.savefig('Naive_Bayes_Predicted.png', format='png')
    else:
      plt.savefig('SVM_Predicted.png', format='png')


    y_axis = [0,0,0,0,0]

    for i in range(len(Y_actual)):
        p = int(Y_actual[i-1]-1)
        y_axis[p] = y_axis[p]+1
    
    for i in range(len(y_axis)):
       print("Number of ",i+1,"stars in actual Data: ",y_axis[i])
    x_axis = [1,2,3,4,5]
    
    ind = np.arange(len(x_axis))
    #print(ind)
    

    
    plt.figure()
    """plt.subplot(211)"""
    plt.xticks(ind,x_axis)
    plt.bar(ind,y_axis)
    plt.xlabel('Stars')
    plt.ylabel('Number')
    plt.title('Actual Ratings')
    if(classi == "naive"):
       plt.savefig('Actual.png', format='png')
    x =1
    

    
    """plt.subplot(212)
    plt.plot(t, 2*s1)"""

def Naive_bayes(X_train,Y_train,X_test,Y_actual):
    clf = GaussianNB()
    clf.fit(X_train,Y_train)
    for i in range(len(X_test)):
        t = clf.predict([X_test[i]])
        Y_n_predicted.append(t[0])

    #print(Y_n_predicted)


def open_file_and_divide():
    n=0
    t=1

    data = pd.read_csv("MDATASET.csv")

    X = data[['P_Word_Count','N_Word_Count']].as_matrix(columns = None)
    Y = data[['Stars']].as_matrix(columns = None)

    X_train ,X_test,Y_train,Y_actual = train_test_split(X,Y,test_size = 0.5,random_state = 7)

    """print (X)

    
    print(Y)"""
    print("Number of rows in Training Set: ",len(X_train))
    print("Number of rows in Testing Set: ",len(X_test))
    """print(X_train)
    print(X_test)
    print(Y_train)
    print(Y_actual)"""
    
    

    #print(type(X))
    """ sns.pairplot (data,x_vars = ['P_Word_Count'],y_vars = ['N_Word_Count'],size = 7,aspect = 0.7)
    sns.plt.show()"""
    return X_train ,X_test,Y_train,Y_actual
    """with open(r"MDATASET (2).csv") as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            if(row):
                n=n+1
    print (n)
    with open(r"MDATASET (2).csv") as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            t = t +1
            if (t <= 3):
                pass
        
            elif (row and t<=(n/2 + 2)):
               # print(row[2])
                p = int(row[2])
                x= int(row[4])
                #type(p)
                s = [p,x]
                X_train.append(s)
                Y_train.append(int(row[5]))
            else:
                p = int(row[2])
                x= int(row[4])
                #type(p)
                s = [p,x]
                X_test.append(s)
                Y_actual.append(int(row[5]))
                """

    

"""with open('MDATASET.csv') as f:
    numbers = [[int(j) for j in i.split('\t')] for i in f.read().split('\n')]

print(numbers)
"""

#print(Y)

"""for i in range(5):
   # print('1')
    print(X[i])"""

#print(X_train)
#print(X_test)

def classify_and_predict(X_train,Y_train,X_test,Y_actual,kernel1):

    clf = svm.SVC(decision_function_shape='ovo' ,kernel = "linear")
    clf.fit(X_train, Y_train)



    for i in range(len(X_test)):
        t = clf.predict([X_test[i]])
        Y_predicted.append(t[0])
    #print(Y_predicted)

    

    
def find_accuracy(Y_actual,Y_predicted,classi):
    x = 0
    """print(Y_predicted)
    print(Y_actual)"""
    for i in range(len(Y_predicted)):
        if(Y_predicted[i] == Y_actual[i]):
            x= x +1
    print(classi,("-->%.2f"%(x/len(Y_predicted)*100)),'%')
    
    
def main():
    X_train ,X_test,Y_train,Y_actual = open_file_and_divide()
          
   # print(X_train)
    kernel = ""
    classify_and_predict(X_train,Y_train,X_test,Y_actual,kernel)
    plot(Y_predicted,Y_actual,"SVM")
    cm = np.zeros((5, 5), dtype=np.int)
    for i in range(len(Y_actual)):
       Y_actual_normal_list.append(Y_actual[i][0])
    print("")
    print("SVM:")    
    print("Actual Rating: ",Y_actual_normal_list)
   
    print("Predicted Rating:",Y_predicted)

    print("")
    print("Confusion Matrix --> SVM")
    print("")
    
    for i, j in zip(Y_actual, Y_predicted):
       cm[i[0]-1][j-1] += 1
    print(cm)

    plot_cm(cm,"SVM")
    print("\nAccuracy:")
    find_accuracy(Y_actual,Y_predicted,"SVM")
    Naive_bayes(X_train,Y_train,X_test,Y_actual)
    find_accuracy(Y_actual,Y_n_predicted,"Naive-Bayes")
    plot(Y_n_predicted,Y_actual,"naive")

    cm = np.zeros((5, 5), dtype=np.int)
    
    for i, j in zip(Y_actual, Y_n_predicted):
       cm[i[0]-1][j-1] += 1

    print("\nConfusion Matrix --> Naive-Bayes")
    print(cm)
    plot_cm(cm,"naive")

    

main()

