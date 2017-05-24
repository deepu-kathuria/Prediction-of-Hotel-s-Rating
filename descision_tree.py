import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
import seaborn as sns
from sklearn import tree


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
   
   plt.savefig('Decision_tree_confusion_Matrix.png', format='png')

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
    
    plt.savefig('Descision_tree.png', format='png')
    


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
    




def open_file_and_divide():
    n=0
    t=1

    data = pd.read_csv("MDATASET.csv")

    X = data[['P_Word_Count','N_Word_Count']].as_matrix(columns = None)
    Y = data[['Stars']].as_matrix(columns = None)

    X_train ,X_test,Y_train,Y_actual = train_test_split(X,Y,test_size = 0.5,random_state = 7)


    print("Number of rows in Training Set: ",len(X_train))
    print("Number of rows in Testing Set: ",len(X_test))

    return X_train ,X_test,Y_train,Y_actual

def classify_and_predict(X_train,Y_train,X_test,Y_actual,kernel1):

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, Y_train)



    for i in range(len(X_test)):
        t = clf.predict([X_test[i]])
        Y_predicted.append(t[0])
    #print(Y_predicted)

    

    
def find_accuracy(Y_actual,Y_predicted):
    x = 0
    """print(Y_predicted)
    print(Y_actual)"""
    for i in range(len(Y_predicted)):
        if(Y_predicted[i] == Y_actual[i]):
            x= x +1
    print(("-->%.2f"%(x/len(Y_predicted)*100)),'%')
    
    
def main():
    X_train ,X_test,Y_train,Y_actual = open_file_and_divide()
          
   # print(X_train)
    kernel = ""
    classify_and_predict(X_train,Y_train,X_test,Y_actual,kernel)
    plot(Y_predicted,Y_actual,"DT")
    cm = np.zeros((5, 5), dtype=np.int)
    for i in range(len(Y_actual)):
       Y_actual_normal_list.append(Y_actual[i][0])
    print("")
    print("Descision Tree:")    
    print("Actual Rating: ",Y_actual_normal_list)
   
    print("Predicted Rating:",Y_predicted)

    print("")
    print("Confusion Matrix --> Descision Tree")
    print("")
    
    for i, j in zip(Y_actual, Y_predicted):
       cm[i[0]-1][j-1] += 1
    print(cm)

    plot_cm(cm,"Descision Tree")
    print("\nAccuracy:")
    find_accuracy(Y_actual,Y_predicted)

    

    

main()

