# Import Libararies and Data
import numpy as np
import pandas as pd
from sklearn import preprocessing
import seaborn as sns 
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Functions and classes
class Node:
    
    def __init__(self,dataset,depth,is_gini=True,purity_to_stop_split = 0.95):
        self.right = None
        self.left = None
        self.feature_index = None
        self.value = None
        self.depth_in_tree = depth
        self.dataset = dataset
        self.label = None
        self.is_gini = is_gini
        self.purity_to_stop_split = purity_to_stop_split
         
    def split_one_node(self,index,value):
        right = self.dataset[ self.dataset[:,index]<=value ]
        left = self.dataset[ self.dataset[:,index]>value]
        return left,right
    
    def get_features_to_check(self,num_columns):
        return range(num_columns-1)
    
    def get_best_split(self):
        num_rows = self.dataset.shape[0]
        num_columns = self.dataset.shape[1]
        best_score = 999
        features_to_search = self.get_features_to_check(num_columns)
        for feature_index in features_to_search:
            for row_num in range(num_rows):
                left,right = self.split_one_node(feature_index,self.dataset[row_num,feature_index])
                score = self.calcualte_metrics(left,right)
                if score < best_score:
                    best_score = score
                    best_feature = feature_index
                    best_value = self.dataset[row_num,feature_index]
                    best_right = right
                    best_left = left
        return best_feature,best_value,best_right,best_left  
     
    def calcualte_metrics(self,left_dataset,right_dataset,is_gini=True):
        metrics = 0
        total_size = len(left_dataset) + len(right_dataset)
        # total_size = left.shape[0] + right.shape[0]
        for branch in [left_dataset,right_dataset]:
            branch_size = len(branch)
            if branch_size==0:
                continue
            # arr_branch = np.array(branch)
            labels = branch[:,-1]
            vals,freq = np.unique(labels,return_counts=True)
            probabilities = freq/branch_size
            gini_branch = 1-(probabilities**2).sum()
            entropy_branch = (probabilities*np.log(probabilities)).sum()
            metrics_branch = gini_branch if is_gini else entropy_branch
            metrics += metrics_branch*branch_size/total_size
        return metrics
    
    def split_node_recursively(self,max_depth = 1):
        purity = self.calc_node_purity()
        if self.depth_in_tree > max_depth or purity > self.purity_to_stop_split:
            self.label = self.calc_majority_label()
            return
        else:
            self.feature_index,self.value,right_dataset,left_dataset = self.get_best_split()
            self.right = Node(right_dataset,self.depth_in_tree+1,self.is_gini)
            self.right.split_node_recursively(max_depth)
            self.left = Node(left_dataset,self.depth_in_tree+1,self.is_gini)
            self.left.split_node_recursively(max_depth)

             
    def calc_majority_label(self):
        labels = self.dataset[:,-1]
        vals,freq = np.unique(labels,return_counts=True)
        max_index = np.argmax(freq)
        return vals[max_index]
            
    def calc_node_purity(self):
        labels = self.dataset[:,-1]
        vals,freq = np.unique(labels,return_counts=True)        
        probabilities = freq/sum(freq)   
        purity = np.max(probabilities)
        return purity
    
    def predict(self,row):
       if self.label != None:
           return self.label
       else:
           if row[self.feature_index] > self.value:
               return self.left.predict(row)
           else:
               return self.right.predict(row)
           
    def print_tree(self):
        if self.label!=None:
            print('\t'*(self.depth_in_tree-1),'label is: ',self.label,'and depth is: ',self.depth_in_tree)
        else:
            print('\t'*(self.depth_in_tree-1),'split is at: ',self.feature_index,' ',self.value,' and depth is: ',self.depth_in_tree)
        if self.right:
            self.right.print_tree()
        if self.left:
            self.left.print_tree()
    
    def evaluate(self,x,y):
        predict_labels = [self.predict(row_data) for row_data in x]
        #accuracy = 100*(1-np.sum(abs(predict_labels - y)/len(y)))
        return predict_labels
    
def outliers(dataset,n,features):
    outlier = []
    for col in features:
        Q1 = np.percentile(dataset[col], 25)
        Q3 = np.percentile(dataset[col], 75)
        IQR = Q3 - Q1
        out_step = 1.5 * IQR
        out_list_col = dataset[(dataset[col] < Q1 - out_step) | (dataset[col] > Q3 + out_step )].index
        outlier.extend(out_list_col)
    outlier = Counter(outlier)        
    mult_outliers = list( k for k, v in outlier.items() if v > n )
    return mult_outliers

# Calcualte the error epsilon for each data point, "terror"
def calc_terror(predict,row,y):
    flag = np.zeros((1,len(y)))
    for i in range(len(y)):
        if predict[row,i] != y[i]: flag[0,i] = 1
    #print(np.sum(flag)/len(y))
    return flag

# Calcualte the total error for the whole dataset: error = sum(w(i) * terror(i)) / sum(w)
def calc_epsilon(w,flag):
    error = (w @ flag.T)
    return error

# Calculate the significance using the formula: significance = ln((1-error) / error)
def calc_alpha(error):
    alpha = np.log(1/error - 1)
    #print('alpha',alpha)
    return alpha

    
# Update w(i) the weigths of the points using: w = w * exp(stage * terror)
def update_w(w,tree,alpha,predict_tree,y):
    for i in range(len(y)):
        w[tree,i] *= np.exp(-1 * alpha * predict_tree[tree,i] * y[i])
    w[tree,:] /= w[tree,:].sum()
    #print(w)
    return w[tree,:]

# Build the weighted sum of the weak learners to produce one strong learner using the 
# significance as the model weights, and return the sign of the result
def predict_booster(predict_tree,alpha):
    booster_prediction = 1 * np.sign(alpha.T @ predict_tree)
    return booster_prediction
##########################################################################################
        
    
if __name__ == '__main__':
    dataset = pd.read_csv('heart.csv')
    dataset = dataset.fillna(np.nan)
# Seperate labels from dataset
    #print(dataset.isnull().sum())
    print(dataset.info())
# It seems that there is no missing data. All features are full

# Data Pre-processing
    dataset_len = len(dataset)
    #print(dataset.describe())
    male_rate = np.round(dataset['sex'].mean()*100,2)
    heart_disease_rate = np.round(dataset['target'].mean()*100,2) # Labels are more or less evenly distributed

# Define categirial features as ones and replace with 1 hot labels
    dataset['cp'] = dataset['cp'].astype('category')
    dataset['restecg'] = dataset['restecg'].astype('category')
    dataset['slope'] = dataset['slope'].astype('category')
    dataset['ca'] = dataset['ca'].astype('category')
    dataset['thal'] = dataset['thal'].astype('category')
    dataset = pd.get_dummies(dataset, columns = ['cp'],prefix = 'cp',drop_first = True)
    dataset = pd.get_dummies(dataset, columns = ['restecg'],prefix = 'r_ecg',drop_first = True)
    dataset = pd.get_dummies(dataset, columns = ['slope'],prefix = 'sl',drop_first = True)
    dataset = pd.get_dummies(dataset, columns = ['ca'],prefix = 'ca',drop_first = True)
    dataset = pd.get_dummies(dataset,columns=['thal'],prefix = 'th',drop_first = True)

# Seperate labels from dataset    
    norm_dataset = preprocessing.scale(dataset)
    Q = np.cov(norm_dataset.T,bias = True)
    sns.heatmap(Q, annot=False, fmt='g')
    plt.show()
    
    #b = sns.factorplot(y="chol",x="sex",data=dataset,kind="box")
    g1 = sns.distplot(dataset["age"], color="m", label="Skewness : %.2f"%(dataset["age"].skew()))
    g1 = g1.legend(loc="best")
    # Age pdf is relatively close to a gaussian (mean=54.37 yrs std=9.08 yrs)
    
    
# detect & drop outliers
    #print(dataset.columns)
    outliers_to_drop = outliers(dataset,1,dataset.columns)
    clean_dataset = dataset.drop(outliers_to_drop, axis = 0).reset_index(drop=True)
    
# Replace labels of '0' with '-1'   
    norm_dataset = pd.DataFrame(norm_dataset)
    Y = clean_dataset['target']
    Y = Y.replace(0,-1)
    clean_dataset.drop(labels = ['target'], axis = 1, inplace = True)
    
    x_train,x_test,y_train,y_test = train_test_split(clean_dataset,Y,train_size=0.8)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_train)
    y_test = np.array(y_train)
    
    train = np.column_stack((x_train,y_train))
    test = np.column_stack((x_test,y_test))
    
# Initialize the data points weight w(i) to be equal for all points   
    records = len(x_train)
    max_stages = 100
    max_trees = 3
    w = np.full((max_trees,records), 1/records)
    predict_tree = np.zeros((max_trees,w.shape[1]))
    predict = np.zeros((max_stages,w.shape[1]))
    hypothesis = 0
    learning_rate = 0.5
    
# Build a weak learner using a stump decision tree by calling a class Decision Tree
    stump = Node(train,0)
    stump.split_node_recursively()
    predict_tree[0,:] = stump.evaluate(x_train,y_train)
    
    log_reg = LogisticRegression(random_state=10)
    log_reg.fit(x_train,y_train)
    predict_tree[1,:] = log_reg.predict(x_train)
    
    svm_model = SVC(kernel = 'linear')
    svm_model.fit(x_train,y_train)
    predict_tree[2,:] = svm_model.predict(x_train)
    
    for stage in range(max_stages):
        predict_tree[0,:] = stump.evaluate(x_test,y_test)
        flag_wl1 = calc_terror(predict_tree,0,y_test)
        error_wl1 = calc_epsilon(w[0,:],flag_wl1)
        alpha_wl1 = calc_alpha(error_wl1)
        w[0,:] = learning_rate * update_w(w,0,alpha_wl1,predict_tree,y_test)
        #h1 = predict_forest(predict,max_stages,alpha_wl1,h1)
            
        predict_tree[1,:] = log_reg.predict(x_test)
        flag_wl2 = calc_terror(predict_tree,1,y_test)
        error_wl2 = calc_epsilon(w[1,:],flag_wl2)
        alpha_wl2 = calc_alpha(error_wl2)
        w[1,:] = learning_rate * update_w(w,1,alpha_wl2,predict_tree,y_test)
        #h2 = predict_forest(predict,max_stages,alpha_wl2,h2)
        
        predict_tree[2,:] = svm_model.predict(x_test)
        flag_wl3 = calc_terror(predict_tree,2,y_test)
        error_wl3 = calc_epsilon(w[2,:],flag_wl3)
        alpha_wl3 = calc_alpha(error_wl3)
        w[2,:] = learning_rate * update_w(w,2,alpha_wl3,predict_tree,y_test)
        #h3 = predict_forest(predict,max_stages,alpha_wl3,h3)
        alpha = np.array([alpha_wl1,alpha_wl2,alpha_wl3])
        booster_predict = predict_booster(predict_tree,alpha)
        #print('error = ',error,' alpha = ',alpha)
    
    print('Decision Tree Model accuracy is: ',
          np.round(100*np.sum(np.abs(predict_tree[0,:]+y_test))/(2*len(y_test)),2),'%')
    print('Logistic Reg. Model accuracy is: ',
          np.round(100*np.sum(np.abs(predict_tree[1,:]+y_test))/(2*len(y_test)),2),'%')
    print('SVM Model accuracy is: ',
          np.round(100*np.sum(np.abs(predict_tree[2,:]+y_test))/(2*len(y_test)),2),'%')

    print('Model accuracy is: ',
          np.round(100*np.sum(np.abs(booster_predict+y_test))/(2*len(y_test)),2),'%')
    
    
    
    
 
    
    
    

 

