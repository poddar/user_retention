#including all libraries needed for the analysis.
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# show plots in the notebook
#%matplotlib inline
import seaborn as sns
from ggplot import *
import math
import random
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, mean_absolute_error, log_loss
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn import tree

def main():

#this is a hack to get the data into csv format from the json format. The csv format will make it easier for pandas to read it.
    with open('uber_data_challenge.json') as data_file:
        data = json.load(data_file)
    with open("uber_data.csv", "w") as file:
        csv_file = csv.writer(file)
        for item in data:
            csv_file.writerow([item['city'], item['avg_dist'], item['signup_date'], item['avg_rating_of_driver'], item['avg_surge'], item['last_trip_date'], item['phone'], item['surge_pct'], item['uber_black_user'], item['weekday_pct'], item['trips_in_first_30_days'], item['avg_rating_by_driver']])

    df=pd.read_csv('uber_data.csv',header=None)
    df.head()
    df.columns = ['city', 'avg_dist','signup_date','avg_rating_of_driver','avg_surge','last_trip_date','phone','surge_pct','uber_black_user','weekday_pct','trips_in_first_30_days','avg_rating_by_driver']

    df.head()
    len(df.index)
    max(df.avg_dist)

#histogram of users trips in the first 30 days
    plt.hist(df.trips_in_first_30_days,bins=125)
    plt.title("Histogram of trips in first 30 days")
    plt.xlabel("Trips at user level")
    plt.ylabel("Frequency")
    plt.yscale('log')
    plt.savefig("hist_trips_first_30_days")
    plt.close()

#checking to see which data points are missing 
    df.isnull().sum()

#histogram of users avg. distance in the first 30 days
    plt.hist(df.surge_pct,bins=10)
    plt.title("Histogram of surge_pct")
    plt.xlabel("Surge_pct")
    plt.ylabel("Frequency")
    plt.savefig("hist_surge_pct")
    plt.close()
#histogram of users avg. distance in the first 30 days
    plt.hist(df.avg_dist,bins=170)
    plt.title("Histogram of avg. dist in first 30 days")
    plt.xlabel("Dist at user level")
    plt.ylabel("Frequency")
    plt.yscale('log')
    plt.savefig("hist_avg_dist_first_30_days")
    plt.close()

#histogram is the users mobile device operating system
    sns.barplot(x="phone", data=df);


    labels=['phone','uber_black_user','city']
    for i in labels:
        sns.barplot(x=i,data=df)

#making a copy of the dataset here and storing in df_2
    df_2=df

#removing nans from df_2 
    df_2 = df[np.isfinite(df['avg_rating_by_driver'])]
    df_2 = df_2[np.isfinite(df_2['avg_rating_of_driver'])]

#reindexing the new dataframe
    df_2=df_2.reset_index(drop=True)
    df_2.head(10)

    plt.hist(df_2.avg_rating_by_driver,bins=5)
    plt.title("Histogram of avg. rating by driver")
    plt.xlabel("Rating")
    plt.savefig("hist_avg_rating_by_driver")
    plt.close()

    plt.hist(df_2.avg_rating_of_driver,bins=5)
    plt.title("Histogram of avg. rating of driver")
    plt.xlabel("Rating")
    plt.savefig("hist_avg_rating_of_driver")
    plt.close()

    scatplot= ggplot(aes(x='avg_rating_by_driver', y='avg_rating_of_driver'), data=df_2) + geom_point()
    ggsave(plot=scatplot,file="scatter_plot_ratings.png")

#converting signup dates to unitime and weekday number in order to consider singup_date dependency on user retention in the model.
    df['signup_date']=pd.to_datetime(df['signup_date'])
    df['signup_unixtime']=0
    df['week_day']=0
    for i in range(0,49999):
        df['signup_unixtime'][i]=((df['signup_date'][i]).value)/(10**9)
    df['week_day']= (df['signup_date'].apply(lambda x: x.dayofweek))+1

#defining a new column name 'retained' which will be the dependent variable in the learning algorithm. 
    df['retained']=0
    users_not_retained=0.0
    users_retained=0.0
    for i in range(0,49999):
        if df['trips_in_first_30_days'][i] == 0:
            df['retained'][i]=0
            users_not_retained+=1
        if df['trips_in_first_30_days'][i] > 0:
            df['retained'][i]=1
            users_retained+=1    

#percentage of users retained
    percentage_of_users_retained=users_retained*100/(users_retained+users_not_retained)
    print("% of users retained: ", percentage_of_users_retained)

#filling the missing values for driver rating and user rating. Drawing randomly from the distribution
#of known values for each variable. Note that this value can also be predicted using linear regression in the case of 
#the driver ratings and logistic regression in case of device (Android and iOS). In this case I choose to draw randomly
#from the distribution based on the histograms results shown earlier since they are heavily skewed in the case of high
#ratings.
    for i in range(0,49999):
        if math.isnan(df['avg_rating_of_driver'][i]):
            df['avg_rating_of_driver'][i]=df_2['avg_rating_of_driver'][random.randint(0, 41749)]
        if math.isnan(df['avg_rating_by_driver'][i]):
            df['avg_rating_by_driver'][i]=df_2['avg_rating_by_driver'][random.randint(0, 41749)]

#now the dataset is complete, I choose to use logistic regression to predict the probability of a user bring retained.
# create dataframes with an intercept column and dummy variables for
# all categorical variables.
    y, X = dmatrices('retained ~ C(city) + C(phone)+C(uber_black_user)+ avg_rating_of_driver + avg_rating_by_driver + avg_dist + avg_surge + weekday_pct+(weekday_pct*avg_surge) + (avg_rating_of_driver*avg_dist)+ C(week_day)',
                      df, return_type="dataframe")
    print X.columns

    X.head()
    y.head()

# flatten y into a 1-D array for scikit learn
    y = np.ravel(y)

#splitting the data into test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#initiating a logistic regression model here     
    clf_lr = LogisticRegression(penalty='l2', dual=False, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)
    model_lr=clf_lr.fit(X_train, y_train)
    print("**************LogisticRegression Results with L2 regularisation************")
    print("Model Score on training set: ", model_lr.score(X_train,y_train))
    print("Model Score on test set:     ", model_lr.score(X_test,y_test))
    y_pred_prob = model_lr.predict_proba(X_test)[:,1]
    y_pred_prob= np.ravel(y_pred_prob)
    print "roc_auc_Score: ", roc_auc_score(y_test,y_pred_prob)
    print(pd.DataFrame(zip(X.columns, np.transpose(model_lr.coef_))))

# The mean square error
    print("Residual sum of squares: %.2f"
          % np.mean((model_lr.predict(X_test) - y_test) ** 2))

    yHat=model_lr.predict(X_test)
    y_train_mean=np.mean(y_train)
    y_test_mean=np.mean(y_test)

    print("Mean of the y_test: %.2f" % np.mean(y_test))
    print("Mean of the y_pred: %.2f" % np.mean(y_pred_prob))

    entropy = -(y_train_mean*math.log(y_train_mean)+(1-y_train_mean)*math.log(1-y_train_mean))
    log_loss_preds = np.zeros(shape=(len(y_test),2))
    for i in range(0,len(y_test)-1):
        log_loss_preds[i] = [1-y_train_mean,y_train_mean]
    cross_entropy   = -log_loss(y_test.astype(int).ravel(), log_loss_preds)
    rig_test        = 1 + cross_entropy/entropy

    print "Cross-entropy: ", cross_entropy
    print "This is the RIG for the GTA (should be 0 or close to 0): ", rig_test

#roc_auc curve
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_prob)
    auc = metrics.auc(fpr,tpr)

    df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
    roc_plot=ggplot(df, aes(x='fpr', y='tpr')) + geom_line() + geom_abline(linetype='dashed')   
    ggsave(plot=roc_plot,file="roc_plot.png") 

#area under the ROC curve
    auc_roc_plot=ggplot(df, aes(x='fpr', ymin=0, ymax='tpr')) +geom_area(alpha=0.2) + geom_line(aes(y='tpr')) + ggtitle("ROC Curve w/ AUC=%s" % str(auc))
    ggsave(plot=auc_roc_plot,file="auc_roc_plot.png")

#trying a random forest classifier to benchmark the logistic regression performance.
    clf_rf = RandomForestClassifier(n_estimators=100)
    model_rf=clf_rf.fit(X_train, y_train)
    output = model_rf.predict(X_test)
    # The mean square error
    print("**************RandomForest Results with 100 estimators************")
    print("Residual sum of squares: %.2f"
          % np.mean((model_rf.predict(X_test) - y_test) ** 2))
    # Explained Model score: 1 is perfect prediction     
    print('Model score: %.2f' % model_rf.score(X_test, y_test))
    print('mean value for the y_pred: %.2f' % np.mean(model_rf.predict(X_test)))

#cross-validation method presented here using k-folds.
    cfr = RandomForestClassifier(n_estimators=100)
#Simple K-Fold cross validation. 5 folds.
    cv = cross_validation.KFold(len(X), n_folds=5, indices=False)

#iterate through the training and test cross validation segments and
#run the classifier on each one, aggregating the results into a list
    results = []
    for traincv, testcv in cv:
        probas = cfr.fit(X[traincv], y[traincv]).predict_proba(X[testcv])
        results.append( log_loss(y[testcv], [x[1] for x in probas]) )

#print out the mean of the cross-validated results
    print("**************RandomForest Results with 100 estimators and cross validation************")
    print "Results: " + str( np.array(results).mean() )

    clf_lr_sgd = SGDClassifier(loss='log',penalty='l2',fit_intercept=True, class_weight=None, random_state=None)
    model_lr_sgd=clf_lr_sgd.fit(X_train, y_train)
    print("**************SGDClassifier with L2 regularisation************")
    print "Model Score on training set: ", model_lr_sgd.score(X_train,y_train)
    print "Model Score on test set:     ", model_lr_sgd.score(X_test,y_test)
    print(pd.DataFrame(zip(X.columns, np.transpose(model_lr_sgd.coef_))))

#decision tree classifier
    clf_tree= tree.DecisionTreeClassifier()
    model_tree =clf_tree.fit(X_train,y_train)
    print("**************Decision Tree results************")
    print "Model Score on training set: ", model_tree.score(X_train,y_train)
    print "Model Score on test set:     ", model_tree.score(X_test,y_test)

#what we see in these results is the Decision Tree heavily overfitting in the train set and performing badly
#on the test set.

if __name__ == "__main__":
    main()