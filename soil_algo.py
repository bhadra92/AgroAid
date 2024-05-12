# import os
# import numpy as np
import pandas as pd
from sklearn import tree, model_selection, metrics

def predict_result(rule_file,
                    val1,val2,val3,val4,val5,val6,val7,val8,val9,val10,
                    val11,val12,val13
                   ):

    data = pd.read_csv(rule_file,
                       names=['val1', 'val2', 'val3', 'val4', 'val5', 'val6', 'val7', 'val8', 'val9', 'val10',
                    'val11', 'val12', 'val13','class'])

    data['class'], class_names = pd.factorize(data['class'])
    #print("Class = {}".format(class_names))
    #print(data['class'].unique())


    ####################################################
    d_class = dict()
    i = 0
    for n in class_names:
        ll = data['class'].unique()[i]
        d_class[n] = ll
        i += 1
    #print(d_class)


    ######################################################
    # print(data.head())
    # print(data.info())

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # split data randomly into 70% training and 30% test
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=0)

    # train the decision tree
    dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=None)
    dtree.fit(X_train, y_train)

    tx = [
        val1,val2,val3,val4,val5,val6,val7,val8,val9,val10,
        val11, val12, val13
         ]

    y_pred = dtree.predict([tx])
    #print("{} -> \t {}".format(tx, class_names[y_pred[0]]))
    return class_names[y_pred[0]]

if __name__ == '__main__':
    rule_file= '../project/data/crop_dataset.csv'  #'../data/data_set.csv'
    #12,1,50000,domestic,yes
    #result = predict_result(rule_file,1,-1.35835406159823,-1.34016307473609,1.77320934263119,0.379779593034328,-0.503198133318193,1.80049938079263,0.791460956450422,0.247675786588991,-1.51465432260583,0.207642865216696,0.624501459424895,0.066083685268831,0.717292731410831,-0.165945922763554,2.34586494901581,-2.89008319444231,1.10996937869599,-0.121359313195888,-2.26185709530414,0.524979725224404,0.247998153469754,0.771679401917229,0.909412262347719,-0.689280956490685,-0.327641833735251,-0.139096571514147,-0.0553527940384261,-0.0597518405929204,378.66)#7519,1.23423504613468,3.0197404207034,-4.30459688479665,4.73279513041887,3.62420083055386,-1.35774566315358,1.71344498787235,-0.496358487073991,-1.28285782036322,-2.44746925511151,2.10134386504854,-4.6096283906446,1.46437762476188,-6.07933719308005,-0.339237372732577,2.58185095378146,6.73938438478335,3.04249317830411,-2.72185312222835,0.00906083639534526,-0.37906830709218,-0.704181032215427,-0.656804756348389,-1.63265295692929,1.48890144838237,0.566797273468934,-0.0100162234965625,0.146792734916988,1)
    # print(result)
'''
from .ml_algo import predict_result

    ############### ML PART ###############
        data_file_path = os.path.join(BASE_DIR, 'data/data_set.csv')
        result = predict_result(data_file_path,
                        float(age),float(sex),float(cp),float(trestbps),
                        float(chol),float(fbs),float(restecg),float(thalach),
                        float(exang),float(oldpeak),float(slope),float(ca),float(thal))

        print(result)

        ######################################
'''
