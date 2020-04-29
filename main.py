import numpy as np
from utils import initialize
from sklearn.externals import joblib

if __name__ == '__main__':
    # 1. Choose Test Ratio : 0.1 ~ 0.3
    # 2. Choose Model : LinearRegression / KnnRegression / DecisionTreeRegression / RandomForestRegression
    test_ratio = 0.2
    model_name = 'RandomForestRegression'

    # Load dataset and model
    test_data, train_data, model = initialize(test_ratio, model_name)
    train_x, train_y = train_data
    
    num_data, num_features = train_x.shape
    print('# of Training data : ', num_data)

    # TRAIN
    model.train(train_x, train_y)

    # Save model as pkl file
    joblib.dump(model, 'model.pkl')

    # EVALUATION
    test_x, test_y = test_data
    accuracy = model.test(test_x, test_y)
    print(model_name, "test file accuracy:", accuracy)

    # draw graph
    if model_name == 'LinearRegression' or model_name == 'KnnRegression':
        model.graph()
    else:
        model.graph(test_x,test_y)