import numpy as np
from utils import initialize
from sklearn.externals import joblib

if __name__ == '__main__':
    # Choose test ratio
    test_ratio = 0.2

    # Load dataset and model
    test_data, train_data, model = initialize(test_ratio)
    train_x, train_y = train_data
    
    num_data, num_features = train_x.shape
    print('# of Training data : ', num_data)

    # Make model
    model = model()

    # TRAIN
    model.train(train_x, train_y)

    # Save model as pkl file
    joblib.dump(model, 'LR_model.pkl')

    # EVALUATION
    test_x, test_y = test_data
    accuracy = model.test(test_x, test_y)
    print(model, "test file accuracy:", accuracy)