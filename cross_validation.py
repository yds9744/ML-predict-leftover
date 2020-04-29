from utils import initialize

# 1. Choose Test Ratio : 0.1 ~ 0.3
# 2. Choose Model : LinearRegression / KnnRegression / DecisionTreeRegression / RandomForestRegression
test_ratio = 0.2
models = ['LinearRegression', 'KnnRegression', 'DecisionTreeRegression', 'RandomForestRegression']

N = 10
max_acc = 0
max_model = ''

# Load dataset and model
for model_name in models:
    avg_acc = 0
    for i in range(N):
        test_data, train_data, model = initialize(test_ratio, model_name)
        train_x, train_y = train_data

        num_data, num_features = train_x.shape
        print('# of Training data : ', num_data)

        # TRAIN
        model.train(train_x, train_y)

        # EVALUATION
        test_x, test_y = test_data
        accuracy = model.test(test_x, test_y)
        avg_acc += accuracy
    avg_acc /= N
    print(model_name, "test file accuracy:", avg_acc)
    if avg_acc > max_acc:
        max_acc = avg_acc
        max_model = model_name

print('[final]', max_model, "is best model. accuracy: ", max_acc)