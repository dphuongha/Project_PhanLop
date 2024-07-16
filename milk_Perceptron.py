import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Đọc dữ liệu từ file CSV
df = pd.read_csv('./milknew.csv')
data = pd.DataFrame(df, columns=['pH', 'Temprature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour', 'Grade'])

# Chia dữ liệu thành tập huấn luyện và tập kiểm thử
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle=True)

# Tạo một bản sao của X_train và X_test để tránh ảnh hưởng đến dữ liệu gốc
X_train_encoded = dt_Train.drop(['Grade'], axis=1).copy()
X_test_encoded = dt_Test.drop(['Grade'], axis=1).copy()

# Áp dụng LabelEncoder cho từng cột trong X_train và X_test
label_encoder = preprocessing.LabelEncoder()

for col in X_train_encoded.columns:
    label_encoder.fit(pd.concat([X_train_encoded[col], X_test_encoded[col]]))
    X_train_encoded[col] = label_encoder.transform(X_train_encoded[col])
    X_test_encoded[col] = label_encoder.transform(X_test_encoded[col])

# Chuẩn bị đầu vào và đầu ra cho mô hình
y_train = dt_Train['Grade']
y_test = dt_Test['Grade']

#Sử dụng mô hình Perceptron
pla = Perceptron(
    penalty = 'l2', #tránh overfitting
    max_iter=1000, #số lượng lặp (epochs) cho quá trình huấn luyện
    eta0=0.1, #Tốc độ học của mô hình
    tol=0.001, #Quá trình huấn luyện sẽ dừng lại dựa trên sự thay đổi nhỏ của trọng số
    random_state=68 #Seed cho quá trình ngẫu nhiên
)
pla.fit(X_train_encoded, y_train)
y_pre = pla.predict(X_test_encoded)
count = 0
for i in range(0, len(y_pre)):
    if y_test.iloc[i] == y_pre[i]:
        count += 1

accuracy = count / len(y_pre)
#print('Tỷ lệ dự đoán đúng: ', accuracy)

# Đánh giá chất lượng của mô hình
accuracy = accuracy_score(y_test, y_pre)
precision = precision_score(y_test, y_pre, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pre, average='weighted')
f1 = f1_score(y_test, y_pre, average='weighted')

print('Tỷ lệ dự đoán đúng (Accuracy): ', accuracy)
print('Precision: ', precision)
print('Recall: ', recall)
print('F1 Score: ', f1)

