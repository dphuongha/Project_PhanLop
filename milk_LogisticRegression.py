import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

# Sử dụng Logistic Regression 
log_reg = LogisticRegression(
    penalty = 'l2', # Loại hình phạt
    C = 1, # Nghịch đảo của độ mạnh của phạt
    tol=0.001,
    solver = 'sag', # Thuật toán giải quyết bài toán tối ưu hóa
    max_iter = 1000, # Số lượng lớn nhất của các lần lặp để giải quyết bài toán tối ưu hóa
    random_state = 0, # Seed cho việc phát sinh số ngẫu nhiên
    multi_class='multinomial'
)
log_reg.fit(X_train_encoded, y_train)
y_pre = log_reg.predict(X_test_encoded)

# Đánh giá chất lượng của mô hình
accuracy = accuracy_score(y_test, y_pre)
precision = precision_score(y_test, y_pre, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pre, average='weighted')
f1 = f1_score(y_test, y_pre, average='weighted')

print('Tỷ lệ dự đoán đúng (Accuracy): ', accuracy)
print('Precision: ', precision)
print('Recall: ', recall)
print('F1 Score: ', f1)
