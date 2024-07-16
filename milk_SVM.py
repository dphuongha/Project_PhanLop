import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

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

# Sử dụng SVM với kernel tuyến tính
svm_model = SVC(
    C=0.1, # Tham số kiểm soát độ chặt chẽ của việc phân loại sai.
    kernel='poly', # Sử dụng kernel đa thức để tạo đường phân chia dữ liệu.
    gamma='auto', # Tham số này xác định độ ảnh hưởng của mỗi điểm dữ liệu
    coef0 = 0.1 # Đây là hệ số độc lập trong hàm kernel, quan trọng đối với kernel đa thức và sigmoid
)
svm_model.fit(X_train_encoded, y_train)
y_pre_svm = svm_model.predict(X_test_encoded)

# Đánh giá chất lượng của mô hình SVM
accuracy_svm = accuracy_score(y_test, y_pre_svm)
precision_svm = precision_score(y_test, y_pre_svm, average='weighted',zero_division=1)
recall_svm = recall_score(y_test, y_pre_svm, average='weighted')
f1_svm = f1_score(y_test, y_pre_svm, average='weighted')

print('Tỷ lệ dự đoán đúng (Accuracy):', accuracy_svm)
print('Precision: ', precision_svm)
print('Recall: ', recall_svm)
print('F1 Score: ', f1_svm)



