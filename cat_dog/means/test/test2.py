# -*- coding: utf-8 -*-
"""
@File  : test2.py
@author: FxDr
@Time  : 2023/11/20 23:09
@Description:
"""
from cat_dog.means.predict_model import ImageClassifier

# 实例化自定义的ImageClassifier类
classifier = ImageClassifier(svm_model_path='../pth/svm_model.pth', pca_model_path='../pth/pca_model.pth')

# 调用predict_image方法进行预测
new_image_path = r"C:\Users\lenovo\Downloads\archive\cat和dog\test\cat\cat.1480.jpg"
predicted_class = classifier.predict_image(new_image_path)

print(f'新图像的预测类别是：{predicted_class}')
