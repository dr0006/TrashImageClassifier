# -*- coding: utf-8 -*-
"""
@File  : ui2.py
@author: FxDr
@Time  : 2023/11/20 23:14
@Description:
"""
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog

from cat_dog.means.predict_model import ImageClassifier

# 加载保存的SVM模型
svm_path = './pth/svm_model.pth'

# 使用保存的PCA模型
pca_path = './pth/pca_model.pth'


class ImagePredictionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.image_path = None
        self.image_label = QLabel()
        self.prediction_label = QLabel()

        self.init_ui()

        self.image_classifier = ImageClassifier(svm_model_path=svm_path, pca_model_path=pca_path)

    def init_ui(self):
        layout = QVBoxLayout()

        layout.addWidget(self.image_label)
        layout.addWidget(self.prediction_label)

        # 加载图片按钮
        load_button = QPushButton('加载图片')
        load_button.clicked.connect(self.load_image)
        layout.addWidget(load_button)

        # 预测图片按钮
        predict_button = QPushButton('预测图片')
        predict_button.clicked.connect(self.predict_image)
        layout.addWidget(predict_button)

        self.setLayout(layout)
        self.setWindowTitle('图像预测器')

    def load_image(self):
        file_dialog = QFileDialog()
        self.image_path, _ = file_dialog.getOpenFileName(self, '选择图像', '', 'Images (*.png *.jpg *.bmp)')
        if self.image_path:
            pixmap = QPixmap(self.image_path)
            self.image_label.setPixmap(pixmap)
            self.image_label.setAlignment(Qt.AlignCenter)

    def predict_image(self):
        if hasattr(self, 'image_classifier'):
            if self.image_path:
                # 进行图像预测
                predicted_class = self.image_classifier.predict_image(self.image_path)

                # 更新预测标签显示
                self.prediction_label.setText(f'预测标签: {predicted_class}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImagePredictionApp()
    window.show()
    sys.exit(app.exec_())
