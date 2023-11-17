# -*- coding: utf-8 -*-
"""
@File  : main_ui.py
@author: FxDr
@Time  : 2023/11/12 23:48
@Description:
"""
import sys
import warnings

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog

from garbage_classfication.garbage.model import get_default_device, to_device, ResNet
from garbage_classfication.garbage.tools.predict_img import predict_img

warnings.filterwarnings('ignore')


class ImagePredictorApp(QWidget):
    def __init__(self):
        super().__init__()

        self.predict_label = None
        self.image_label = None
        self.device = get_default_device()

        # 创建两个模型实例
        self.model1 = to_device(ResNet(), self.device)
        # self.model2 = to_device(ResNet(), self.device)
        # self.model3 = to_device(GoogleNet(), self.device)
        # self.model4 = to_device(GoogleNet(), self.device)

        # 为每个模型加载权重
        self.model1.load_model_dict('../model/resnet50/95.54%_model_weights.pth')
        # self.model2.load_model_dict('model/resnet50/95.54%_model_weights.pth')
        # self.model3.load_model_dict('model/googleNet/92.50%_model_weights.pth')
        # self.model4.load_model_dict('model/googleNet/91.50%_model_weights.pth')

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('垃圾分类预测')
        self.setGeometry(100, 100, 600, 400)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.predict_label = QLabel(self)
        self.predict_label.setAlignment(Qt.AlignCenter)

        select_button = QPushButton('选择图片', self)
        select_button.clicked.connect(self.select_image)

        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)
        layout.addWidget(select_button)
        layout.addWidget(self.predict_label)

    def select_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, '选择图片',
                                                   r'X:\Coding\Github\PyTorch-ImageClassifier\test_images',
                                                   'Images (*.png *.xpm *.jpg *.jpeg)')

        if file_path:
            self.show_image(file_path)

            # 使用两个模型进行预测
            label1, prob1 = predict_img(file_path, self.model1)
            # label2, prob2 = predict_img(file_path, self.model2)
            # label3, prob3 = predict_img(file_path, self.model3)
            # label4, prob4 = predict_img(file_path, self.model4)

            self.predict_label.setText(
                f"预测类别: {label1} (Prob: {prob1:.3f}))")

    def show_image(self, path):
        pixmap = QPixmap(path)
        pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImagePredictorApp()
    window.show()
    sys.exit(app.exec_())
