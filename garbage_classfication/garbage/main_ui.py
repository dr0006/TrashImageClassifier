# -*- coding: utf-8 -*-
"""
@File  : main_ui.py
@author: FxDr
@Time  : 2023/11/12 23:48
@Description:
"""
import sys
import warnings

import concurrent.futures

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QStackedWidget, \
    QScrollArea

from garbage_classfication.garbage.model import get_default_device, to_device, ResNet, CustomResNet, GoogleNet
from garbage_classfication.garbage.tools.predict_img import predict_img

warnings.filterwarnings('ignore')


class GarbageInfoPage(QWidget):
    def __init__(self, main_page, class_index):
        super().__init__()

        self.main_page = main_page
        self.class_index = class_index
        self.stacked_widget = QStackedWidget(self)
        self.class_buttons = []

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('垃圾分类知识')
        self.setGeometry(100, 100, 600, 400)

        # 创建按钮组，每个按钮对应一个垃圾分类类别
        button_layout = QVBoxLayout()
        for idx, garbage_class in enumerate(self.class_index):
            button = QPushButton(garbage_class, self)
            button.clicked.connect(lambda state, idx=idx: self.show_page(idx))
            button_layout.addWidget(button)
            self.class_buttons.append(button)

            info_page = GarbageInfoSubPage(garbage_class)
            self.stacked_widget.addWidget(info_page)

        back_button = QPushButton('返回', self)
        back_button.clicked.connect(self.go_back)

        layout = QVBoxLayout(self)
        layout.addLayout(button_layout)
        layout.addWidget(self.stacked_widget)
        layout.addWidget(back_button)

    def show_page(self, idx):
        self.stacked_widget.setCurrentIndex(idx)

    def go_back(self):
        self.hide()  # 隐藏当前子页面
        self.main_page.show()  # 显示主页面


class GarbageInfoSubPage(QWidget):
    def __init__(self, garbage_class):
        super().__init__()

        self.garbage_class = garbage_class
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        label = QLabel(f"{self.garbage_class}的垃圾分类知识说明：\n", self)
        label.setAlignment(Qt.AlignLeft)
        layout.addWidget(label)

        if self.garbage_class == '纸板' or self.garbage_class == '纸':
            info_text = "这类垃圾主要是废弃的纸质制品，例如纸箱、报纸、纸袋等。大部分纸张是可回收的，因此请将这些垃圾放入可回收物桶中。"
        elif self.garbage_class == '玻璃':
            info_text = "玻璃垃圾包括玻璃制品，如玻璃瓶、玻璃杯等。玻璃是可回收的，应投放到可回收物桶中。注意避免投放易破碎的玻璃制品。"
        elif self.garbage_class == '金属':
            info_text = "金属垃圾包括铝罐、铁罐等金属制品。这些物品是可回收的，应投放到可回收物桶中。请注意清空残余物，以保证回收效果。"
        elif self.garbage_class == '塑料':
            info_text = "塑料垃圾包括塑料瓶、塑料袋等。大多数塑料是可回收的，应投放到可回收物桶中。请注意分类投放，避免混入其他垃圾。"
        elif self.garbage_class == '其他垃圾':
            info_text = "其他垃圾包括一些无法归类到以上类别的废弃物。这些垃圾通常不可回收，应投放到普通垃圾桶中。"
        else:
            info_text = "暂无详细说明。"

        info_label = QLabel(info_text, self)
        info_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(info_label)


class ImagePredictorApp(QWidget):
    def __init__(self):
        super().__init__()

        self.predict_label = None
        self.image_label = None
        self.scroll_area = None
        self.scroll_widget = None
        self.scroll_layout = None
        self.device = get_default_device()

        self.model1 = to_device(ResNet(), self.device)
        self.model2 = to_device(CustomResNet(), self.device)
        self.model3 = to_device(GoogleNet(), self.device)

        self.model1.load_model_dict('../model/resnet50/96.50%_model_weights.pth')
        self.model2.load_model_dict('../model/89.67%_model_weights.pth')
        self.model3.load_model_dict('../model/googleNet/92.50%_model_weights.pth')

        self.predictions_history = []

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('垃圾分类预测')
        self.setGeometry(100, 100, 600, 400)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.predict_label = QLabel(self)
        self.predict_label.setAlignment(Qt.AlignCenter)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)

        self.scroll_widget = QWidget(self.scroll_area)
        self.scroll_area.setWidget(self.scroll_widget)

        self.scroll_layout = QVBoxLayout(self.scroll_widget)

        select_button = QPushButton('选择图片', self)
        select_button.clicked.connect(self.select_image)

        batch_predict_button = QPushButton('批量预测', self)
        batch_predict_button.clicked.connect(self.batch_predict_images)

        info_button = QPushButton('垃圾分类知识', self)
        info_button.clicked.connect(self.show_garbage_info)

        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)
        layout.addWidget(select_button)
        layout.addWidget(batch_predict_button)
        layout.addWidget(self.predict_label)
        layout.addWidget(self.scroll_area)
        layout.addWidget(info_button)

    def select_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, '选择图片',
                                                   r'X:\Coding\Github\PyTorch-ImageClassifier\test_images\garbage',
                                                   'Images (*.png *.xpm *.jpg *.jpeg)')

        if file_path:
            self.show_image(file_path)

            label1, prob1 = predict_img(file_path, self.model1)
            label2, prob2 = predict_img(file_path, self.model2)
            label3, prob3 = predict_img(file_path, self.model3)

            prediction_result = (
                f"Image: {file_path}\n"
                f"model1: 预测类别: {label1} (Prob: {prob1:.3f})\n"
                f"model2: 预测类别: {label2} (Prob: {prob2:.3f})\n"
                f"model3: 预测类别: {label3} (Prob: {prob3:.3f})"
            )

            self.predictions_history.append(prediction_result)
            self.predict_label.setText(prediction_result)
            self.update_scroll_area()

    def show_image(self, path):
        pixmap = QPixmap(path)
        pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    def batch_predict_images(self):
        file_dialog = QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(self, '选择图片文件夹',
                                                     r'X:\Coding\Github\PyTorch-ImageClassifier\test_images\garbage',
                                                     'Images (*.png *.xpm *.jpg *.jpeg)')

        if file_paths:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # 使用多线程进行批量预测
                predictions = list(executor.map(self.predict_single_image, file_paths))

            # 将结果显示在界面上
            self.predictions_history.extend(predictions)
            self.update_scroll_area()

    def predict_single_image(self, file_path):
        label1, prob1 = predict_img(file_path, self.model1)
        label2, prob2 = predict_img(file_path, self.model2)
        label3, prob3 = predict_img(file_path, self.model3)

        return (
            f"Image: {file_path}\n"
            f"model1: 预测类别: {label1} (Prob: {prob1:.3f})\n"
            f"model2: 预测类别: {label2} (Prob: {prob2:.3f})\n"
            f"model3: 预测类别: {label3} (Prob: {prob3:.3f})"
        )

    def show_garbage_info(self):
        garbage_info_page.stacked_widget.setCurrentIndex(0)  # 默认显示第一页
        garbage_info_page.show()  # 显示子页面
        self.hide()  # 隐藏主页面

    def display_predictions(self):
        # Clear previous predictions from the scroll layout
        for i in reversed(range(self.scroll_layout.count())):
            self.scroll_layout.itemAt(i).widget().setParent(None)

        # Add the result to the scroll layout
        for prediction in self.predictions_history:
            prediction_label = QLabel(prediction, self)
            prediction_label.setWordWrap(True)
            self.scroll_layout.addWidget(prediction_label)

    def update_scroll_area(self):
        self.display_predictions()
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main_page = ImagePredictorApp()
    class_index = ['纸板', '玻璃', '金属', '纸', '塑料', '其他垃圾']
    garbage_info_page = GarbageInfoPage(main_page, class_index)

    main_page.show()

    sys.exit(app.exec_())
