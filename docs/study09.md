# 使用预训练模型进行图片分类任务

## 什么是预训练模型

-   已经事先训练好的模型，无需训练即可预测
-   在 Tensorflow.js 中可以调用 web 格式的模型文件

## 操作步骤

-   加载 MobileNet 模型（一种卷积神经网络，轻量，准确度不高）
-   进行预测

## 加载 MobileNet 模型

### 操作步骤

-   从课程示例代码中下载 MobileNet 模型文件
-   在本地开启静态文件服务器
-   使用 Tensorflow 的 loadLayerModel 方法加载模型

## 进行预测

### 操作步骤

-   编写前端界面输入待预测的数据
-   使用训练好的模型进行预测
-   将输出的 Tensor 转为普通的数据并显示