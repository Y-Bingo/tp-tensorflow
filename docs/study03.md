# 归一化

## 什么是归一化

-   把大数量级特征转化到较小数量级下，通常是 [ 0, 1 ]或 [ -1, 1 ]
-   例子： 身高体重预测、房价预测

## 为什么要归一化

-   绝大多数 Tensorflow.js 的模型都不是给特别大的数设计的
-   将不同的数量级的特征转换到统一数量级，防止某个特征影响过大

## 归一化训练

### 操作步骤

-
-   准备身高体重训练数据
-   使用 tfvis 可视化训练过程
-   使用 tensorflow 的 API 对数据进行归一化

## 训练、预测、 反归一化

### 操作步骤

-   定义一个神经网络模型
-   将归一化后的数据喂给模型学习 —— 训练模型
-   预测后，把结果反归一化为正常数据
