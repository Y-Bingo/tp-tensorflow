import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

window.onload = async () => {
	/**
	 * 定义数据
	 */
	const heights = [150, 160, 170];
	const weights = [40, 50, 60];

	/**
	 * 数据可视化
	 */
	tfvis.render.scatterplot(
		{
			name: '身高体重训练数据',
		},
		{ values: heights.map((x, i) => ({ x, y: weights[i] })) },
		{ xAxisDomain: [140, 180], yAxisDomain: [30, 70] },
	);

	/**
	 * 归一化数据
	 */
	const inputs = tf.tensor(heights).sub(150).div(20);
	inputs.print();
	const labels = tf.tensor(weights).sub(40).div(20);
	labels.print();

	/**
	 * 构建神经网络
	 */
	// 初始化一个神经网络模型
	const model = tf.sequential();
	// 为神经网络模型添加层
	model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
	// 损失函数：均方误差（MSE） & 优化器： 随机梯度下降（SGD）
	model.compile({ loss: tf.losses.meanSquaredError, optimizer: tf.train.sgd(0.1) });

	/**
	 * 训练模型
	 */
	await model.fit(inputs, labels, {
		batchSize: 3,
		epochs: 100,
		callbacks: tfvis.show.fitCallbacks({ name: '训练过程' }, ['loss']),
	});

	/**
	 * 预测, 反归一化
	 */
	const predictData = [180];
	const output = model.predict(tf.tensor(predictData)).sub(150).div(20);
	document.querySelector('#result').innerHTML = `如果 x = 180 ， y = ${output.mul(20).add(40).dataSync()[0]}`;
};
