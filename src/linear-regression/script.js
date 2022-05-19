import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

window.onload = async () => {
	// 准备、可视化训练数据
	const xs = [1, 2, 3, 4];
	const ys = [1, 3, 5, 7];

	// 构建神经网络
	tfvis.render.scatterplot(
		{
			name: '线性回归样本',
		},
		{ values: xs.map((x, i) => ({ x, y: ys[i] })) },
		{ xAxisDomain: [0, 5], yAxisDomain: [0, 8] },
	);

	// 初始化一个神经网络模型
	const model = tf.sequential();
	// 为神经网络模型添加层
	model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
	// 损失函数：均方误差（MSE） & 优化器： 随机梯度下降（SGD）
	model.compile({ loss: tf.losses.meanSquaredError, optimizer: tf.train.sgd(0.1) });

	/**
	 * 训练数据
	 */
	const inputs = tf.tensor(xs);
	const labels = tf.tensor(ys);
	await model.fit(inputs, labels, {
		batchSize: 2,
		epochs: 100,
		callbacks: tfvis.show.fitCallbacks({ name: '训练过程' }, ['loss']),
	});

	/**
	 * 预测
	 */
	const output = model.predict(tf.tensor([5]));
	output.print();
	console.log(`如果 x = 5 ， y = ${output.dataSync()[0]}`);
};
