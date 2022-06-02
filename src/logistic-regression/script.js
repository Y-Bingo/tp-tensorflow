import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getData } from './data';

// console.log(tf, tfvis);
window.onload = async () => {
	const data = getData(400);
	// console.log(data);

	tfvis.render.scatterplot(
		{ name: '逻辑回归训练数据' },
		{
			// 多组数据，分为不同的点簇
			values: [data.filter(p => p.label === 1), data.filter(p => p.label === 0)],
		},
	);

	// 初始化一个神经网络模型
	const model = tf.sequential();
	// 为神经网络模型添加层
	model.add(
		tf.layers.dense({
			units: 1,
			inputShape: [2],
			activation: 'sigmoid',
		}),
	);
	// 损失函数：对数损失（logLoss）
	model.compile({ loss: tf.losses.logLoss, optimizer: tf.train.adam(0.1) });

	/**
	 * 训练数据
	 */
	const inputs = tf.tensor(data.map(p => [p.x, p.y]));
	const labels = tf.tensor(data.map(p => p.label));
	await model.fit(inputs, labels, {
		batchSize: 40,
		epochs: 50,
		callbacks: tfvis.show.fitCallbacks({ name: '训练数据' }, ['loss']),
	});

	/**
	 * 预测
	 */

	window.predict = form => {
		const pred = model.predict(tf.tensor([[form.x.value * 1, form.y.value * 1]]));
		document.querySelector('#result').children[0].textContent = `预测结果： ${pred.dataSync()[0]}`;
		console.log(`预测结果： ${pred.dataSync()[0]}`);
	};
};
