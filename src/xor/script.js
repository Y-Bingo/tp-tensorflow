import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getData } from './data';

// console.log(tf, tfvis);
window.onload = async () => {
	const data = getData(400);
	// console.log(data);

	tfvis.render.scatterplot(
		{ name: 'XOR 训练数据' },
		{
			values: [
				// 数据集 类型 1
				data.filter(p => p.label == 1),
				// 数据集 类型 2
				data.filter(p => p.label === 0),
			],
		},
	);

	// 初始化一个神经网络模型
	const model = tf.sequential();
	// 为神经网络模型添加层
	model.add(
		tf.layers.dense({
			units: 4,
			inputShape: [2],
			activation: 'relu',
		}),
	);
	model.add(
		tf.layers.dense({
			units: 1,
			activation: 'sigmoid',
		}),
	);
	model.compile({
		loss: tf.losses.logLoss,
		optimizer: tf.train.adam(0.1),
	});

	/**
	 * 训练数据
	 */
	const inputs = tf.tensor(data.map(p => [p.x, p.y]));
	const labels = tf.tensor(data.map(p => p.label));
	await model.fit(inputs, labels, {
		epochs: 10,
		callbacks: tfvis.show.fitCallbacks({ name: 'xor 训练效果' }, ['loss']),
	});

	/**
	 * 预测
	 */
	window.predict = async form => {
		const pred = await model.predict(tf.tensor([[form.x.value * 1, form.y.value * 1]]));
		document.querySelector('#result').children[0].textContent = `预测结果： ${pred.dataSync()[0]}`;
		console.log(`预测结果： ${pred.dataSync()[0]}`);
	};
};
