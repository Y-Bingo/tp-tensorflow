import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getIrisData, IRIS_CLASSES } from './data';

window.onload = async () => {
	const [xTrain, yTrain, xTest, yTest] = getIrisData(0.15);

	// 模型结构定义
	const model = tf.sequential();
	model.add(
		tf.layers.dense({
			units: 10,
			inputShape: [xTrain.shape[1]],
			activation: 'sigmoid',
		}),
	);
	model.add(
		tf.layers.dense({
			units: 3,
			activation: 'softmax',
		}),
	);

	// 训练模型
	model.compile({
		loss: 'categoricalCrossentropy', // 交叉熵损失函数
		optimizer: tf.train.adam(0.1),
		metrics: ['accuracy'],
	});

	await model.fit(xTrain, yTrain, {
		epochs: 100,
		validationData: [xTest, yTest],
		callbacks: tfvis.show.fitCallbacks({ name: '训练效果' }, ['loss', 'val_loss', 'acc', 'val_acc'], { callbacks: ['onEpochEnd'] }),
	});

	// 预测
	window.predict = form => {
		const input = tf.tensor([[form.a.value * 1, form.b.value * 1, form.c.value * 1, form.d.value * 1]]);
		const pred = model.predict(input);
		document.querySelector('#result').children[0].textContent = `预测结果：${IRIS_CLASSES[pred.argMax(1).dataSync(0)]}`;
		console.log(`预测结果：${IRIS_CLASSES[pred.argMax(1).dataSync(0)]}`);
		// alert(`预测结果：${IRIS_CLASSES[pred.argMax(1).dataSync(0)]}`);
	};
};
