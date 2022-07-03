import * as speechCommands from '@tensorflow-models/speech-commands';
import * as tfvis from '@tensorflow/tfjs-vis';

const MODEL_PATH = 'http://127.0.0.1:8080';
let transferRecognizer; // 迁移学习器
window.onload = async () => {
	const btns = document.getElementsByTagName('button');
	btns.disable = false;
	//  (btn => {
	// 	btn.disable = true;
	// });

	const recognizer = speechCommands.create(
		'BROWSER_FFT',
		null,
		MODEL_PATH + '/speech/model.json', //
		MODEL_PATH + '/speech/metadata.json', //
	);
	// 加载模型
	await recognizer.ensureModelLoaded();

	transferRecognizer = recognizer.createTransfer('轮播图');

	btns.disable = true;
};

window.collect = async btn => {
	btn.disable = true;
	const label = btn.innerHTML;

	if (transferRecognizer) {
		await transferRecognizer.collectExample(label === '背景噪音' ? '_background_noise_' : label);
		document.querySelector('#count').innerHTML = JSON.stringify(transferRecognizer.countExamples(), null, 2);
	}

	btn.disable = false;
};

window.train = async () => {
	await transferRecognizer.train({
		epochs: 30,
		callback: tfvis.show.fitCallbacks({ name: '训练效果' }, ['loss', 'acc'], { callbacks: ['onEpochEnd'] }),
	});
};

window.toggle = async checked => {
	if (checked) {
		await transferRecognizer.listen(
			result => {
				const { scores } = result;
				const labels = transferRecognizer.wordLabels();
				const index = scores.indexOf(Math.max(...scores));
				console.log(labels[index]);
			},
			{ overlapFactor: 0, probabilityThreshold: 0.9 },
		);
	} else {
		transferRecognizer.stopListening();
	}
};

window.save = () => {
	const arrayBuffer = transferRecognizer.serializeExamples();
	const blob = new Blob([arrayBuffer]);
	const link = document.createElement('a');
	link.href = window.URL.createObjectURL(blob);
	link.download = 'data.bin';
	link.click();
};
