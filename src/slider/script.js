import * as speechCommands from '@tensorflow-models/speech-commands';

const MODEL_PATH = 'http://127.0.0.1:8080';
let transferRecognizer; // 迁移学习器
let currentIndex = 0;
window.onload = async () => {
	const recognizer = speechCommands.create(
		'BROWSER_FFT',
		null,
		MODEL_PATH + '/speech/model.json', //
		MODEL_PATH + '/speech/metadata.json', //
	);
	// 加载模型
	await recognizer.ensureModelLoaded();
	transferRecognizer = recognizer.createTransfer('轮播图');
	// 加载声音训练数据
	const res = await fetch(MODEL_PATH + '/slider/data.bin');
	const arrayBuffer = await res.arrayBuffer();
	transferRecognizer.loadExamples(arrayBuffer);
	await transferRecognizer.train({ epochs: 30 });
	console.log('done');
};

window.toggle = async checked => {
	if (checked) {
		await transferRecognizer.listen(
			result => {
				const { scores } = result;
				const labels = transferRecognizer.wordLabels();
				const index = scores.indexOf(Math.max(...scores));
				play(labels[index]);
			},
			{ overlapFactor: 0, probabilityThreshold: 0.7 },
		);
	} else {
		transferRecognizer.stopListening();
	}
};

window.play = label => {
	const div = document.querySelector('.slider>div');
	if (label === '上一张') {
		currentIndex = Math.max(0, currentIndex - 1);
	} else {
		currentIndex = Math.min(5, currentIndex + 1);
	}
	div.style.transition = 'transform 1s';
	div.style.transform = `translateX(-${100 * currentIndex}%)`;
	console.log('当前：', currentIndex);
};
