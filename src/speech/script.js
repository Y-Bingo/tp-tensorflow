import * as speechCommands from '@tensorflow-models/speech-commands';

const MODEL_PATH = 'http://127.0.0.1:8080/speech';
const resultEle = document.querySelector('#result');
window.onload = async () => {
	const recognizer = speechCommands.create(
		'BROWSER_FFT',
		null,
		MODEL_PATH + '/model.json', //
		MODEL_PATH + '/metadata.json', //
	);
	await recognizer.ensureModelLoaded();

	const labels = recognizer.wordLabels().slice(2);
	resultEle.innerHTML = labels
		.map(label => {
			return `<div>${label}</div>`;
		})
		.join('');
	recognizer.listen(
		result => {
			const { scores } = result;
			const maxValue = Math.max(...scores);
			const index = scores.indexOf(maxValue) - 2;
			resultEle.innerHTML = labels
				.map((label, i) => {
					return `<div style='background:${i === index && 'green'}'>${label}</div>`;
				})
				.join('');
		},
		{
			overlapFactor: 0.35,
			probabilityThreshold: 0.75,
		},
	);
};
