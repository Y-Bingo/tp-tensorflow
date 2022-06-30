// 加载图片
const loadImage = src => {
	return new Promise(resolve => {
		const img = new Image();
		img.src = src;
		img.crossOrigin = 'anonymous';
		img.width = 224;
		img.height = 224;
		img.onload = () => {
			resolve(img);
		};
	});
};

export const getInputs = async () => {
	const imgTasks = [];
	const labels = [];
	for (let i = 0; i < 30; i++) {
		['android', 'apple', 'windows'].forEach(label => {
			imgTasks.push(loadImage(`http://127.0.0.1:8080/brand/train/${label}-${i}.jpg`));
			labels.push([
				label === 'android' ? 1 : 0, // android
				label === 'apple' ? 1 : 0, // apple
				label === 'windows' ? 1 : 0, // windows
			]);
		});
	}
	const inputs = await Promise.all(imgTasks);
	return { inputs, labels };
};
