/**
*	Split an array into buckets with n-members
*/
function chunk(arr, n) {
	return arr.slice(0,(arr.length+n-1)/n|0).
           map(function(c,i) { return arr.slice(n*i,n*i+n); });
}

/**
*	Normalize tensor to only include values from 0 to 1
*/
function normalizeTensor(tensor) {
	const max = tensor.max();
  const min = tensor.min();
  return tensor.sub(min).div(max.sub(min));
}

/**
*	Translate species name into one-hot encoding
*/
function speciesOneHotEncoding(species) {
	return [
		species === 'setosa' ? 1 : 0,
    species === 'virginica' ? 1 : 0,
    species === 'versicolor' ? 1 : 0,
	];
}

/**
*	Translate predicted one-hot array into a species name
*/
function decodeOneHotEncoding(arr) {
	// convert all the preditions to integers
	const intArr = arr.map(Math.round);

	//	workout which encode is turned on?
	//	for now, if there are more specieis with probability more than 0.5
	//	use the last one
	let specie = 'unknown';
	if (intArr[0] === 1) {
		specie = 'setosa';
	}
	if (intArr[1] === 1) {
		specie = 'virginica';
	}
	if (intArr[2] === 1) {
		specie = 'versicolor';
	}
	
	return specie;
}

/**
*	Split shuffled data into a training and testing datasets
*/
function splitData(data) {
	tf.util.shuffle(data);

	const testCasesNum = 10;
	const train = data.slice(0, data.length - testCasesNum);
	const test = data.slice(data.length - testCasesNum);
	
	return [train, test];
}

/**
*	Extract features and labels from the dataset and convert them into a tensor
*/
function prepareData(data) {
	console.log('Preparing data');

	const xs = data.map((d) => [d.sepalLength, d.sepalWidth,
		d.petalLength, d.petalWidth ]);
	const ys = data.map((d) => speciesOneHotEncoding(d.species));

	// Wrapping these calculations in a tidy will dispose any 
	// intermediate tensors.
	return tf.tidy(() => {
		let xsTensor = tf.tensor2d(xs, [xs.length, 4]);
		xsTensor = normalizeTensor(xsTensor);

		let ysTensor = tf.tensor2d(ys, [ys.length, 3]);
		ysTensor = normalizeTensor(ysTensor);

		return { xsTensor, ysTensor };
	});
}

/**
*	Define architecture of a neural network using the simplest 
* possible model configuration
*/
function createModel() {
	console.log('Creating model');

	//	Create a sequential model
	const model = tf.sequential();

	//	Add a single hidden layer, we're using 4 features (all numbers between 0-1)
	//	to predict the species so input shape needs to be
	model.add(tf.layers.dense({ inputShape: [4], units: 8 }));

	//	Add an output layer, output is one hot encoding array with
	//	3 items, so need 3 units
	model.add(tf.layers.dense({ units: 3 }));

	return model;
}	

/**
*	Train model is batches using training data with default paramaters
* for the model configuration (loss, metrics etc)
*/
async function trainModel(model, data) {
	console.log('Training model');
		
	const { xsTensor, ysTensor } = data;

	// Prepare the model for training
	model.compile({
		optimizer: tf.train.adam(),
		loss: tf.losses.meanSquaredError,
		metrics: ['mse'],
	});

	const batchSize = 30;
	const epochs = 100;

	return await model.fit(xsTensor, ysTensor, {
		batchSize,
		epochs,
		shuffle: true,
		callbacks: tfvis.show.fitCallbacks(
			{	name: 'Training Performance' },
			['loss', 'mse', 'accuracy'],
			{ height: 200, callbacks: ['onEpochEnd']
		})
	});
}

/**
*	Use trained model to make prediction on a testing data
* stored in a tensor.
*/
async function predict(model, testTensor) {
	console.log('Predicting');
	const pred = model.predict(testTensor);

	//	get typed array from prediction tensor and convert it
	//	to untyped
  const predArr = Array.from(pred.dataSync());
  pred.print();
  
  //	slice it 3 
  const predictions = chunk(predArr, 3);
 	return predictions.map(decodeOneHotEncoding);
}

/**
*		Main function
*/
async function run() {
	console.log('Running script');
	//	Step 1. - split data into training and testing
	const [train, test] = splitData(irisData);
	console.log(train, test);

	//	Step 2. - load and pre-process the data
	const trainData = prepareData(train);
	const testData = prepareData(test);

	//	Step 2. - create the model and train it
	const model = createModel();
	await trainModel(model, trainData);

	//	Step 3. - use the model for predictions
	const { xsTensor:testTensor } = testData;
	const predictions = await predict(model, testTensor);

	// Step 4. - compare predicitons with train data
	predictions.forEach((p, i) => {
		console.log(`For ${test[i].species}, prediction ${p}.`);
	})
}


document.addEventListener('DOMContentLoaded', run);
