let net;

const webcamElement = document.getElementById("webcam");
const classifier = knnClassifier.create();
const _runner = new Runner('#runner');

async function app() {
  console.log("Loading mobilenet..");

  // Load the model.
  net = await mobilenet.load();
  console.log("Successfully loaded model");

  // Create an object from Tensorflow.js data API which could capture image
  // from the web camera as Tensor.
  const webcam = await tf.data.webcam(webcamElement);

  // Reads an image from the webcam and associates it with a specific class
  // index.
  const addExample = async classId => {
    // Capture an image from the web camera.
    const img = await webcam.capture();

    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier.
    const activation = net.infer(img, "conv_preds");

    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, classId);

    // Dispose the tensor to release the memory.
    img.dispose();
  };

  // When clicking a button, add an example for that class.
  document
    .getElementById("class-a")
    .addEventListener("click", () => addExample(0));
  document
    .getElementById("class-b")
    .addEventListener("click", () => addExample(1));
  document
    .getElementById("class-c")
    .addEventListener("click", () => addExample(2));

  let prev_states = "";
  while (true) {
    if (classifier.getNumClasses() > 0) {
      const img = await webcam.capture();

      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(img, "conv_preds");

      // Get the most likely class and confidences from the classifier module.
      const result = await classifier.predictClass(activation);

      const classes = ["Run", "Jump", "Duck"];
      document.getElementById("console").innerHTML = `
        <div class="row">
          <div class="col-3">
            <b>Prediction:</b>
          </div>
          <div class="col-9">
            ${classes[result.label]}
          </div>
        </div>
        <div class="row">
          <div class="col-3">
            <b>Probability:</b>
          </div>
          <div class="col-9">
            ${result.confidences[result.label]}
          </div>
        </div>
      `;

      if (prev_states != classes[result.label]) {
        if (_runner.isRunning() && prev_states == 'Jump') {
          _runner.tRex.endJump()
        } else if (prev_states == 'Duck') {
          _runner.tRex.speedDrop = false;
          _runner.tRex.setDuck(false);
        }
      }

      prev_states = classes[result.label];
      if (!_runner.crashed && _runner.started && classes[result.label] == 'Jump') {
        if (!_runner.activated) {
          _runner.loadSounds();
          _runner.activated = true
        }
        if (!_runner.tRex.jumping && !_runner.tRex.ducking) {
          _runner.playSound(_runner.soundFx.BUTTON_PRESS);
          _runner.tRex.startJump(_runner.currentSpeed)
        }
      }

      if (_runner.activated && !_runner.crashed && classes[result.label] == 'Duck') {
        if (_runner.tRex.jumping) {
          _runner.tRex.setSpeedDrop()
        } else if (!_runner.tRex.jumping && !_runner.tRex.ducking) {
          _runner.tRex.setDuck(true)
        }
      }

      document.getElementById("class-a").innerText = `Run : ${classifier.classExampleCount[0] || 0}`;
      document.getElementById("class-b").innerText = `Jump : ${classifier.classExampleCount[1] || 0}`;
      document.getElementById("class-c").innerText = `Duck : ${classifier.classExampleCount[2] || 0}`;

      // Dispose the tensor to release the memory.
      img.dispose();
    }

    await tf.nextFrame();
  }
}

function startRunnerGame() {
  if (!_runner.activated) {
    _runner.loadSounds();
    _runner.activated = true
    _runner.tRex.startJump(_runner.currentSpeed)
  }
}

app();
