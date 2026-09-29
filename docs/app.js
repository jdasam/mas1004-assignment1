// The whole demo. It loads the model your export script wrote, and runs it
// right here in the browser. There is no server and nothing is uploaded.
//
// You do not need to change this file. Read it if you are curious: the
// forward() function below is the same matrix multiply you did by hand in
// class, written out in JavaScript.

let model = null;      // the contents of model.json
let weights = null;    // every number in the model, as one long Float32Array

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// The model itself
// ---------------------------------------------------------------------------

function forward(input) {
  let values = input;
  for (const layer of model.layers) {
    if (layer.type === "linear") {
      const weightStart = layer.w[0];
      const biasStart = layer.b[0];
      const output = new Float32Array(layer.out);
      for (let o = 0; o < layer.out; o++) {
        let sum = weights[biasStart + o];
        const rowStart = weightStart + o * layer.in;
        for (let i = 0; i < layer.in; i++) {
          sum += weights[rowStart + i] * values[i];
        }
        output[o] = sum;
      }
      values = output;
    } else if (layer.type === "relu") {
      const output = new Float32Array(values.length);
      for (let i = 0; i < values.length; i++) {
        output[i] = values[i] > 0 ? values[i] : 0;
      }
      values = output;
    }
  }
  return values;
}

function softmax(logits) {
  const biggest = Math.max(...logits);
  const exponentials = Array.from(logits, (v) => Math.exp(v - biggest));
  const total = exponentials.reduce((a, b) => a + b, 0);
  return exponentials.map((v) => v / total);
}

// ---------------------------------------------------------------------------
// Preparing an image, the same way Python prepared the training images
// ---------------------------------------------------------------------------

const scratch = document.createElement("canvas");

function shrinkOnto(source, size) {
  // Going straight from a 640x480 camera frame to 32x32 in one step throws
  // away almost every pixel. Halving repeatedly keeps much more, and lands
  // closer to what Pillow does in Python.
  let width = source.videoWidth || source.naturalWidth || source.width;
  let height = source.videoHeight || source.naturalHeight || source.height;

  let stage = document.createElement("canvas");
  stage.width = width;
  stage.height = height;
  stage.getContext("2d").drawImage(source, 0, 0, width, height);

  while (width > size * 2 && height > size * 2) {
    const next = document.createElement("canvas");
    next.width = Math.max(size, Math.floor(width / 2));
    next.height = Math.max(size, Math.floor(height / 2));
    const context = next.getContext("2d");
    context.imageSmoothingEnabled = true;
    context.drawImage(stage, 0, 0, next.width, next.height);
    stage = next;
    width = next.width;
    height = next.height;
  }

  scratch.width = size;
  scratch.height = size;
  const context = scratch.getContext("2d", { willReadFrequently: true });
  context.imageSmoothingEnabled = true;
  context.clearRect(0, 0, size, size);
  context.drawImage(stage, 0, 0, size, size);
  return context.getImageData(0, 0, size, size);
}

function toInput(source) {
  const size = model.input.size;
  const channels = model.input.channels;
  const pixels = shrinkOnto(source, size).data;

  const values = new Float32Array(size * size * channels);
  for (let p = 0, slot = 0; p < size * size; p++) {
    const r = pixels[p * 4], g = pixels[p * 4 + 1], b = pixels[p * 4 + 2];
    if (channels === 3) {
      values[slot++] = r / 255;
      values[slot++] = g / 255;
      values[slot++] = b / 255;
    } else {
      // Exactly Pillow's convert("L").
      values[slot++] = Math.round(0.299 * r + 0.587 * g + 0.114 * b) / 255;
    }
  }
  return values;
}

function showWhatItSees() {
  const target = $("small");
  const context = target.getContext("2d");
  context.imageSmoothingEnabled = false;
  context.clearRect(0, 0, target.width, target.height);
  context.drawImage(scratch, 0, 0, target.width, target.height);
}

// ---------------------------------------------------------------------------
// Showing the answer
// ---------------------------------------------------------------------------

function drawBars(probabilities) {
  const ranked = model.labels
    .map((name, index) => ({ name, p: probabilities[index] }))
    .sort((a, b) => b.p - a.p);

  $("bars").innerHTML = ranked.map((item, place) => `
    <div class="row${place === 0 ? " top" : ""}">
      <div class="name">${item.name}</div>
      <div class="bar"><div class="fill" style="width:${(item.p * 100).toFixed(1)}%"></div></div>
      <div class="pct">${(item.p * 100).toFixed(1)}%</div>
    </div>`).join("");
}

function classify(source) {
  if (!model) return;
  const probabilities = softmax(forward(toInput(source)));
  showWhatItSees();
  drawBars(probabilities);
}

// ---------------------------------------------------------------------------
// The self test: does JavaScript agree with Python?
// ---------------------------------------------------------------------------

function loadImage(source) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = source;
  });
}

async function runSelfTest(cases) {
  let worst = 0;
  for (const test of cases) {
    const image = await loadImage("data:image/png;base64," + test.png);
    const got = forward(toInput(image));
    for (let i = 0; i < got.length; i++) {
      worst = Math.max(worst, Math.abs(got[i] - test.logits[i]));
    }
  }

  const badge = $("selftest");
  if (worst < 0.02) {
    badge.className = "badge ok";
    badge.textContent =
      `self test passed: the browser agrees with Python (largest gap ${worst.toFixed(4)})`;
  } else {
    badge.className = "badge bad";
    badge.textContent =
      `self test FAILED: the browser and Python disagree by ${worst.toFixed(3)}. ` +
      `The page is preparing images differently from the way you prepared them ` +
      `for training. Check image_size and color.`;
  }
}

// ---------------------------------------------------------------------------
// Wiring up the three ways of giving it a picture
// ---------------------------------------------------------------------------

function showPanel(which) {
  for (const name of ["Cam", "File", "Draw"]) {
    $("panel" + name).hidden = name !== which;
    $("tab" + name).classList.toggle("on", name === which);
  }
}
$("tabCam").onclick = () => showPanel("Cam");
$("tabFile").onclick = () => showPanel("File");
$("tabDraw").onclick = () => showPanel("Draw");

// Webcam
let cameraRunning = false;
$("camStart").onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    const video = $("video");
    video.srcObject = stream;
    await video.play();
    cameraRunning = true;
    $("camStart").disabled = true;
    $("camStart").textContent = "Camera is on";
    const tick = () => {
      if (!cameraRunning) return;
      classify(video);
      setTimeout(() => requestAnimationFrame(tick), 120);
    };
    tick();
  } catch (error) {
    alert(
      "The browser would not give us the camera.\n\n" +
      "Cameras only work on https:// pages and on http://localhost. " +
      "If you opened this file by double clicking it, run " +
      "`python -m http.server` in this folder and open " +
      "http://localhost:8000 instead."
    );
  }
};

// Upload
$("file").onchange = async (event) => {
  const chosen = event.target.files[0];
  if (!chosen) return;
  const image = await loadImage(URL.createObjectURL(chosen));
  const preview = $("preview");
  preview.src = image.src;
  preview.hidden = false;
  preview.style.maxHeight = "240px";
  classify(image);
};

// Drawing
const pad = $("drawPad");
const padContext = pad.getContext("2d", { willReadFrequently: true });
function clearPad() {
  padContext.fillStyle = "#ffffff";
  padContext.fillRect(0, 0, pad.width, pad.height);
}
clearPad();
padContext.lineWidth = 16;
padContext.lineCap = "round";
padContext.lineJoin = "round";
padContext.strokeStyle = "#000000";

let drawing = false;
function padPoint(event) {
  const box = pad.getBoundingClientRect();
  const point = event.touches ? event.touches[0] : event;
  return {
    x: (point.clientX - box.left) * (pad.width / box.width),
    y: (point.clientY - box.top) * (pad.height / box.height),
  };
}
function startStroke(event) {
  event.preventDefault();
  drawing = true;
  const { x, y } = padPoint(event);
  padContext.beginPath();
  padContext.moveTo(x, y);
}
function continueStroke(event) {
  if (!drawing) return;
  event.preventDefault();
  const { x, y } = padPoint(event);
  padContext.lineTo(x, y);
  padContext.stroke();
  classify(pad);
}
function endStroke() {
  if (!drawing) return;
  drawing = false;
  classify(pad);
}
pad.addEventListener("pointerdown", startStroke);
pad.addEventListener("pointermove", continueStroke);
window.addEventListener("pointerup", endStroke);
$("clearPad").onclick = () => { clearPad(); };

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

async function start() {
  try {
    const [modelResponse, weightsResponse] = await Promise.all([
      fetch("model.json"),
      fetch("weights.bin"),
    ]);
    if (!modelResponse.ok || !weightsResponse.ok) throw new Error("not found");

    model = await modelResponse.json();
    weights = new Float32Array(await weightsResponse.arrayBuffer());

    if (weights.length !== model.n_floats) {
      throw new Error(
        `weights.bin holds ${weights.length} numbers but model.json expects ` +
        `${model.n_floats}. Run the export script again.`
      );
    }

    document.title = model.labels.join(" / ");
    $("title").textContent = model.labels.join("  ·  ");
    $("subtitle").textContent =
      `${model.labels.length} classes, ` +
      `${model.input.size}x${model.input.size} ` +
      `${model.input.channels === 3 ? "colour" : "grayscale"} input, ` +
      `${model.n_floats.toLocaleString()} parameters.`;

    drawBars(model.labels.map(() => 0));

    const selfTestResponse = await fetch("selftest.json");
    if (selfTestResponse.ok) {
      await runSelfTest((await selfTestResponse.json()).cases);
    } else {
      $("selftest").textContent = "self test: selftest.json is missing";
    }
  } catch (error) {
    $("subtitle").textContent = "Could not load the model.";
    $("selftest").className = "badge bad";
    $("selftest").textContent =
      "Could not read model.json and weights.bin. Either you have not trained " +
      "a model yet, in which case run `python src/run.py` first, or you opened " +
      "this file by double clicking it, which browsers block. Run " +
      "`python -m http.server -d docs 8000` and open http://localhost:8000. " +
      "(" + error.message + ")";
  }
}

start();
