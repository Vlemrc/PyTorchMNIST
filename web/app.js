const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let drawing = false;

// Gestion du dessin
canvas.addEventListener('mousedown', () => drawing = true);
canvas.addEventListener('mouseup', () => drawing = false);
canvas.addEventListener('mouseout', () => drawing = false);
canvas.addEventListener('mousemove', draw);

function draw(e) {
  if (!drawing) return;
  ctx.fillStyle = 'white'; // dessin en blanc
  ctx.beginPath();
  ctx.arc(e.offsetX, e.offsetY, 10, 0, 2 * Math.PI);
  ctx.fill();
}

function clearCanvas() {
  ctx.fillStyle = 'black'; // remplissage du fond en noir
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById('result').textContent = '';
}

// Initialisation
clearCanvas();

// Inférence
async function predict() {
  const imageData = ctx.getImageData(0, 0, 280, 280);
  const input = preprocess(imageData);

  const session = await ort.InferenceSession.create('./mnist.onnx');
  const tensor = new ort.Tensor('float32', input, [1, 1, 28, 28]);
  const output = await session.run({ input: tensor });
  const logits = output.output.data;
  const prediction = logits.indexOf(Math.max(...logits));

  document.getElementById('result').textContent = `Chiffre prédit : ${prediction}`;
}

// Prétraitement avec centrage
function preprocess(imageData) {
  const tempCanvas = document.createElement('canvas');
  const tempCtx = tempCanvas.getContext('2d');
  tempCanvas.width = 28;
  tempCanvas.height = 28;

  // Redessine en 28x28 brut
  tempCtx.drawImage(canvas, 0, 0, 28, 28);
  let imgData = tempCtx.getImageData(0, 0, 28, 28);
  let pixels = imgData.data;

  // Bounding box du chiffre dessiné
  let gray = [], minX = 28, minY = 28, maxX = 0, maxY = 0;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      let i = (y * 28 + x) * 4;
      let val = pixels[i]; // Rouge = Gris
      let norm = val / 255;
      gray.push(norm);
      if (norm > 0.2) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  // Si rien de dessiné
  if (minX > maxX || minY > maxY) {
    return new Float32Array(28 * 28).fill(0);
  }

  // Mise à l’échelle et recentrage
  let boxWidth = maxX - minX;
  let boxHeight = maxY - minY;
  let scale = 20 / Math.max(boxWidth, boxHeight);
  let dx = (28 - boxWidth * scale) / 2;
  let dy = (28 - boxHeight * scale) / 2;

  const centeredCanvas = document.createElement('canvas');
  const centeredCtx = centeredCanvas.getContext('2d');
  centeredCanvas.width = 28;
  centeredCanvas.height = 28;

  centeredCtx.fillStyle = "black"; // fond noir
  centeredCtx.fillRect(0, 0, 28, 28);
  centeredCtx.drawImage(
    canvas,
    minX * 10, minY * 10, boxWidth * 10, boxHeight * 10,
    dx, dy, boxWidth * scale, boxHeight * scale
  );

  let finalData = centeredCtx.getImageData(0, 0, 28, 28).data;
  const input = new Float32Array(1 * 1 * 28 * 28);
  for (let i = 0; i < 28 * 28; i++) {
    input[i] = finalData[i * 4] / 255; // plus d'inversion
  }

  return input;
}