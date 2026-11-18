# hey-elemental — Spanish wake-word model for openWakeWord

Custom openWakeWord model that detects the Spanish wake phrase **"hey elemental"**.

## Files

- `hey_elemental.tflite` — 251 KB, TensorFlow Lite model for on-device detection.
  Input `[1, 20, 96]` float32, output `[1, 1]` (sigmoid).
- `hey_elemental.onnx` — 247 KB, ONNX version of the same model (opset 18,
  self-contained, no external tensors).

## How it was trained

- Tool: openWakeWord automatic training notebook
  (`automatic_model_training.ipynb`, [dscripka/openWakeWord](https://github.com/dscripka/openWakeWord))
  on Google Colab (free T4 GPU).
- Positive samples: synthetic Spanish speech only — 13 distinct voices from
  Fish Audio TTS (model `s2.1-pro-free`) saying "hey elemental", augmented
  with noise, room impulse responses, pitch and speed variation
  (1,000 training / 1,000 validation samples).
- Negative samples: adversarial phrases generated with Piper TTS plus
  ESC-50 background audio.
- Config: `target_phrase = ["hey elemental"]`, `steps = 10000`,
  `target_accuracy = 0.6`, `target_recall = 0.25`, `layer_size = 32`,
  `model_type = dnn`.

## Metrics (second training run)

| Metric | Result | Target |
|---|---|---|
| Accuracy | 0.756 | 0.6 |
| Recall | 0.517 | 0.25 |
| False positives | 1.593 / hour | 0.2 |

False positives did not meet the 0.2/hour target, so in real use the model
may wake more often than ideal. Raising the detection threshold helps
(see below).

## End-to-end verification

Scored with openWakeWord `Model.predict_clip` on held-out clips:

- Spanish positives ("hey elemental", including voices held out from
  training): ~0.93 (adam 0.927, femenina 0.933, valentino 0.930)
- Negative clips (Piper speech, ESC-50 background): <= 0.001
- ONNX and TFLite scores agree to ~1e-6.

## Usage

```python
from openwakeword import Model

model = Model(wakeword_models=["hey_elemental.tflite"],
              inference_framework="tflite")
# Feed 80 ms frames (1280 samples @ 16 kHz) to model.predict(audio);
# fire when the score stays >= 0.75 for 4 consecutive frames.
```

Recommended settings: threshold `0.75`, `confirmation_frames = 4`.

## License

MIT. The model was trained with openWakeWord (Apache-2.0); this
repository's contents are released under the MIT license (see `LICENSE`).
