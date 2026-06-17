# Dog Detection Documentation Site

A beautiful, self-contained documentation website with a live demo page that runs inference using the trained model weights.

## Run locally

```bash
cd docs
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-docs.txt
uvicorn docs.app:app --reload --host 0.0.0.0 --port 5050
```

Then open http://localhost:5050 in your browser.

## Pages

- `/` — Full project documentation (features, quick start, scripts, database schema, training results)
- `/demo` — Upload any image and run real-time dog detection with the model at `runs/detect/DogDetection/train-30/weights/best.pt`

## Notes

- The demo requires the model weights file to exist at the path above.
- First inference request will load the model into memory; subsequent requests are faster.
- The site supports dark and light themes. The theme preference is saved in your browser.
