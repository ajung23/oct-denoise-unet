setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
synth:
	python scripts/make_synthetic_oct.py --out data/synth --n 20
n2v:
	python scripts/denoise_n2v.py --data data/synth/raw --out runs/n2v_demo --epochs 1
unet:
	python scripts/train_unet.py --data data/synth --out runs/unet_demo --epochs 2
eval:
	python scripts/eval_metrics.py --pred runs/unet_demo/preds --gt data/synth/masks
