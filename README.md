# CADGen: A Causal Language Model for CAD Generation
The CADGen model (for now) takes sketch-and-extrude command sequence for CAD models as input and performs next-token prediction similar to language models do.

Models with different training strategy or model architecture other than `transformer_lm` will be available sooner or later.

Example run:
```bash
python transformer_lm/train.py
```

You can create your own config file and run with `hydra` like the following:
```bash
python transformer_lm/train.py --config-path=your-config-path --config-name=your-config-filename
```

