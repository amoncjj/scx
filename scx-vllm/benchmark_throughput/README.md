```bash
conda activate scx

python xx.py 2>&1 | tee xx_plaintext.log

python xx.py --scx_enc_layers 0 2>&1 | tee xx_scx_input.log

python xx.py --scx_enc_layers 0,27 2>&1 | tee xx_scx_io.log
```