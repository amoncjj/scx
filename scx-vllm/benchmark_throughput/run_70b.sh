#!/bin/bash
set -e  # 如果任何命令出错则退出

python llama-70b.py 2>&1 | tee 70b_plaintext.log
python llama-70b.py --scx_enc_layers 0 2>&1 | tee 70b_scx_input.log
python llama-70b.py --scx_enc_layers 0,27 2>&1 | tee 70b_scx_io.log
