CUDA_VISIBLE_DEVICES=0 python masking.py --model saves/s03_pcvos_ytvos.pth --output ./output/ --clip_length 5 --refine_clip ICR --T_window 2 --S_window 7 --shared_proj --memory_read PMM --predict_all --time


python masking.py --model saves/s03_pcvos_ytvos.pth --output ./output/ --shared_proj --memory_read PMM --predict_all --time



CUDA_VISIBLE_DEVICES=0 python masking.py --model saves/s03_pcvos_ytvos.pth --output ./output/ --T_window 2 --S_window 7 --shared_proj --memory_read PMM --predict_all --time
