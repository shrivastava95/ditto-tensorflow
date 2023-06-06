cd storage/ditto/ditto-tensorflow

python3  -u main.py --dataset=fmnist \
            --optimizer=ditto \
            --learning_rate=0.05 \
            --num_rounds=1000 \
            --eval_every=10 \
            --clients_per_round=10 \
            --batch_size=16 \
            --q=0 \
	      --seed=0 \
            --model='cnn' \
	      --sampling=2  \
            --num_corrupted=0 \
            --boosting=0 \
            --random_updates=0 \
            --gradient_clipping=0 \
            --krum=0 \
            --mkrum=0 \
            --median=0 \
            --k_norm=0 \
            --k_loss=0 \
            --fedmgda=0 \
            --fedmgda_eps=0 \
            --alpha=0 \
            --global_reg=-1 \
            --lam=1 \
            --dynamic_lam=0 \
            --finetune_iters=40 \
            --decay_factor=1 \
            --local_iters=2 \



