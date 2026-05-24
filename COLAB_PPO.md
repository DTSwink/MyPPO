# MyPPO PPO Colab Runner

Use a GPU runtime, then run these cells from the repository root in Colab.

```bash
!git clone https://github.com/DTSwink/MyPPO.git
%cd MyPPO
!pip install -r requirements.txt
```

Single PPO run:

```bash
!python scripts/train_ppo.py \
  --fresh \
  --checkpoint-dir checkpoints/ppo_colab \
  --num-envs 2048 \
  --num-steps 180 \
  --total-timesteps 25920000 \
  --eval-interval 5 \
  --target-loss 1e-5
```

Autonomous sweep:

```bash
!python scripts/sweep_ppo.py \
  --output-dir checkpoints/ppo_sweep \
  --max-trials 5 \
  --timesteps-per-trial 25920000 \
  --num-envs 2048 \
  --eval-envs 2048 \
  --target-loss 1e-5
```

The visualizer-compatible weights are written to each trial's `latest.npz`.
The sweep stops early when a trial reaches the target loss; otherwise it stops
plateaued trials and moves to the next PPO configuration.
