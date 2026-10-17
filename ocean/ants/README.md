# Ants

Two ant colonies compete to forage food. Each ant is an agent: it finds food, carries it back to its own colony, and lays pheromone trails while carrying food. Other ants of the same colony can sense those trails. The whole env is in [ants.h](ants.h), and the hyperparameters are in [config/ants.ini](../../config/ants.ini).

## Train

Run everything from the repo root. `build.sh` compiles the env into a single trainer binary, `./puffer`.

```bash
./build.sh ants            # builds ./puffer with ants compiled in
./puffer train             # 100M steps, about 1 minute on a DGX Spark
```

Override any config value from the command line with `--section.key=value`:

```bash
./puffer train --train.total_timesteps=30_000_000 --base.seed=2
```

Checkpoints are written to `checkpoints/ants/<run_id>/<step>.bin`. When a run finishes, its config and binned metric curves go to `logs/ants/<run_id>.ini`, under `[metrics]`. Experiment trackers like Neptune and wandb aren't supported in 5.0.

Keep `learning_rate = 0.005`. At the default of 0.015, every 100M-step seed collapsed into a uniform random policy (entropy = ln 4) around 30 to 60M steps, after which ants stopped delivering.

## Evaluate / visualize

Both commands use the trainer binary from `./build.sh ants`. `latest` selects the newest checkpoint under `checkpoints/ants/`; you can also pass a path, such as `checkpoints/ants/<run_id>/<step>.bin`.

```bash
./puffer eval latest                                        # watch the policy in a raylib window
./puffer eval latest --headless --base.eval_episodes=2000   # no window; prints score/perf
```

For headless eval the episode count is the number of deliveries scored; the default, 10000, takes longer.

Window controls: `V` toggles vision cones, `P` toggles the pheromone sensing range, `TAB` toggles fullscreen, and `ESC` quits.

You can also build a standalone CPU viewer, which needs no CUDA at runtime:

```bash
./build.sh ants --cpu      # builds ./ants
./ants latest
```

## Building on DGX Spark (aarch64)

`build.sh` picks `raylib-5.5_linux_aarch64` automatically on ARM Linux. Linking needs NCCL and clang's OpenMP runtime; the CPU viewer needs the OpenMP runtime too:

```bash
sudo apt install libnccl-dev libomp-dev
```

## Environment

| | |
|---|---|
| Agents per env | `env.num_agents` (default 32, max 64), split evenly between the two colonies |
| Map | 1280 × 720, wraps at the edges |
| Food | up to 20 sources of 20 units each; a depleted source respawns somewhere else |
| Reset | each ant respawns at its colony every 2048 steps (`ANT_RESET_INTERVAL`), with `terminal = 1` |

**Actions** (1 discrete head, 4 options): 0 turn left, 1 turn right, 2 move forward, 3 no-op. Turns are 15°, and forward moves 5 px. An ant that has gone more than 100 steps without food also turns randomly 5% of the time.

**Observations** (27 floats per ant):

| Index | Feature |
|---|---|
| 0–1 | vector to own colony (dx/width, dy/height) |
| 2–3 | vector to nearest food inside the vision cone (75 px range, 60°), or 0 if none |
| 4–23 | 5 strongest own-colony pheromones within 100 px: dx, dy, direction/π, strength |
| 24 | carrying food (0/1) |
| 25 | heading / 2π |
| 26 | fraction of friendly ants within 100 px |

**Rewards:** `reward_food_pickup` (0.1) when an ant picks up food, and `reward_delivery` (10.0) when it delivers to its own colony. The trainer clips rewards to [-1, 1], so a delivery counts as 1.0 during training.

**Pheromones:** an ant carrying food drops one every 5 steps, recording its heading. Strength starts at 1.0 and evaporates by 0.005 per step, with at most 5000 pheromones in the env.

## Metrics

All metrics are logged once per delivery and averaged over deliveries:

| Key | Meaning |
|---|---|
| `episode_length` / `avg_delivery_steps` | steps the ant has lived (since spawn or its last delivery) when it delivers; lower is better |
| `episode_return` | reward per delivery, including pickups by ants that didn't deliver |
| `score` | mean of 1000 / steps-per-delivery (the sweep metric) |
| `perf` | mean of 1 − steps/2048, in [0, 1] |
| `colony1_food`, `colony2_food` | fraction of deliveries made by each colony |
| `successful_trips`, `total_resets` | food pickups and ant resets per delivery |

Before 5.0 these metrics had different meanings. In 3.0, `score` and `perf` were computed differently, and `score` shrank as more deliveries were logged. Only `episode_length` and `episode_return` can be compared with 3.0 runs.

Reference result (3 seeds, 100M steps, `learning_rate = 0.005`): about 169 steps per delivery after about 0.8 minutes. That matches the 3.0 baseline of about 166 steps per delivery, which took 3.9 minutes.
