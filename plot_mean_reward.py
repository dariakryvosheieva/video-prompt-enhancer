import json
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "out/stage2_traces.jsonl"

rewards_by_step = defaultdict(list)

with open(path) as f:
    for line in f:
        row = json.loads(line)
        step = row["step"]
        reward = row["reward"]
        rewards_by_step[step].append(reward)

steps = sorted(rewards_by_step.keys())
mean_rewards = [sum(rewards_by_step[s]) / len(rewards_by_step[s]) for s in steps]

df = pd.DataFrame({"step": steps, "mean_reward": mean_rewards})

coeffs = np.polyfit(steps, mean_rewards, deg=1)
fit_fn = np.poly1d(coeffs)

plt.figure(figsize=(8, 5))
plt.plot(steps, mean_rewards, marker="o", label="Mean reward")
plt.plot(steps, fit_fn(steps), linestyle="--", label="Best-fit line")
plt.xlabel("Training step")
plt.ylabel("Mean reward")
plt.title("Mean Reward vs. Training Step")
plt.grid(True)
plt.tight_layout()
plt.savefig("mean_reward.png")
