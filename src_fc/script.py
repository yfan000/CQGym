import io
import matplotlib.pyplot as plt

pg_cpu = []
pg_gpu = []
#cpu rt(s): 133.89343452453613
with open('pgcpu.txt', 'r') as file:
    for line in file:
      line = line[:6]
      if float(line) != 0.0:
          pg_cpu.append(float(line))
#gpu rt(s): 125.54547047615051
with open('pggpu.txt', 'r') as file:
    for line in file:
      line = line[:6]
      if float(line) != 0.0:
          pg_gpu.append(float(line))

plt.hist(pg_cpu, bins=10, alpha=0.6, label='cpu')
plt.hist(pg_gpu, bins=10, alpha=0.4, label='gpu')
plt.xlabel('Train time (s)')
plt.ylabel('Count')
plt.title("PG train time")
plt.legend(loc='upper right')
plt.show()
plt.clf()

ppo_cpu = []
ppo_gpu = []
#ppo cpu rt(s): 858.0965530872345
with open('ppocpu.txt', 'r') as file:
    for line in file:
      line = line[:6]
      if float(line) != 0.0:
          ppo_cpu.append(float(line))
#ppo gpu rt(s): 188.3126256465912
with open('ppogpu.txt', 'r') as file:
    for line in file:
      line = line[:6]
      if float(line) != 0.0:
          ppo_gpu.append(float(line))

plt.hist(ppo_cpu, bins=40, alpha=0.6, label='cpu')
plt.hist(ppo_gpu, bins=2, alpha=0.4, label='gpu')
plt.xlabel('Train time (s)')
plt.ylabel('Count')
plt.title("PPO train time")
plt.legend(loc='upper right')
plt.show()
