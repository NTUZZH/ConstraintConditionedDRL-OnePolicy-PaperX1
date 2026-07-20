# E5: mean makespan per regime (fair-warm, incumbent-retaining CP-SAT)

CP columns: best-of-four-PDR warm start, 8 workers on dedicated cores,
value = min(warm start, search). "+" = the search improved on the warm
start on ZERO instances, so the cell is the best dispatching rule.

| regime | greedy | best single PDR | PDR portfolio | GA-30s | CP 5s | CP 30s | CP 300s | opt | n(300s) |
|---|---|---|---|---|---|---|---|---|---|
| N | 130.8 | 132.6 (SPT) | 132.6 | 132.1 | 128.0 | 124.5 | 122.0 | 0 | 30 |
| L | 212.6 | 216.3 (SPT) | 215.8 | 214.2 | 204.0 | 203.5 | 202.6 | 25 | 30 |
| S | 194.6 | 195.6 (SPT) | 195.6 | 192.8 | 195.4+ | 191.2 | 188.2 | 0 | 30 |
| W | 148.3 | 151.0 (SPT) | 151.0 | 150.3 | 143.7 | 139.2 | 135.7 | 0 | 30 |
| LS | 271.2 | 275.3 (SPT) | 275.2 | 272.0 | 273.1+ | 269.0 | 265.5 | 0 | 30 |
| LW | 223.0 | 227.5 (SPT) | 226.7 | 226.7 | 212.8 | 210.0 | 208.2 | 15 | 30 |
| SW | 219.8 | 222.2 (SPT) | 222.2 | 220.2 | 221.2+ | 217.5 | 215.9 | 0 | 30 |
| LSW | 289.0 | 292.6 (SPT) | 292.6 | 290.5 | 290.0+ | 286.7 | 284.2 | 0 | 30 |

## LaTeX body

```
N & 130.8 & 132.6 & 132.6 & 132.1 & 128.0 & 124.5 & \textit{122.0} & \textit{0} \\
L & 212.6 & 216.3 & 215.8 & 214.2 & 204.0 & 203.5 & \textit{202.6} & \textit{25} \\
S & 194.6 & 195.6 & 195.6 & 192.8 & 195.4$^\dagger$ & 191.2 & \textit{188.2} & \textit{0} \\
W & 148.3 & 151.0 & 151.0 & 150.3 & 143.7 & 139.2 & \textit{135.7} & \textit{0} \\
LS & 271.2 & 275.3 & 275.2 & 272.0 & 273.1$^\dagger$ & 269.0 & \textit{265.5} & \textit{0} \\
LW & 223.0 & 227.5 & 226.7 & 226.7 & 212.8 & 210.0 & \textit{208.2} & \textit{15} \\
\midrule
SW & 219.8 & 222.2 & 222.2 & 220.2 & 221.2$^\dagger$ & 217.5 & \textit{215.9} & \textit{0} \\
LSW & 289.0 & 292.6 & 292.6 & 290.5 & 290.0$^\dagger$ & 286.7 & \textit{284.2} & \textit{0} \\
```
