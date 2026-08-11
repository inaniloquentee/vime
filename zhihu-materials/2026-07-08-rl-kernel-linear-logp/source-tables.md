# Source Tables

这些是 `article-zhihu.md` 里转成图片的原始表格。知乎编辑器如果能接受表格，可以直接复制这些表格替换对应图片。

## Operator-level CUDA timing

| Config | Tokens/call | fwd CUDA vime | fwd CUDA vime + RL-Kernel | fwd speedup | fwd+bwd CUDA vime | fwd+bwd CUDA vime + RL-Kernel | fwd+bwd speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T1 | 796 | 5.14 ms | 3.51 ms | 1.46x | 15.24 ms | 12.37 ms | 1.23x |
| T2 | 1820 | 7.78 ms | 3.36 ms | 2.32x | 18.56 ms | 10.37 ms | 1.79x |
| T3 | 6827 | 14.52 ms | 7.62 ms | 1.91x | 33.96 ms | 18.50 ms | 1.84x |

## Single-operator peak reserved memory delta

| Config | reserved delta vime | reserved delta vime + RL-Kernel | single-op reserved saving |
| --- | ---: | ---: | ---: |
| T1 | 4056 MB | 3112 MB | 944 MB |
| T2 | 6684 MB | 4862 MB | 1822 MB |
| T3 | 32342 MB | 26710 MB | 5632 MB |

## Training sanity metrics

| Config | raw_reward vime | raw_reward vime + RL-Kernel | abs_diff vime | abs_diff vime + RL-Kernel | fallback |
| --- | ---: | ---: | ---: | ---: | ---: |
| T1 | 0.0000 | 0.0000 | 0.02668 | 0.02537 | 0 |
| T2 | 0.0278 | 0.0278 | 0.02264 | 0.02395 | 0 |
| T3 | 0.1111 | 0.0972 | 0.02034 | 0.02224 | 0 |
