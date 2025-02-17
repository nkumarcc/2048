# 2048: Analysis of Game Winning Strategies

Comparison of:

1. Expectimax
2. N-Tuple
3. Monte Carlo Tree Search
4. Deep Q-Network
5. Proximal Policy Optimization
6. NeuroEvolution of Augmenting Topologies

## Expectimax

Pre-Watching:

- [Minimax and Alpha-Beta Pruning](https://www.youtube.com/watch?v=zp3VMe0Jpf8)
- [Expectimax](https://www.youtube.com/watch?v=4yMvc1Uph-Y)

## N-Tuple

Pre-Watching/Reading:

- [Systematic Selection of N-Tuple Networks for 2048](https://www.youtube.com/watch?v=eoVAukW2etA)
  - [Paper](https://ieeexplore.ieee.org/document/7880154)
- [Multi-Stage Temporal Difference Learning for 2048-like Games](https://arxiv.org/pdf/1606.07374)

Improvements:

- Rotating & mirroring the board to reduce variance
- Add More Expressive N-Tuples
- Combining multiple tuples into a single function approximator
- Add Exploration (ε-Greedy Strategy)
- Improve Credit Assignment (Delayed Rewards)