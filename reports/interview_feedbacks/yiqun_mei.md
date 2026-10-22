# Interview Feedback: Yiqun Mei

**Interviewer:** Xingqian Xu (xingqianx@nvidia.com)
**Score:** 4

---

Yiqun Mei is a researcher in video generation, most recently at Adobe, where he works on an interactive world model (interactive video generation). His recent focus is on turning bidirectional video diffusion models into autoregressive ones that can be controlled in real time, and on making long-horizon generation efficient through sparse attention.

During the first session of the interview, we discussed Yiqun's most recent work at Adobe on an interactive world model (an interactive video generation model). His main contribution was the design of a pipeline that distills a bidirectional video model into an autoregressive (AR) model, which then takes 3D pose as the control signal and generates the next video frame AR-ly.

Beyond the distillation itself, Yiqun also explored a sparse attention structure to make long video generation feasible:
- During training, the causal mask is further dropped out according to whether frames are related to each other.
- During inference, attention weights between frame tokens that fall below a relevance threshold are cut off to zero, making generation sparse.
- The resulting sparsity reduces the KV cache, which is what makes long video generation practical.

The project is directly relevant to where Cosmos is heading (for example, controllable, long-horizon generation), and Yiqun was able to explain the design choices and the motivation behind them.

Next We went through two diffusion related short question.

1. Understanding of CFG: Yiqun answered this completely correctly.
2. Improving a rectified-flow generative model: With data and architecture held fixed, Yiqun proposed using RL to improve the model. This is a reasonable direction, but not the first thing to try. The more natural first step is a parameter sweep, e.g. learning rate and the timestep shift. Yiqun was not familiar with the fact that the optimal shift depends on resolution and needs to be re-tuned when resolution changes. This points to a gap in practical, hands-on experience with training rectified-flow models.

Based on the entire interview process, I give Yiqun a 4 accept.
