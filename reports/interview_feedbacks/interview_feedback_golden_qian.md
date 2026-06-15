# Interview Feedback: Golden Qian

**Date:** 2026-04-29
**Interviewer:** Xingqian Xu (xingqianx@nvidia.com)
**Co-interviewer:** Jinwei
**Recommendation:** Lean Hire (with reservations on scope fit)

---

## A. Coding Round — Flow Matching

The coding round centered on flow matching. Golden completed the training portion cleanly and within expected time. He got tripped up on a shape-mismatch issue when implementing the Euler sampler and was unable to fully resolve it during the session. Aside from that bug, the surrounding code was structured correctly and the intent of his sampler was clear.

## B. Overall Coding / Theory Assessment

Despite the sampler bug, I would consider this a pass on the coding round. His theoretical understanding of flow matching was fully correct — he reasoned about the velocity field, the training objective, and the sampler trajectory accurately, and the coding gap looked like a debugging issue rather than a conceptual one. Strong fundamentals.

## C. Research Discussion (with Jinwei) — VLM as Back-propagatable Reward in RL

Jinwei walked through one of Golden's research projects in which a VLM is used as a back-propagatable reward function inside an RL loop for generative models. The discussion demonstrated real hands-on experience with RL for generation models — he could speak to the reward design, gradient flow through the VLM, and trade-offs of the approach with credibility. This is a clear differentiator and directly relevant to where Cosmos is heading.

## D. Background and Scope Concern

Golden has led multiple research projects at Snap, mostly in:
- Pretraining architectures
- Personalization
- Reinforcement learning for generative models

The breadth and the fact that he has been the lead on these efforts is positive. **My main concern is scope fit.** Coming from leading multiple independent research threads, he may be looking for a larger scope than what we can offer on Cosmos in the near term. Worth probing in the next round whether he is comfortable joining as a contributor on a more focused slice of the roadmap before scope expands, vs. expecting a lead-sized charter from day one.

---

## Summary

| Dimension | Assessment |
|---|---|
| Coding (flow matching) | Pass — bug on Euler sampler, theory solid |
| Theoretical depth | Strong |
| RL for generative models | Strong, hands-on (per Jinwei) |
| Research leadership | Proven at Snap |
| Scope fit for Cosmos role | Concern — may want bigger scope |

**Next step:** align with Jinwei and the hiring manager on whether the scope/level conversation can be had explicitly in the next round.
