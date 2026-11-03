Self-Evaluation 2026 — Xingqian Xu

---

What are your key contributions, accomplishments and/or strengths?

This year I was fortunate to work with the team across several stages of text-to-image development — data, training, evaluation, and early RL exploration.

One of my main duty is to work on curating OCR data and, together with the team, building SDG T2I data (OCR data being one form of SDG data). On our CVTG English benchmark, the resulting model was able to outperform leading open-source models of the time such as Qwen-Image-2512, and this data continues to benefit in Cosmos3.5.

I also worked with the team on the Cosmos3-Super-Text2Image model delivery. It reached #1 among open-source models at release, and currently ranks around 22–30 on Arena.ai including closed-source models.

After Cosmos3, we continued to study captioning as a team. I helped build Caption V2, a new captioner based on Qwen3-27B, and worked with the team to update all of our image data to the newly designed dense captions.

Over roughly one month, I worked with Runlong and other teams to explore Diffusion Reinforcement Learning. We were able to preliminary confirm that recent algorithms such as DPO, DiffusionNFT and SIFR are effective on both T2I and T2I text rendering, and on the infra side we enabled using frontier VLMs as reward models. This suggests Cosmos3-Super-Text2Image can still be improved further.

More recently, I have been driving the generation of new text layout / UI / chart / OCR data to support the team's T2I pretraining and midtraining experiments, and spending time diving into our image database to select pretrain/midtrain data and ingest more high-resolution images.

Also, I am working with Boxiang and the team on the Cosmos language model — data, architecture, and evaluation — with the plan to train a new LLM from scratch.
---

What are some challenges or barriers to doing your best work?

I have not encountered any significant blockers this year. The team has provided clear direction, sufficient compute, and strong support. The main thing I am mindful of (and honestly very motivated about) is ramping up quickly on new areas (e.g., Diffusion RL and LLM pretraining) as priorities shift to overcome new challenges. I see more as a learning opportunity than a barrier.
---

What are your future deliverables or goals?

I would like to keep pushing Cosmos text-to-image toward the top of the open-source leaderboard, and improve the model's text rendering quality. In parallel, I help build a medium size Cosmos LLM from scratch that can earn a place on open-source leaderboards and benefit downstream VLM and generation tasks.

Learderboard pushing on both diffusion and LLM are new to me, and I would try my best to build solid, hands-on experience on data, model and all related areas.

Longer term, I hope to grow more knowledge about both visual generation and language modeling, and continue contribute to Cosmos ultimate goal: the unified omni models.
