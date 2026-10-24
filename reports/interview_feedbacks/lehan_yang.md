# Interview Feedback: Lehan Yang (Research Intern)

**Interviewer:** Xingqian Xu (xingqianx@nvidia.com)
**Position:** Research Intern
**Score:** 4 (accept)

---

## Background

Lehan Yang is a PhD candidate in Data Science at the University of Virginia (since Aug 2024, expected May 2028), advised by Prof. Sheng Li. He holds a Bachelor of Advanced Computing from the University of Sydney (2020-2024). He has done two research scientist internships at Adobe Research (May 2025 - Nov 2025, and May 2026 - present).

His research sits at the intersection of diffusion models and dense prediction / video:
- PixelDense (NeurIPS 2026, first author): dense prediction as representation alignment for pixel diffusion.
- Unified Dense Prediction of Video Diffusion (CVPR 2025, first author).
- UniGS: unified representation for image generation and segmentation (CVPR 2024, co-first author).
- CoMatting (unified image matting and correction) and VRMDiff (text-guided video referring matting generation), both first author and under review / on arXiv.

He also has a strong competition track record: multiple NTIRE challenge placements at CVPR 2021-2024 workshops and four Kaggle medals (one gold) in 2019.

---

## A. Research Discussion

Lehan presented his research on PixelDense (dense prediction as representation alignment for pixel diffusion). Since the work is in pixel-space diffusion, we discussed Kaiming He's JiT ("Back to Basics: Let Denoising Generative Models Denoise") and I asked why it moves from velocity prediction to x0 (clean image) prediction. Lehan was able to explain what JiT does. I then followed up with the sigma shift question below, since it is closely related to this topic.

## B. Interview Quiz

1. CFG formulation: Lehan answered all of it correctly.
2. Sigma shift: I asked whether he knew that the sigma (timestep) shift needs to be adjusted according to the information dimension (e.g. resolution / number of tokens). He was not very familiar with this.

---

## Assessment

Lehan has a solid publication record with first-author papers at top venues (CVPR, NeurIPS), directly on diffusion models for images and video, and has already worked at an industry research lab for two internships. The combination of generative modeling and dense prediction is a good fit for a research internship on our team, and the competition background suggests he is strong at hands-on execution and iterating on experiments.

He got CFG fully correct. The gap is in training details such as the sigma shift, which depends on the information dimension. This is a practical detail that is easy to pick up during an internship and does not change my recommendation.

Based on the entire interview process, I recommend Lehan for the internship and give him a **4**.
