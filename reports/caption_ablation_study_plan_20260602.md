# Caption Ablation Study Plan

**Date:** 2026-06-02

---

## 1. Introduction and Motivation

High-quality captions are a foundational pillar of the omni-model. The exact alignment between a caption and its corresponding multi-media is the key for alignment between the two modalities at inference time. Despite being an important pipeline, so far our caption study is incomplete, and thus more evidence needs to be provided.

Our current situation is: Cosmos3 VFM is trending toward the top of the generation leaderboard, which demonstrates that JSON caption training with upsampled JSON inference most likely works. However, the consistent performance gap between VFM results from JSON caption upsampled by Cosmos3 Reasoner vs. external LLMs/VLMs (i.e. Gemini, Claude Opus) becomes our newest trouble.

No doubt, the ultimate goal is to close this gap in our future study. Yet this is never a single problem. It is a composite of several tightly coupled sub-problems which are listed below:

- **Caption accuracy.** Are the captions in our training data good, maybe simply this is a data issue?
- **Caption format entropy.** Does the caption JSON format give us a hard time finetuning a language model due to the "text entropy" and all redundant tokens? Making our finetuning of the Cosmos3 reasoner very challenging?
- **Caption completeness.** Are our structured captions in training data complete? Are all subjects, style, lighting, composition, motion, camera, mood, etc. mentioned? Missing fields leave the model unable to respond to prompts that exercise those axes.
- **Training-side issues.** Do the train-time parameters, like bad learning rate and scheduler, affect our final judgement?

Too many interacting factors and we have to rule them out one after another to guide us through the mystery of the problem.

---

## 2. Experiment Design

All experiments target maximum controllability so that each result provides clean directional evidence rather than noisy correlation. All ablations will be conducted on the **image SFT** setting and mainly on UniGenBench scores, for two reasons:

**(a) Training-feedback efficiency.** Image models train and evaluate significantly faster than video models, compressing the iteration cycle.

**(b) The all-time-high rule.** We shall conduct all runs on 64B model (Cosmos3-T2I in this case), depending on the experiment goal, we should resume from all-time-high Cosmos3-T2I, our GA checkpoint, or other "nearby" checkpoints.

The following sections list potential experiments, ordered roughly from easy to hard, from low to high cost...

---

### 2.1 [Inference-Only] All-Time-High Model with Upsampled Captions from Different Accessible VLMs

This is simply a reproduction of the gap between model inference using upsampled JSON prompt from Cosmos3 and other SoTA LLMs.

Please see the following result:

| Upsampler                 | UGB all   | UGB orig   | UGB phi   | Aesthetic overall   | HPSv3 overall   | Photorealism phi   |
|---------------------------|-----------|------------|-----------|---------------------|-----------------|--------------------|
| GPT-5.5 xhigh-fast        | 0.9292    | 0.9472     | 0.9127    | 6.109               | 11.1237         | 0.4056             |
| Opus4.7                   | 0.9252    | 0.9462     | 0.9059    | 6.1413              | 11.3263         | 0.3653             |
| Gemini 3.1 Pro think32768 | 0.9235    | 0.9452     | 0.9036    | 5.9904              | 10.5802         | 0.3386             |
| Qwen3.5 397B thinking     | 0.9192    | 0.9355     | 0.9043    | 6.0259              | 10.847          | 0.3667             |
| Nemotron3Ultra            | 0.9006    | 0.9242     | 0.879     | 6.0351              | 11.2893         | 0.2737             |
| Nemotron3Ultra thinking   | 0.8987    | 0.9281     | 0.8718    | 6.0736              | 11.1395         | 0.2985             |
| Raw dense prompts         | 0.8848    | 0.9097     | 0.8619    | 6.0521              | 11.0625         | 0.298              |
| Reasoner v4.2 expressive  | 0.879     | 0.9105     | 0.8502    | 5.9702              | 10.8613         | 0.4649             |

**Expected conclusion:** This is our "anchor gap" when using different upsampling models.

---

### 2.2 [Inference-Only] All-Time-High Model with Different Upsampled Caption Formats

Given a fixed upsampler (e.g., Opus), evaluate how sensitive the all-time-high checkpoint is to the surface format of the upsampled caption. Candidate formats:

| Format                               | Description                                         |
|--------------------------------------|-----------------------------------------------------|
| Dense                                |                                                     |
| JSON (Opus 4.7)                      |                                                     |
| Format-Clean Flat Dense              | Upsampled but no quotes, no indent, no semicolons   |
| Format-Clean Bag of Words            | What if we even break the sentence completely, into bag of words |
| Field Agnostic JSON                  | What if we trash the field name information         |
| HTML                                 | Structured with tags (`<subject>`, `<style>`, etc.) |
| YAML                                 | Key-value pairs, human-readable                     |

**Expected conclusion:** A clue about the impact of caption format on model performance. The all-time-high model is already overfitted on JSON captions. But if the scores are rather uniform, Cosmos3 has perhaps already worked out how to "filter out" text format entropy.

---

### 2.3 [SFT-Required] Train a Group of Branched Checkpoints to Obtain _Golden Caption Format_

Branch from the current non-preference aligned checkpoints (i.e. the checkpoint prior to finetuned on ultra high quality data), each with a different caption format (i.e. JSON vs. XML vs. Markdown… ).

The result will be put into this table:

| UGB Score               | Dense   | JSON (Opus 4.7)   | Format-Clean Flat Dense   | Field Agnostic JSON   | HTML   | YAML   | MarkDown   |
|-------------------------|---------|-------------------|---------------------------|-----------------------|--------|--------|------------|
| Dense                   |         |                   |                           |                       |        |        |            |
| JSON (Opus 4.7)         |         |                   |                           |                       |        |        |            |
| Format-Clean Flat Dense |         |                   |                           |                       |        |        |            |
| Field Agnostic JSON     |         |                   |                           |                       |        |        |            |
| HTML                    |         |                   |                           |                       |        |        |            |
| YAML                    |         |                   |                           |                       |        |        |            |

**Expected conclusion:** We can identify an ultimate "easy" caption format that is friendly for both VFM training and inference, namely _Golden Caption Format_. This will likely also be "Golden" for VLM (i.e. Cosmos3 Upsampler) training.

---

### 2.4 [SFT-Required] Use 2.3 Dense SFT checkpoint. Try different upsampling schema using a SoTA LLM Opus 4.7 to Obtain _Golden Caption Schema_

The hypothesis is that our current "schema" for generating upsampled captions may be sub-optimal: some important fields are missing, and some fields may be redundant and thus yield diminishing returns. CosCapBench-Image [doc] may not reveal this faithfully.

- We need a Dense SFT checkpoint to rule out the effect of caption format.
- Different captions with different schema / hierarchy / focused fields will all be generated by Opus 4.7 to rule out the factor of upsampler capability.
- By doing this, we will identify what "information" is critical to boost T2I.

Results will be put into this table:

| Caption Version    | UGB all   | UGB orig   | UGB phi   |
|--------------------|-----------|------------|-----------|
| Dense Baseline     |           |            |           |
| JSON (Opus 4.7)    |           |            |           |
| JSON Up With Schema 1   |           |            |           |
| JSON Up With Schema 2   |           |            |           |
| JSON Field Dropout |           |            |           |

**Expected conclusion:** To identify the ideal JSON for training that (a) covers all important information and (b) contains only necessary, non-duplicated information (to save context tokens at the VLM tower).

---

### 2.5 [SFT-Required] Obtain a subset of data with "golden ground truth caption" (using Gemini 3.1 Pro, the best VLM so far), train VLM and VFM to see if data quality is the key.

The hypothesis is that our JSON caption quality is poor, which is the reason why our finetuned Cosmos3 reasoner (upsampler) is not performing very well.

- To train VFM, we SFT from Cosmos3 GA 64B checkpoint.
- To train VLM, we SFT from Qwen3 32B.
- Focus on one HQ real image dataset, because caption quality is noticeably more important for real data (than SGD). Obtain golden ground truth caption with Gemini 3.1 Pro.
- Run a controlled SFT pair, one using current JSON caption, one using Gemini 3.1 Pro JSON caption (i.e. golden ground truth caption) with exactly the same captioning schema. For both VFM and VLM.

Results will be put into this table:

|                                 | UGB (opus)   | UGB (current VLM)   | UGB (new VLM)   |
|---------------------------------|--------------|---------------------|-----------------|
| Current Best T2I                |              |                     |                 |
| Finetuned VFM with old JSON     |              |                     |                 |
| Finetuned VFM with G3p1Pro JSON |              |                     |                 |

_new VLM = VLM finetuned with Gemini 3.1 Pro Caption_

**Expected conclusion:** Better caption quality is all we need to get a good Cosmos3 Upsampler. Better caption quality is more important to VLM than VFM.


### 2.6 [SFT-Required] Understand the impact of caption entropy to both VLM and VFM.

When we reach this part, we should already hold multiple versions of captions. It is time to check how omni we can make our model. This will split into (a) whether VLM can understand captions while disentangling from format, and (b) whether VFM can generate high quality images disregarding format.

- For both VFM and VLM, we train a pair of SFTs: (a) one sees only JSON Gemini 3.1 Pro golden captions. (b) The other sees "a mixture formatted" Gemini 3.1 Pro golden caption, and some level of information dropout may be applied.
- TBH, such "caption augmentation" is applied while training the leaderboard Cosmos3-T2I (also for I2V)

Results will be put into this table:

| Checkpoint                                                                        | Benchmark Caption Version   | UGB all   | UGB orig   | UGB phi   |
|-----------------------------------------------------------------------------------|-------------------|-----------|------------|-----------|
| Finetuned VFM with G3p1Pro JSON                                                   | Opus JSON         |           |            |           |
| Finetuned VFM with G3p1Pro JSON                                                   | Opus MoF         |           |            |           |
| Finetuned VFM with G3p1Pro mixture of format                                      | Opus JSON              |           |            |           |
| Finetuned VFM with G3p1Pro mixture of format                                      | Opus MoF              |           |            |           |
| Finetuned VLM with G3p1Pro JSON (then evaluate on all-time-high VFM)              | Opus JSON                  |           |            |           |
| Finetuned VLM with G3p1Pro JSON (then evaluate on all-time-high VFM)              | Opus MoF JSON                  |           |            |           |
| Finetuned VLM with G3p1Pro mixture of format (then evaluate on all-time-high VFM) | Opus JSON              |           |            |           |
| Finetuned VLM with G3p1Pro mixture of format (then evaluate on all-time-high VFM) | Opus MoF              |           |            |           |

_MoF = Mixture of Caption Formats_

**Expected conclusion:** Hopefully, we will see that the performance of VFM and VLM is higher when trained with a mixture of caption formats. We successfully make VLM and VFM understand the semantics instead of overfitting the format. So in later training, prompt format augmentation is always preferred. It also demonstrates that a "correct and accurate" caption is more important, and JSON/XML/Dense format is not the key.
