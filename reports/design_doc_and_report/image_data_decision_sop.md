<!-- synced from https://docs.google.com/document/d/1qdTDx5GzMWP3uCNvCORyxNvYngYK9QdGnRXRXwru9-g/edit revision AHj4eMQcRnTNroHvZXvzekApfCCcdHvi_iyRBdYOKuWXnLl1Xo_5jE0nWRhQlzkyQfAaEi46-qIz57nFIRHggsFEMrovB13vCDtAO4kfeyA -->

# Design Doc: Image Data Onboarding SOP

Authors: @Xingqian Xu US
Created: Oct 21, 2026
Status:

| **Reviewer** | **Last Updated** | **Status** |
|---|---|---|
| @Ming-Yu Liu US |  |  |
| @Yogesh Balaji US |  |  |
| @Francesco Ferroni GB |  |  |
| @Jinwei Gu US |  |  |

## Motivation

Occasionally, new image data sources become available for potential integration into our training pipeline. Rather than conducting resource-intensive and time-consuming ablation experiments for every candidate dataset, we establish a standardized workflow to evaluate data quality and potential impact using rigorous statistical analysis prior to model training.
The ultimate goal is to make our image dataset cover as diverse domains as possible, while ensuring that the data scale for each domain is sufficient to support a wide range of training recipes.

## Prior Works

This SOP builds on the [Cosmos3 Image & Video Data Curation Pipeline](https://docs.google.com/document/d/1DeVFsxs-apQV9OzBYC-mN542uRwqtfgXV0dj1Jhpbfk/edit) design, which defines the curation steps: ingestion, embedding, semantic dedup, categorization, filtering, captioning and sharding. This SOP reuses three of them:

- **Semantic dedup:** [Image Dedup Stage Overview](https://docs.google.com/document/d/10yAGK688CUZohTDU9A3RPKpLdYMbWuqoWyR11mc-xSs/edit). Image embeddings are clustered into 20K clusters from a 140M-image sample, every image is assigned to its nearest centroid, and images in the same cluster with cosine similarity above 0.96 are grouped as duplicates, keeping the highest-resolution and most central one. The [incremental dedup design](https://gitlab-master.nvidia.com/dir/imaginaire4/-/blob/main/pipelines/sila/image/dedup/INCREMENTAL_DEDUP_DESIGN.md) deduplicates new images against the already deduplicated corpus.
- **Categorization:** [Image Taxonomy Classifier](https://docs.google.com/document/d/1Qls4O4vgDP42il_kxEEv1Y6m54fCMsyHHBts9-k29Rs/edit). A hierarchical image taxonomy, labeled by a Qwen3-VL-8B classifier distilled from a frontier VLM that also outputs scene-text density, entities and semantic tags. The [Arena-Extracted Taxonomy](https://docs.google.com/document/d/1mUpTW0OVjSiuReK-qucOBplneUv2aP8nAZhzEY_xnKM/edit) is a separate, request-side taxonomy derived from arena.ai prompts and used for SDG data generation, RL prompt generation and benchmark creation.
- **Hard + soft all-in-one quality filter:** [VLM-Based Image Filtering](https://docs.google.com/document/d/1eMDoz-bKXgPp3180Bcp7AA_CE5JSCmwWAzzr7m5CDt0/edit). Our existing hard filter: a Qwen3-VL-8B model predicts an aesthetic score and binary flags for collage, NSFW, text banner, watermark and white background in a single pass, and images that fail any check are dropped.

## Overview

The proposed data onboarding SOP mainly follows the [Cosmos3 data curation pipeline](https://docs.google.com/document/d/1DeVFsxs-apQV9OzBYC-mN542uRwqtfgXV0dj1Jhpbfk/edit): semantic dedup, then categorization, then filtering. New dataset goes through similar steps, but filtering is slightly modified based on Cosmos3.5 need, splitting into a hard filter and a soft quality score:So a glance looks like:

1. **Semantic dedup:** cluster image embeddings and remove near-duplicates, both within the candidate and against the existing corpus.
2. **Categorization:** assign each image a domain label from the taxonomy. Collage is its own taxonomy category, not a filter.
3. **Hard quality filter:** drop images flagged as NSFW or watermark. Text banner and white background are not filtered; the caption describes them and the aesthetic score reflects them.
4. **Soft quality score:** keep the aesthetic score, and other continuous quality scores, as soft signals that describe image quality instead of dropping images.

## Domain × Quality Statistics

Following this pipeline, we maintain a dynamic statistic of our existing image database, can be categorized into the pretrain and midtrain we used in different model developing stages.
We run the four steps over image_meta_table_full to build a snapshot as, per category, per total count, the filtered count (after dedup and the hard filter), and show statistics on resolution and quality scores. (see demo chat below)
By doing such we are able to understand in a statistical way the current status of our image dataset.

(An example image, not the true status)
When a new dataset comes in, it goes through the same pipeline and produces the same stat report. Putting it next to the current snapshot gives a direct view of how the data distribution would change: which categories grow, how much of the new data survives dedup and the hard filter, and how the resolution and quality distributions shift.

## Quality Dimension

The soft quality scores below are mostly predicted by a VLM or using an existed low-level vision, stored as columns in the table:

| **Dimension** | **Signals** |
|---|---|
| Technical quality | blur and sharpness, compression artifacts, noise |
| Aesthetic quality | aesthetic score model (in cosmos3) |
| Photorealistic quality | how real the image looks: natural lighting and textures, plausible anatomy and geometry, no CG or AI-generated look |
| Image-text alignment | caption-image similarity, caption quality |

## Onboarding a New Dataset

A new dataset runs through the same four steps as the corpus and produces the same stat report. Whether to onboard it is then decided by how our corpus stats change before and after ingest.

### Decision rules

The decision is based on how our corpus stats change before and after ingesting the candidate, not on fixed thresholds. For each category, the report puts the two snapshots side by side:

- **Counts:** total and filtered counts before and after ingest, and how much each category grows.
- **Distributions:** resolution and quality score distributions before and after ingest, showing whether the candidate fills a gap, adds more of what we already have, or pulls quality down.

Reviewers use this before/after view to accept the dataset, keep only some of its categories, or reject it. Cases the stats cannot settle go to a proxy-model ablation on the UGB and CVTG benchmarks.

## Possible Extensions

The same stats can do more than support onboarding decisions. Once data volume is no longer the bottleneck, they let us choose training data deliberately instead of taking everything that passes the filters:

- **Stat-driven data selection:** pick the training data per category from the snapshot, for example capping categories that are already large, drawing more from thin ones, and choosing the quality score range per training stage (broad for pretraining, the top of the distribution for midtraining and SFT).
- **Testing mixtures:** turn the snapshot into a few candidate mixtures and compare them with proxy-model runs, so the choice of mixture is tested rather than assumed.
- **Gap-driven sourcing:** use categories that stay thin in the snapshot to decide what to collect or synthesize next, for example SDG targeted with the Arena-Extracted Taxonomy.
- **Tracking over time:** keep dated snapshots to see how each category and its quality distribution change as datasets are added.
