# T2I dims 标注校准报告

日期：2026-11-07。参考集：Qwen-Image-Bench 官方 1000 条（奇数 ID 500 条调参，偶数 ID 500 条 held-out）。三个模型 opus-5.5 / kimi-k3 / gpt-6-astra，温度 1.0，并发 ≤16。

## 结论

- 最终方案：**facets2**（方案 b：让每个模型从官方 56 个 L3 facet 中选 3–10 个，再由所选 facet 推出 L1），规则 **RWF = 三个模型中任一选了 RWF 类 facet（any）；CG = 三模型多数票**。
- held-out 上 RWF 预测率 51.6% vs 官方 45.8%（**+5.8，略超出 ±5 目标**），kappa 0.430；CG 预测率 64.8% vs 62.8%（+2.0，达标），kappa 0.593。
- 对比旧标注器（baseline 多数票）：RWF 82.2%（kappa 0.211）、CG 75.8%（kappa 0.581）。RWF 大幅改善，CG 的 kappa 基本不变但正例率从 +13 降到 +2。
- RWF 的 kappa 上限约 0.45–0.52：官方标签本身稀疏且不一致（见分歧样例），三个模型都做不到更高。最高 kappa 的单点（RWF 用 majority/all，kappa 0.45–0.49）正例率只有 26–34%，偏离官方 >11 个百分点，故按"kappa 最好且正例率接近官方"的约束选了 any。如果只看 kappa，CG 用 all 规则可到 0.65（调参集）但正例率偏低 8 个百分点，未采用。
- 规则（any / majority）在调参集上选定，held-out 只做一次评测，未再调整。

## 指标表

缩写：P=precision，R=recall；"预测%"为预测为正的比例。每个方案都给出三个模型单独结果和三种投票（majority≥2/3，any≥1/3，all=3/3）。

### 调参集（奇数 ID，n=500，官方 RWF 49.2% / CG 63.2%）

| 方案 | 投票 | RWF 预测% | acc | P | R | kappa | CG 预测% | acc | P | R | kappa |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | opus-5.5 | 80.2 | 0.642 | 0.584 | 0.951 | 0.291 | 74.2 | 0.806 | 0.795 | 0.934 | 0.555 |
| baseline | kimi-k3 | 68.2 | 0.694 | 0.636 | 0.882 | 0.392 | 70.0 | 0.816 | 0.820 | 0.908 | 0.589 |
| baseline | gpt-6-astra | 91.4 | 0.550 | 0.523 | 0.972 | 0.112 | 79.4 | 0.786 | 0.763 | 0.959 | 0.493 |
| baseline | vote_majority | 82.0 | 0.628 | 0.573 | 0.955 | 0.264 | 74.6 | 0.806 | 0.794 | 0.937 | 0.554 |
| baseline | vote_any | 91.6 | 0.552 | 0.524 | 0.976 | 0.116 | 80.2 | 0.786 | 0.761 | 0.965 | 0.491 |
| baseline | vote_all | 66.2 | 0.706 | 0.650 | 0.874 | 0.415 | 68.8 | 0.816 | 0.826 | 0.899 | 0.591 |
| strict | opus-5.5 | 63.0 | 0.718 | 0.667 | 0.854 | 0.438 | 69.8 | 0.818 | 0.822 | 0.908 | 0.594 |
| strict | kimi-k3 | 40.8 | 0.748 | 0.794 | 0.659 | 0.495 | 61.4 | 0.786 | 0.840 | 0.816 | 0.545 |
| strict | gpt-6-astra | 74.2 | 0.622 | 0.577 | 0.870 | 0.250 | 71.6 | 0.776 | 0.785 | 0.889 | 0.494 |
| strict | vote_majority | 62.0 | 0.708 | 0.661 | 0.833 | 0.418 | 68.0 | 0.800 | 0.818 | 0.880 | 0.558 |
| strict | vote_any | 76.8 | 0.628 | 0.578 | 0.902 | 0.262 | 73.8 | 0.794 | 0.789 | 0.921 | 0.529 |
| strict | vote_all | 39.2 | 0.752 | 0.811 | 0.646 | 0.502 | 61.0 | 0.786 | 0.843 | 0.813 | 0.546 |
| facets | opus-5.5 | 51.6 | 0.748 | 0.733 | 0.768 | 0.496 | 72.0 | 0.804 | 0.803 | 0.915 | 0.556 |
| facets | kimi-k3 | 35.0 | 0.742 | 0.834 | 0.593 | 0.482 | 61.2 | 0.796 | 0.850 | 0.823 | 0.566 |
| facets | gpt-6-astra | 29.8 | 0.742 | 0.893 | 0.541 | 0.481 | 66.2 | 0.794 | 0.822 | 0.861 | 0.549 |
| facets | vote_majority | 35.6 | 0.748 | 0.837 | 0.606 | 0.494 | 67.6 | 0.808 | 0.825 | 0.883 | 0.577 |
| facets | vote_any | 52.8 | 0.744 | 0.723 | 0.776 | 0.488 | 73.2 | 0.796 | 0.792 | 0.918 | 0.535 |
| facets | vote_all | 28.0 | 0.740 | 0.914 | 0.520 | 0.476 | 58.6 | 0.790 | 0.860 | 0.797 | 0.560 |
| facets2 | opus-5.5 | 50.8 | 0.764 | 0.752 | 0.776 | 0.528 | 67.4 | 0.826 | 0.840 | 0.896 | 0.617 |
| facets2 | kimi-k3 | 35.2 | 0.720 | 0.801 | 0.573 | 0.437 | 59.0 | 0.822 | 0.885 | 0.826 | 0.626 |
| facets2 | gpt-6-astra | 29.0 | 0.730 | 0.883 | 0.520 | 0.456 | 64.8 | 0.796 | 0.830 | 0.851 | 0.557 |
| facets2 | vote_majority | 33.8 | 0.726 | 0.822 | 0.565 | 0.449 | 64.8 | 0.808 | 0.840 | 0.861 | 0.583 |
| facets2 | vote_any | 52.6 | 0.758 | 0.738 | 0.789 | 0.516 | 71.2 | 0.808 | 0.809 | 0.911 | 0.568 |
| facets2 | vote_all | 28.6 | 0.730 | 0.888 | 0.516 | 0.456 | 55.2 | 0.828 | 0.917 | 0.801 | 0.646 |
### Held-out（偶数 ID，n=500，官方 RWF 45.8% / CG 62.8%）

| 方案 | 投票 | RWF 预测% | acc | P | R | kappa | CG 预测% | acc | P | R | kappa |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | opus-5.5 | 80.6 | 0.592 | 0.531 | 0.934 | 0.224 | 75.4 | 0.822 | 0.798 | 0.959 | 0.591 |
| baseline | kimi-k3 | 70.6 | 0.644 | 0.572 | 0.882 | 0.312 | 71.8 | 0.830 | 0.819 | 0.936 | 0.617 |
| baseline | gpt-6-astra | 92.2 | 0.512 | 0.484 | 0.974 | 0.089 | 80.8 | 0.796 | 0.762 | 0.981 | 0.516 |
| baseline | vote_majority | 82.2 | 0.584 | 0.526 | 0.943 | 0.211 | 75.8 | 0.818 | 0.794 | 0.959 | 0.581 |
| baseline | vote_any | 92.8 | 0.506 | 0.481 | 0.974 | 0.078 | 81.6 | 0.788 | 0.755 | 0.981 | 0.494 |
| baseline | vote_all | 68.4 | 0.658 | 0.585 | 0.873 | 0.337 | 70.6 | 0.842 | 0.833 | 0.936 | 0.647 |
| strict | opus-5.5 | 66.4 | 0.650 | 0.581 | 0.843 | 0.319 | 71.0 | 0.826 | 0.820 | 0.927 | 0.610 |
| strict | kimi-k3 | 42.2 | 0.712 | 0.701 | 0.646 | 0.416 | 59.4 | 0.790 | 0.852 | 0.806 | 0.559 |
| strict | gpt-6-astra | 72.0 | 0.610 | 0.547 | 0.860 | 0.248 | 72.8 | 0.772 | 0.775 | 0.898 | 0.484 |
| strict | vote_majority | 63.6 | 0.654 | 0.588 | 0.817 | 0.323 | 68.8 | 0.812 | 0.820 | 0.898 | 0.584 |
| strict | vote_any | 75.0 | 0.604 | 0.541 | 0.886 | 0.240 | 76.0 | 0.796 | 0.779 | 0.943 | 0.529 |
| strict | vote_all | 42.0 | 0.714 | 0.705 | 0.646 | 0.420 | 58.4 | 0.780 | 0.849 | 0.790 | 0.540 |
| facets | opus-5.5 | 50.8 | 0.730 | 0.685 | 0.760 | 0.461 | 73.6 | 0.812 | 0.799 | 0.936 | 0.572 |
| facets | kimi-k3 | 34.4 | 0.738 | 0.785 | 0.590 | 0.462 | 59.4 | 0.806 | 0.865 | 0.818 | 0.592 |
| facets | gpt-6-astra | 27.4 | 0.724 | 0.832 | 0.498 | 0.426 | 66.2 | 0.802 | 0.825 | 0.869 | 0.568 |
| facets | vote_majority | 33.6 | 0.754 | 0.815 | 0.598 | 0.494 | 67.6 | 0.808 | 0.822 | 0.885 | 0.578 |
| facets | vote_any | 52.4 | 0.718 | 0.668 | 0.764 | 0.438 | 74.6 | 0.814 | 0.796 | 0.946 | 0.574 |
| facets | vote_all | 26.6 | 0.720 | 0.835 | 0.485 | 0.417 | 57.0 | 0.798 | 0.874 | 0.793 | 0.581 |
| facets2 | opus-5.5 | 50.4 | 0.722 | 0.679 | 0.747 | 0.444 | 69.8 | 0.814 | 0.817 | 0.908 | 0.586 |
| facets2 | kimi-k3 | 34.6 | 0.732 | 0.775 | 0.585 | 0.450 | 60.0 | 0.808 | 0.863 | 0.825 | 0.595 |
| facets2 | gpt-6-astra | 27.0 | 0.720 | 0.830 | 0.489 | 0.417 | 64.0 | 0.804 | 0.838 | 0.854 | 0.578 |
| facets2 | vote_majority | 34.4 | 0.734 | 0.779 | 0.585 | 0.454 | 64.8 | 0.812 | 0.840 | 0.866 | 0.593 |
| facets2 | vote_any | 51.6 | 0.714 | 0.667 | 0.751 | 0.430 | 73.0 | 0.826 | 0.811 | 0.943 | 0.606 |
| facets2 | vote_all | 26.0 | 0.726 | 0.854 | 0.485 | 0.429 | 56.0 | 0.788 | 0.871 | 0.777 | 0.563 |

方案说明：
- baseline：原 DIMS_SYSTEM_PROMPT（二分类，宽定义）。
- strict（方案 a）：收紧二分类定义，默认 false，只在 prompt 含明确触发条件时才 true。
- facets（方案 b）：官方流程模仿，选 3–10 个 facet 再推 L1。
- facets2：facets + 针对调参集上 facet 级误报（Text/Cross-lingual、Shot Sizes/Angles/Composition/Camera、Imagination/Feature Matching/Logical Resolution）加的三条通用限定规则。prompt 中没有任何参考集原文。

## 最终 prompt 全文（system，facets2）

user 消息格式：`Prompt: {prompt}`。

```
You prepare the evaluation checklist for a text-to-image judge. The judge scores a generated image against a fixed taxonomy of 56 facets (L1 / L2 / L3). Given only the prompt, choose which facets are actually examined by this prompt, then the judge scores only the L1 groups that your chosen facets belong to.

Taxonomy:
[Quality]
- Quality / Realism / Physical Logic: adherence to physical laws (gravity, reflection, shadow direction, object stability)
- Quality / Realism / Material Texture: realistic surface texture of materials (skin, fabric, metal, wood)
- Quality / Detail / Noise: rich detail without noise or unnatural smoothing
- Quality / Detail / Edge Clarity: sharp, well-defined edges and outlines
- Quality / Detail / Naturalness: no artificial plastic or greasy AI look
- Quality / Resolution / Resolution: high-definition, free from pixelation or compression artifacts
[Aesthetics]
- Aesthetics / Composition / Composition: balanced, visually guided, pleasing composition
- Aesthetics / Color Harmony / Color Harmony: harmonious, cohesive palette fitting the mood
- Aesthetics / Lighting / Lighting & Atmosphere: light and shadow atmosphere matching the scene
- Aesthetics / Anatomical Portraiture / Anatomical Fidelity: correct facial proportions, skeleton, limbs, realistic skin micro-texture
- Aesthetics / Emotional Expression / Emotional Expression: image tone conveys the intended emotion or mood
- Aesthetics / Style Control / Style Control: accurately captures a requested artistic style
[Alignment]
- Alignment / Attributes / Quantity: number of objects matches the prompt
- Alignment / Attributes / Facial Expression: face of a person or animal shows the specified emotion
- Alignment / Attributes / Material Properties: object materials match the description
- Alignment / Attributes / Color: object colors match the specification
- Alignment / Attributes / Shape: object shapes match the description
- Alignment / Attributes / Size: object sizes match the specification
- Alignment / Actions / Contact Interaction: physical contact between subjects
- Alignment / Actions / Non-contact Interaction: non-contact spatial or social relationship between subjects
- Alignment / Actions / Full-body Action: posture and body action of a person or animal performing an activity
- Alignment / Layout / 2D Space: left/right, top/bottom, foreground/background positions
- Alignment / Layout / 3D Space: 3D layout, occlusion, relative position in depth
- Alignment / Relations / Composition Relationship: integrating multiple elements into a coherent whole
- Alignment / Relations / Difference/Similarity: specified differences or similarities between objects
- Alignment / Relations / Containment: containment or enclosure relationships
- Alignment / Scene / Real-world Scene: scene type and environment (office, forest, street)
- Alignment / Scene / Virtual Scene: internal consistency of a fictional or fantasy scene
[Real-world Fidelity]
- Real-world Fidelity / Fairness / Social Bias: avoid tying genders to particular professions or settings
- Real-world Fidelity / Fairness / Cultural Fairness: avoid stereotypical portrayal by region, race or culture
- Real-world Fidelity / Safety & Compliance / Safety & Compliance: avoid prohibited content (pornography, violence, hate symbols)
- Real-world Fidelity / World Knowledge / Animals: real animals with anatomically accurate features and biology
- Real-world Fidelity / World Knowledge / Objects: typical appearance, structure, brand logo or iconic look of real-world items
- Real-world Fidelity / World Knowledge / Information Visualization: translating an abstract or scientific concept into an understandable visual
- Real-world Fidelity / World Knowledge / Temporal Characteristics: iconic elements of a specific historical period (technology, clothing, architecture)
- Real-world Fidelity / World Knowledge / Cultural Elements: symbols, traditional clothing, rituals, customs depicted accurately
[Creative Generation]
- Creative Generation / Imagination / Imagination: creative, imaginative combination of novel or surreal elements
- Creative Generation / Feature Matching / Feature Matching: seamless fusion of multiple elements
- Creative Generation / Logical Resolution / Logical Resolution: causal relationships between events (breaking glass, shards flying)
- Creative Generation / Text Rendering / Text Accuracy: text in the image is clear, legible, spelled correctly
- Creative Generation / Text Rendering / Text Layout: layout of text (alignment, spacing, margins)
- Creative Generation / Text Rendering / Font: a specified font or typeface
- Creative Generation / Text Rendering / Cross-lingual Generation: translation or text in another language
- Creative Generation / Design Applications / Graphic Design: advertisement, poster, information hierarchy
- Creative Generation / Design Applications / Product Design: industrial design logic of a product
- Creative Generation / Design Applications / Spatial Design: interior or architectural space design
- Creative Generation / Design Applications / Fashion Styling: clothing cut, silhouette, makeup style
- Creative Generation / Design Applications / Game Design: game props and UI elements
- Creative Generation / Design Applications / Art Design: a specific artistic design style
- Creative Generation / Visual Storytelling / Cinematic Style: a named director's visual language
- Creative Generation / Visual Storytelling / Camera / Lens Style: specific photographic equipment or lens look
- Creative Generation / Visual Storytelling / Storyboard Creation: panel layout (multi-panel, split-screen)
- Creative Generation / Visual Storytelling / Shot Sizes: close-up, medium shot, wide shot
- Creative Generation / Visual Storytelling / Composition: composition rules (thirds, golden ratio, leading lines)
- Creative Generation / Visual Storytelling / Angles: bird's-eye, low angle, Dutch angle
- Creative Generation / Visual Storytelling / Comic Creation: comic style (American comics, manga, BD)

Rules:
- Choose between 3 and 10 facets, usually around 5. Pick a facet only when the prompt explicitly asks for, describes or strongly depends on what that facet checks; do not add facets just because they could loosely apply.
- Quality, Aesthetics and Alignment facets are about the image itself; pick the ones that this prompt visibly stresses.
- Real-world Fidelity facets are chosen only when judging the image needs outside real-world knowledge or norms: a real species, a real branded or iconic object, a specific culture or historical era, a concept or infographic to be visualized correctly, a person whose profession or origin could invite stereotypes, or safety-sensitive content. Generic scenes, generic people and generic objects do not qualify.
- Creative Generation facets are chosen only when the prompt contains a creative or professional-design demand: text to be rendered, a named font or translation, an ad/poster/product/interior/fashion/game/art-style design task, cinematic or camera instructions (director, lens, storyboard, comic, shot size, angle, composition rule), or surreal, fused or cause-and-effect content. Realistic scenes with rich detail do not qualify.
- Text Accuracy, Text Layout, Font and Cross-lingual Generation are chosen only when words, letters or numbers must appear as visible text in the image; Cross-lingual Generation additionally needs a translation request or text in a non-English script.
- Shot Sizes, Angles, Composition and Camera / Lens Style are chosen only when the framing, viewpoint, composition rule or equipment is itself a stated requirement of the prompt; an incidental word such as close-up or wide in an ordinary scene description is not enough.
- Feature Matching, Imagination and Logical Resolution are chosen only for clearly fused, surreal or cause-and-effect content.
- Use the exact facet names as written above.

Respond with a JSON object only: {"facets": ["<L1> / <L2> / <L3>", ...]}
```

投票：每个模型给出 facet 列表（解析时校验必须是 56 个官方名称），由列表推出 RWF/CG 布尔；RWF 任一模型为真即为真，CG 需 ≥2 个模型。`dims_facets` 存被 ≥2 个模型同时选中的 facet。

## 两个 blend 重标后的 RWF/CG 比例

| blend | 条数 | RWF 旧→新 | CG 旧→新 | 官方参考 |
|---|---|---|---|---|
| arena1000 | 1000 | 82.5% → 54.1% | 78.2% → 61.0% | RWF 47.5% / CG 63% |
| cosmos500 | 500 | 96.8% → 74.6% | 52.4% → 37.8% | 同上 |

arena1000 已接近官方。cosmos500 的 RWF 仍偏高（74.6%）、CG 偏低（37.8%）：该 blend 的分类体系本身就偏向真实世界对象（动物、物品、文化等），这是内容分布造成的，不是标注器宽松；未针对它再调参。已验证：除 dims / dims_votes / dims_facets 外，两个文件的所有字段与重标前完全一致，prompt 未改动。备份：`*.pre_dims_calib.json`。

## 10 条分歧样例（held-out，随机抽取，最终标注器 vs 官方）

1. **RWF 误报（我们标 RWF，官方没有）** · ID 308
   - prompt: Design a full-body render of the “Lion Dance New Year” MOBA skin: wearable animated lion helmet, red-gold Tang-style outfit with modern tailoring, cloud-pattern embroidery on hem, holding drumsticks. Background: fireworks and red lanterns. Festive yet combat-r…
   - 官方 RWF facet: []；官方 CG facet: ['Game Design', 'Imagination', 'Feature Matching']
   - 本方案选中的 RWF/CG facet: ['Cultural Elements', 'Fashion Styling', 'Game Design']
   - 分析: "舞狮/唐装"明显涉及文化元素，我们选了 Cultural Elements，官方只挑了 CG 类 facet，属官方标注稀疏。
2. **RWF 误报（我们标 RWF，官方没有）** · ID 716
   - prompt: An audio guide touchscreen at Borobudur Temple, Yogyakarta: the interface shows Javanese “Sugeng enjing, sedoyo tamu!” (Good morning, honored guests!) in vertical Hanacaraka script with vine-like curves; Latin transliteration and Indonesian translation appear …
   - 官方 RWF facet: []；官方 CG facet: ['Cross-lingual Generation', 'Text Accuracy']
   - 本方案选中的 RWF/CG facet: ['Cultural Elements', 'Text Accuracy', 'Text Layout', 'Font', 'Cross-lingual Generation']
   - 分析: 爪哇文字+翻译，官方只挑了 CG 的文字类 facet，没挑 Cultural Elements；模型认为涉及文化符号。
3. **RWF 误报（我们标 RWF，官方没有）** · ID 672
   - prompt: An unfinished oil painting of a sunny beach sits on the easel, yet the blue paint on the artist’s palette has dried and cracked.…
   - 官方 RWF facet: []；官方 CG facet: ['Logical Resolution']
   - 本方案选中的 RWF/CG facet: ['Logical Resolution']
   - 分析: 官方只标 Logical Resolution；本例的 RWF 正例来自 any 规则（至少一个模型挑了 RWF 类 facet，具体 facet 不在最终 ≥2 票的列表中）。这是 any 规则带来的典型误报。
4. **RWF 漏报（官方有，我们没有）** · ID 266
   - prompt: Skill info card UI: skill name, cooldown, damage numbers (fictional) with icon; bilingual. Require proper numeric/unit formatting; readable aligned text; clear hierarchy for mobile UI.…
   - 官方 RWF facet: ['Information Visualization']；官方 CG facet: ['Game Design', 'Text Layout', 'Graphic Design']
   - 本方案选中的 RWF/CG facet: ['Text Accuracy', 'Text Layout', 'Cross-lingual Generation', 'Graphic Design', 'Game Design']
   - 分析: 官方把虚构数值的信息卡视为 Information Visualization；我们判为纯 UI/设计 CG。官方此类标注不稳定。
5. **RWF 漏报（官方有，我们没有）** · ID 566
   - prompt: A mechanical temple dedicated to the God of Gears; an eternal-motion core hums on the altar, monks clad in copper-foil robes chant prayers daily as they oil their clockwork hearts.…
   - 官方 RWF facet: ['Cultural Elements']；官方 CG facet: []
   - 本方案选中的 RWF/CG facet: ['Imagination', 'Feature Matching']
   - 分析: 官方把"神庙/僧侣祈祷"归为 Cultural Elements；模型认为是虚构的机械神庙，只给 CG（Imagination）。
6. **RWF 漏报（官方有，我们没有）** · ID 798
   - prompt: Weightlifter at the moment of a successful jerk: elbows locked, traps raised, belt pressure marks realistic. Background wall shows an event sign “热身区 Warm-up Zone” with arrow and numbering aligned to a grid and clean whitespace. Requirements: chalk particles v…
   - 官方 RWF facet: ['Objects']；官方 CG facet: ['Text Accuracy']
   - 本方案选中的 RWF/CG facet: ['Text Accuracy', 'Text Layout', 'Cross-lingual Generation']
   - 分析: 官方挑了 Objects（举重器械/腰带）；模型只看到文字叠加，未将通用器械视为需真实性核对的对象。
7. **CG 误报** · ID 454
   - prompt: A butterfly emerges from its chrysalis—wings wet and hanging, waiting to expand—then hemolymph is pumped into wing veins to stiffen them; process delicate yet orderly.…
   - 官方 RWF facet: ['Animals']；官方 CG facet: []
   - 本方案选中的 RWF/CG facet: ['Animals', 'Information Visualization', 'Logical Resolution']
   - 分析: 蝴蝶羽化过程：模型把过程当作 Logical Resolution（因果）；官方只当 Animals。
8. **CG 误报** · ID 574
   - prompt: In the ruins of code, data ghosts wander; fragmented programs project ghostly ad remnants onto broken walls, and hackers steal buried encrypted emotion-packs via neural jacks.…
   - 官方 RWF facet: []；官方 CG facet: []
   - 本方案选中的 RWF/CG facet: ['Imagination']
   - 分析: 赛博朋克幻想场景，模型判 Imagination，官方未标 CG。官方对幻想类 prompt 常不标 CG。
9. **CG 漏报** · ID 950
   - prompt: Interior of dragon's lair: gold and silver coins piled into mountains, their metallic surfaces shimmer and flicker with firelight; a massive dragon coils at the center of the treasure mound, scales showing dark red metallic texture, each scale clearly distingu…
   - 官方 RWF facet: []；官方 CG facet: ['Game Design']
   - 本方案选中的 RWF/CG facet: []
   - 分析: 龙穴场景官方标了 Game Design；模型认为是普通幻想场景，无设计交付物。
10. **CG 漏报** · ID 108
   - prompt: A portrait in the style of classical oil painting: A 17th-century noblewoman sits sideways against a dark brown background. Her pearl earrings reflect the highlights of the window light. Her lace collar is intricately layered with delicate stitches. Her skin i…
   - 官方 RWF facet: ['Temporal Characteristics']；官方 CG facet: ['Art Design']
   - 本方案选中的 RWF/CG facet: ['Temporal Characteristics']
   - 分析: 古典油画风格人像：官方标 Art Design（风格要求），模型只认为是 Style Control（Aesthetics），未给 CG。

总体观察：（1）官方 RWF/CG 标注常"只挑部分 facet"，同类 prompt 标注不一致（例如文化/文字类 prompt 有时标有时不标），这是 kappa 上限不高的主因；（2）我们在含文字、文化元素的 prompt 上偏向标正，在泛化的幻想/风格类 prompt 上偏向标负。

## 代码改动

- 新增 `scripts/t2i_eval/dims_labeler.py`：56 facet 层级、baseline/strict/facets/facets2 四种 prompt、`DimsLabeler`（带磁盘缓存、失败重试、可复投票，`label()` 返回 dims / dims_votes / dims_facets）、`FINAL_RULES`。
- 新增 `scripts/t2i_eval/calibrate_dims.py`：按奇偶切分，输出每模型与 majority/any/all 投票的正例率、acc、precision、recall、kappa；缓存 `scripts/t2i_eval/dims_calib_cache.jsonl`。
- 修改 `scripts/t2i_eval/build_blend_prompts.py`：删除 DIMS_SYSTEM_PROMPT / DIMS_USER_TEMPLATE / BASE_DIMS 与旧 label_dims 实现，label_dims 改用 `DimsLabeler(llm=self.vlm)`；输出增加 `dims_facets`；去重重写时一并清除 `dims_facets`。blend_agent 会话已结束后才修改，`py_compile` 与 `-h` 均通过，未实际重跑生成流程。
- 数据：`scripts/t2i_eval/blend/{arena1000,cosmos500}.json` 已重标；重标缓存 `blend/dims_relabel_cache.jsonl`；校准缓存 `blend/dims_calib_cache.jsonl`。
- 未 git add/commit，未提交集群任务，未上传 s3，未改 imaginaire4。
