<!-- synced from https://docs.google.com/document/d/1mUpTW0OVjSiuReK-qucOBplneUv2aP8nAZhzEY_xnKM/edit revision AHj4eMSW0rpQaU1S-lFID-WXeYeirwOUwP0hWssS7woiEtRbJoPvprFYEI8ZQWGJ2_jMLxXU_fxrNoa79YvhFu6QbxxDyrtoj1pfxDeD9dc -->

## Design Doc: Arena-Extracted Taxonomy

**Author(s)**: @Xingqian Xu US @Runlong Liao US
**Contributors**:
**Created**: Oct 8, 2026
**Status**:

| **Reviewer** | **Last Updated** | **Status** |
|---|---|---|
| @Runlong Liao US |  |  |
| @Yogesh Balaji US |  |  |

## 1. Motivation

The Arena-extracted taxonomy is a hierarchy of text-to-image request categories derived from real user prompts on arena.ai. It describes what users ask image generation models to do. We use it as one shared category system across several workstreams, so that synthetic training data, RL prompts and benchmarks are built against the same distribution. The subsections below describe each use; the categories in sections 3 and 5 are shared by all of them.

### RL prompt generation

We need a high quality RL prompt so that within about 500 steps we push our cosmos3/cosmos3.5 model to maximize the performance gain.

Plan:

- 300 rounds of RL, LoRA
- For each round, we have 96 prompts and 12 rollouts for each prompt

### SDG data generation

Synthetic data generation (SDG) uses the taxonomy to decide what to generate. Prompts are written or sampled per category, rendered with frontier image models, filtered, and added to training data with their category label attached. This fills categories that users ask for but that are thin in our pretraining and midtraining data, and keeps the synthetic mix measurable by category.

### Benchmark creation

Benchmarks sample evaluation prompts per category, so that scores reflect real user demand and can be reported per category rather than as a single number. The taxonomy can also show which categories an existing benchmark covers and which it misses.

## 2. Affecting Factors

The first factor applies to RL prompt generation; the other three apply to every use of the taxonomy.

1. We want our prompt to produce diverse output, a group of good and bad, so that we can find meaningful direction to push our model into.
2. Reflecting true user prompt distribution. (A rough guess is a balanced sample of pretraining data)
3. Reflecting leaderboard testing domain.
4. Reflecting our knowledge that frontier T2I model should handle well.

## 3. Category and Ratio (a heuristic ratio)

The ratio below is the default for RL prompts. Other uses can reweight it, for example oversampling weak categories for SDG or weighting categories evenly for a benchmark.

1. Photorealistic & Cinematic          25%
2. Portraits & Humans                  12%
3. Product / Branding / Commercial     15%
4. Text / Typography / Layout          12%
5. Cartoon / Anime / Fantasy           10%
6. Art / Illustration                   8%
7. 3D / Assets / Render                 6%
8. Complex Composition & Reasoning      8%
9. Frontier / Stress Test               4%

## 4. Method of obtaining prompts:

We have target three paths, maybe a few more options:

- Obtain native [arena.ai](http://arena.ai) prompts directly from the user. This is closer to the true user prompt distribution but it can be a really bad short prompt.
- We use Session3’s category and weighing, try best generate long dense 50k prompt match the topic using frontier VLM model. (@Runlong Liao US worry about diversity)
- We use Session3’s category, fill the category prompt by extract pretrain training prompt and match the weighting. (@Xingqian Xu US worry about memorization)

## 5. Detail Taxonomy

Graphic Design & Typography:
  Social Media & Marketing:
  Logos & Brand Identity:
  Print & Editorial:
  Posters:
  Apparel & Merchandise:
  Typography:
  Small Business Print:
  Patterns & Surface:
People & Portraits:
  Glamour & Beauty Portraits:
  Celebrities & Public Figures:
  Portrait Photography:
  People in Action:
  Body & Detail:
  Painted & Drawn Portraits:
Characters & Creatures:
  Fan Art of Existing IP:
  Anime & Manga Originals:
  Cartoon & Mascots:
  Character Design:
  Toys & Figures:
Art Styles & Media:
  Comics & Animation:
  Digital & 3D:
  Art Movements:
  Photography Styles:
  Craft & Mixed Media:
  Painting:
  Drawing & Print:
Abstract & Conceptual:
  Conceptual:
  Abstract:
  Backgrounds & Textures:
Products & Commercial:
  Product Photography:
  Advertising:
  3D Assets & Printables:
  Packaging & Branding:
Infographics & Education:
  Infographics:
  Educational:
  Slides & Worksheets:
  Maps:
Emotions & Storytelling:
  Cinematic:
  Moods:
  Narrative Moments:
Fantasy & Mythology:
  Characters:
  Mythology:
  Creatures:
  Fantasy Worlds:
UI & Digital:
  Digital Assets:
  Interfaces:
  Screens in Context:
Architecture & Interiors:
  Interiors:
  Architectural Visualization:
  Buildings:
Kids & Family:
  Coloring & Activity Pages:
  Children's Illustration:
  Childhood:
  Family:
Objects & Still Life:
  Still Life:
  Everyday Objects:
  Macro & Material:
Animals & Wildlife:
  Pets:
  Wild Mammals:
  Reptiles, Amphibians & Insects:
  Farm & Domestic Animals:
  Animal Art:
  Birds:
Fashion & Beauty:
  Fashion Photography:
  Accessories & Details:
  Beauty & Grooming:
  Fashion Illustration:
Food & Drink:
  Food Styles:
  Dishes:
  Ingredients & Kitchen:
  Baking & Desserts:
  Drinks:
Culture & Traditions:
  Devotional & Religious Art:
  Festivals:
  Religion & Spirituality:
  Traditional Arts:
Work & Professions:
  Professional & Service:
  Offices & Workspaces:
  Trades & Crafts:
Vehicles & Transportation:
  Cars:
  Vehicle Art:
  Rail & Public Transit:
  Two Wheels:
  Air & Sea:
Science Fiction & Futurism:
  Sci-Fi Scenes:
  Future Cities:
  Sci-Fi Tech:
Sports & Fitness:
  Individual Sports:
  Outdoor & Extreme:
  Sports Graphics:
  Team Sports:
  Fitness & Wellness:
History & Eras:
  Ancient World:
  Medieval to Early Modern:
  Modern History:
  Historical Styles:
Nature & Landscapes:
  Aerial & Grand Views:
  Water:
  Mountains & Highlands:
  Deserts & Plains:
  Forests & Jungles:
  Seasons:
Weather, Sky & Space:
  Space & Astronomy:
  Weather:
  Sky Phenomena:
Cities & Street Scenes:
  Urban Scenes:
  Street Details:
  Small Towns & Villages:
  Urban Styles:
Technology & Science:
  Medicine & Health:
  Technology:
  Science:
Daily Life & Home:
  Domestic Moments:
  Leisure & Hobbies:
  Errands & Neighborhood:
Plants & Gardens:
  Botanical Detail:
  Flowers:
  Trees & Forests:
  Gardens & Cultivation:
Text in Scenes & Signage:
  Signage:
  Text Objects:
Industry & Infrastructure:
  Industry:
  Infrastructure:
  Agriculture:
Games & Play:
  Video Games:
  Tabletop:
  Toys:
Music & Performing Arts:
  Stage:
  Music:
Underwater & Marine:
  Maritime:
  Underwater Life:
Events & Celebrations:
  Celebrations:
  Gatherings:
