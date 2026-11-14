# QImageBench blend IP / 真人筛查（2026-11-07）

筛查对象：`arena1000.json`、`cosmos500.json` 全部 1500 条英文 `prompt`。
方法：Opus 5.5（temperature 0）逐条判断 prompt 是否点名或明确指向：
- **真人**：明星、名人、政治人物、历史人物
- **版权 IP**：动漫 / 游戏 / 影视 / 漫画 / 小说的角色、作品、专有地点和道具
- **真实品牌**：真实公司、商标、产品

泛指（"an actress"、"a superhero"）、虚构品牌名、神话民间人物、地标建筑不算。1500 条全部判完，无解析失败。

## 结论

| blend | 命中条数 | 真人 | 版权 IP | 真实品牌 |
|---|---|---|---|---|
| arena1000 | **58**（5.8%） | 17 | 41 | 0 |
| cosmos500 | **0** | 0 | 0 | 0 |

- cosmos500 干净，不用动。
- arena1000 的命中几乎全部来自 taxonomy 的两个类别，是按分类"如实"写出来的，不是写 prompt 时跑偏：

| arena taxonomy 类别 | 该类条数 | 命中 |
|---|---|---|
| Characters & Creatures > Fan Art of Existing IP | 44 | 41 |
| People & Portraits > Celebrities & Public Figures | 42 | 16 |
| Infographics & Education > Infographics | — | 1（思维导图里写了 Aristotle / Mill / Kant） |

- Celebrities 类另外 26 条没有点名（例如 "an actress in a trench coat"、"a middle-aged female politician"），内容上没问题，但类别本身就是"名人"。
- Logos & Brand Identity 51 条都是虚构品牌名，没有真实商标。
- GPT-image 被安全系统拒绝的 16 条里，有 9 条在此名单内；另外 7 条是身体暴露 / 儿童相关（0163、0256、0260、0304、0308、0321、cosmos500_0238），不属于 IP 问题。

## 命中明细（arena1000）

### 真人（17）

| id | 子类 | 点名 |
|---|---|---|
| arena1000_0212 | Athletes: Casual Off Court Life | Yao Ming |
| arena1000_0215 | Film Stars: Humorous And Absurd Scenarios | Keanu Reeves |
| arena1000_0225 | Intellectuals: Formal Historical Portraits | Robert Burns |
| arena1000_0234 | Film Stars: Humorous And Absurd Scenarios | Keanu Reeves |
| arena1000_0236 | Politicians: Historical And Early Career | Barack / Michelle / Malia / Sasha Obama |
| arena1000_0242 | Film Stars: Magazine And Fashion Shoots | Lupita Nyong'o |
| arena1000_0243 | Athletes: Humorous And Absurd Scenarios | Usain Bolt |
| arena1000_0269 | Athletes: Victory Celebrations And Trophies | Miroslav Klose |
| arena1000_0288 | Historical Leaders: Grand Royal Portraits | Elizabeth I |
| arena1000_0295 | Politicians: Historical And Early Career | Nelson Mandela |
| arena1000_0302 | Musicians: Live Stage Performances | Slash, Keith Richards |
| arena1000_0307 | Musicians: Red Carpet And Awards | BLACKPINK（Jisoo, Jennie, Rosé, Lisa） |
| arena1000_0311 | Musicians: Theatrical And Video Concepts | Beyoncé |
| arena1000_0326 | Intellectuals: Working In Studies And Field | Charles Darwin |
| arena1000_0327 | Film Stars: Formal Studio Portraits | Audrey Hepburn |
| arena1000_0329 | Intellectuals: Working In Studies And Field | David Attenborough |
| arena1000_0576 | Concept Maps And Mind Maps | Aristotle, John Stuart Mill, Immanuel Kant |

其中 Robert Burns、Elizabeth I、Charles Darwin、Aristotle / Mill / Kant 是去世已久的历史人物（6 条），其余 11 条是在世或近代明星、运动员、政治人物。

### 版权 IP（41）

| id | 子类 | 点名 |
|---|---|---|
| arena1000_0339 | Animation: Cosplay And Styles | Son Goku |
| arena1000_0343 | Anime Manga: Group And Story Scenes | Naruto 第七班 |
| arena1000_0344 | Animation: Action Scenes | Tom and Jerry |
| arena1000_0346 | Anime Manga: Cosplay And Styles | Luffy（One Piece） |
| arena1000_0348 | Live Action: Portraits And Scenes | Aragorn（The Lord of the Rings） |
| arena1000_0352 | Live Action: Alternate Outfits | Walter White（Breaking Bad） |
| arena1000_0353 | Video Games: Crossover Fan Art | Mario, Sonic |
| arena1000_0354 | Anime Manga: Cosplay And Styles | Luffy |
| arena1000_0358 | Video Games: Cosplay And Styles | Link, Master Sword |
| arena1000_0362 | Video Games: Group And Story Scenes | Mario, Wario, Mushroom Kingdom |
| arena1000_0365 | Animation: Alternate Outfits | Elsa（Frozen） |
| arena1000_0369 | Comics: Action Scenes | Superman |
| arena1000_0371 | Video Games: Cosplay And Styles | Link, Bokoblin |
| arena1000_0372 | Anime Manga: Solo Portraits | Raphtalia（The Rising of the Shield Hero） |
| arena1000_0373 | Anime Manga: Solo Portraits | Naruto Uzumaki |
| arena1000_0376 | Live Action: Crossover Fan Art | Aragorn, Geralt（The Witcher） |
| arena1000_0378 | Comics: Crossover Fan Art | Wonder Woman, Thanos |
| arena1000_0382 | Animation: Alternate Outfits | Elsa |
| arena1000_0383 | Anime Manga: Scenery And Locations | Doraemon |
| arena1000_0384 | Video Games: Group And Story Scenes | Link, Zelda, Ganondorf |
| arena1000_0389 | Tabletop And Books: Cosplay And Styles | Dungeons & Dragons |
| arena1000_0390 | Anime Manga: Scenery And Locations | Spirited Away（Chihiro, No-Face） |
| arena1000_0392 | Live Action: Cosplay And Styles | Daphne Bridgerton |
| arena1000_0396 | Animation: Crossover Fan Art | Totoro, SpongeBob |
| arena1000_0397 | Video Games: Action Scenes | Ryu, Zangief（Street Fighter） |
| arena1000_0398 | Comics: Crossover Fan Art | Godzilla, Bowser |
| arena1000_0405 | Tabletop And Books: Portraits And Scenes | The Fellowship of the Ring 全员 |
| arena1000_0406 | Video Games: Cosplay And Styles | Link, Master Sword, Hylian Shield |
| arena1000_0407 | Live Action: Scenery And Locations | Echo Base, Hoth（Star Wars） |
| arena1000_0408 | Live Action: Alternate Outfits | Indiana Jones |
| arena1000_0410 | Live Action: Scenery And Locations | Hogwarts |
| arena1000_0413 | Comics: Portraits And Scenes | Avengers（Spider-Man, Captain America, Thor） |
| arena1000_0414 | Live Action: Crossover Fan Art | Doc Brown, Mary Poppins |
| arena1000_0415 | Anime Manga: Action Scenes | Goku（Dragon Ball） |
| arena1000_0416 | Comics: Crossover Fan Art | Superman, Spider-Man |
| arena1000_0418 | Live Action: Cosplay And Styles | Harry Potter, Gryffindor |
| arena1000_0419 | Animation: Portraits And Scenes | Winnie the Pooh, Piglet |
| arena1000_0421 | Comics: Portraits And Scenes | Spider-Man |
| arena1000_0422 | Animation: Alternate Outfits | Elsa |
| arena1000_0424 | Tabletop And Books: Action Scenes | Dungeons & Dragons |
| arena1000_0426 | Video Games: Cosplay And Styles | Link（Breath of the Wild） |

## 处理结果（2026-11-07，已执行方案 B）

- 58 条全部按方案 B 重写：类别不变，真人（含 6 条历史人物）改成虚构的无名人物，版权 IP 改成新编的角色或作品。id、taxonomy、prompt_form、target_words 都不变，writer_model 记录实际改写用的模型。
- 改写后用同一个 Opus 5.5 筛查 prompt 复查，58/58 都没有命中；这 58 条的 dims 用 dims_labeler（facets2，3 模型投票）重新标注。
- 其余 942 条 prompt 和 dims 都没改。
- 新的 `arena1000.json` 已上传 GCS `qimagebench/`。opus-4.7 v5 JSON 改写只重跑了这 58 条（58/58 成功），已合并回 `upsample_output/arena1000_opus4p7_v5_raw.json`。
- `qimagebench_baseline/` 下 6 个模型目录里这 58 个 id 的旧图已删除（共 332 张），其余图保留，补生成时只会生成缺失的图。

## 处理方案（当时的选项）

- **A. taxonomy 层面去掉两类再补足（推荐）**：从 arena 的 taxonomy 里排除 `Fan Art of Existing IP` 和 `Celebrities & Public Figures` 两个 L2，这 86 个名额按原规则（L1 权重 + L2 权重）重新分配给其他类别，只重写这 86 条（加上 0576），其余 913 条 prompt 和 id 不变。好处是根上解决，以后重新生成也不会再出现。
- **B. 原类别保留、重写成原创**：同一 taxonomy 位置，让写 prompt 的模型改成原创角色 / 虚构人物（例如"原创的忍者少年"）。改动小，但"同人图"类改成原创后容易变成擦边仿制，名人类也会和 People & Portraits 其他子类重复。
- **C. 只删命中条目**：arena 变成 942 条。最快，但分类比例被打乱，L1 权重和保底也不再成立。

需要定的点：
1. 选 A / B / C？
2. 去世已久的历史人物（Elizabeth I、Darwin、Aristotle、Kant、Robert Burns）算不算要清掉的"知名人"？
3. 改完 prompt 后，dims 只对改动的条目重标；已生成的 baseline 图和 opus-4.7 JSON 改写只需补这些条目，还是全部重来？
