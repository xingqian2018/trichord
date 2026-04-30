# Why v16's stage-4 structured caption (s4s) hits only 0.6481 recall

**Subject:** `golden_caption_v16_g3pg3p` — variant **s4s** (stage-3 structured + stage-4 camera/style)
**Benchmark:** `CosCapBenchImage/V1` — 294 images evaluated, 4,797 recall assertions
**Judge for P/R evaluation:** `gemini-3.1-pro`
**Run cluster:** gcpcode (Slurm jobs `1017972` precision, `1017975` recall)
**Authored:** 2026-04-29

---

## 1. Headline numbers (v16s4s only)

| Metric | Value |
|---|---|
| **Recall** | **0.6481** (3,109 pass / 4,797 total) |
| Precision | 0.9736 (27,063 correct / 28,382 claim chunks) |
| F1 | 0.7766 |
| Images recall-evaluated | 294 / 294 |
| Images precision-decomposed | 299 / 300 (1 image dropped on the precision side) |
| Recall failures analyzed | 1,688 |

Precision is in line with prior v14/v15 s4s (~0.93–0.97), so the bottleneck is **recall**.

---

## 2. Failure-mode breakdown (1,688 fails)

Buckets are first-match against the judge's natural-language `reason` field:

| Bucket | Count | Share |
|---|---|---|
| **Caption contradicts the assertion** ("states X, not Y") | 996 | 59.0% |
| **Caption omits the asked detail** ("does not mention / describe / specify") | 533 | 31.6% |
| Partial / insufficient detail | 12 | 0.7% |
| Unclassified (mostly contradictions in different phrasing) | 147 | 8.7% |

The s4s flip versus stage-3 captions is notable: s3d/s3s recall fails are dominated by **silence** (~50% omission), while **s4s talks more and disagrees more**. Most s4s misses are not omissions — the caption did say something, it was just not what the assertion expected.

---

## 3. Three things that systematically lose recall on s4s

### 3.1 Camera-format / composition / DOF — **pure omission, every run**

These assertions are about image-level metadata (orientation, aspect ratio, framing, depth-of-field) that the s4s caption pipeline does not emit. The same set of assertions fails near-100% across all three signatures (s3d / s3s / s4s), and on s4s specifically:

| Assertion (v16s4s) | Fails / Total |
|---|---|
| the image is a vertical shot | 10/10 |
| the image is framed vertically | 10/10 |
| the image has a vertical composition | 10/12 |
| the image is framed vertically in a portrait orientation | 8/9 |
| the image has a horizontal composition | 7/8 |
| the image has a vertical, portrait orientation | 5/5 |
| the image is in a portrait orientation | 5/5 |
| the image is in portrait orientation | 5/6 |
| the image is framed horizontally in a landscape orientation | 5/6 |
| the image has a vertical orientation | 5/5 |
| the image has a vertical portrait orientation | 5/5 |
| the image is framed tightly vertically | 4/4 |
| the image has a standard aspect ratio | 4/4 |
| the camera framing is vertical | 3/3 |
| the image is in a landscape orientation | 3/5 |
| the image is in landscape orientation | 3/4 |
| the image has a horizontal orientation | 3/3 |
| the background of the image is in sharp focus | 3/3 |
| the image has a vertical aspect ratio | 3/3 |
| the image is framed horizontally | 3/8 |

→ Conservatively **~80–100 missed assertions** are pure orientation/composition/DOF omissions. Stage-4's camera prompt does emit shot-type, but does not consistently emit **portrait/landscape/aspect-ratio** language that the assertion set looks for.

### 3.2 Adversarial / negative assertions baked into the benchmark

A real share of the "contradicts" bucket is the benchmark's own negative-control assertions — assertions that are deliberately false for the image, where the caption correctly disagrees and the judge nonetheless marks `fail`. Examples seen in the v16s4s fail log:

- `the image is a vertical shot` → caption correctly describes a horizontal image
- `all the courts are exclusively blue` → caption correctly says they aren't
- `the storm is perfectly centered in the frame` → caption correctly localizes it lower-left
- `the bottom right foreground is empty` → caption correctly describes rusty metal beams there

These cap the achievable recall regardless of caption quality and are not fixable on the producer side.

### 3.3 Spatial position calls and small-text transcription — see §4 and §5 below

These are the recall losses where the caption is wrong (or the model mis-grounded). Listed by file name in the next two sections.

---

## 4. Spatial / position-disagreement failures (90 cases, by file)

Each row: the failed assertion is what V1 expected; the "caption said" column is the s4s caption's localization (per the judge's reason). Same image may appear multiple times when several spatial assertions fail on it.

| Image | Assertion | What v16s4s caption said |
|---|---|---|
| andrew-pexels-92377587-31379979 | shadows fall towards the bottom left | shadows are cast towards the upper-left |
| andrew-pexels-alexapopovich-10655527 | protective mesh fence in the foreground | safety net is in the background |
| andrew-pexels-alisa-skripina-2147548092-35518191 | most figures in lower arch face left | most figures in lower band face right |
| andrew-pexels-san-fermin-pamplona-549332-1298992 | man taking a photo in the immediate foreground | man taking a photo on a midground balcony |
| andrew-pexels-steve-28887851 | strong light source from the upper right | strong primary light source from the top left |
| boxiang-pexels-guylain-kipoke-504252962-36829317 | the man on the left is on the raft | man on the left is in the water |
| jiashu-pexels-trileafu-36436918 | main intact piece of glass towards top left | main intact piece in the upper-right quadrant |
| jiashu-pexels-trileafu-36436918 | main glass piece framed on left side | main glass piece on the right side |
| jiashu-pexels-trileafu-36436918 | main glass piece is on the left | main body in the upper right quadrant |
| jiashu-pexels-vadym-alyekseyenko-137433856-36656992 | wood in the bottom right foreground | rusty metal beams and rocks in lower-right foreground |
| jiashu-pexels-vadym-alyekseyenko-137433856-36656992 | bottom right foreground is empty | pile of rusty metal beams and rocks in lower-right foreground |
| jiashu-pexels-vadym-alyekseyenko-137433856-36656992 | wooden planks in the right foreground | rusty metal beams in the right foreground (material disagreement) |
| jiashu-pexels-vadym-alyekseyenko-137433856-36656992 | shadows suggest light from the top right | light is coming from the upper left |
| jiashu-pexels-vi-t-anh-nguy-n-2150409023-36263966 | silver cocktail shaker on the right | shaker cup on the left of center |
| mengyao-IMG_1548 | robot positioned in exact center of the frame | robot in lower-center and lower-right portions |
| mengyao-IMG_1548 | robot is angled slightly towards the right | front face + right side panel visible → robot is angled to the left |
| mengyao-IMG_1548 | yellow arrow floor marking pointing right | yellow arrow pointing to the left |
| mengyao-IMG_1665 | prominent white domed building in center background | mentions light-blue dome and reddish-brown dome, no white center dome |
| mengyao-IMG_1824 | traffic light on yellow pole back-and-right of center | yellow traffic light pole is in the center foreground |
| mengyao-IMG_1824 | traffic light is very close to the dark grey car on the left | traffic light foreground, dark grey car midground |
| mengyao-pexels-antonio-miralles-andorra-475029787-35008182 | a few large wind turbines in the foreground | wind turbines in midground/background; foreground is a dark field |
| pretraining-0fedbe97-dee4-4c54-96c6-865fafb3bb0d | bright moon-like object in the upper right | bright celestial body in upper-center, only stars in upper right |
| pretraining-1858c89a-9d21-43ae-b97e-644afdf6642d | the car is parked on the right side of the road | parked cars on the middle-left and left side |
| pretraining-180f7fd8-e38f-4844-bd50-484ef78f6dfb | distinct shadow cast to the left of the creature | shadow extends slightly to the right |
| pretraining-1cf4d7f4-7836-4563-a820-8f813e62fa8d | patient table extends to the right side | table extends from the left |
| pretraining-1cf4d7f4-7836-4563-a820-8f813e62fa8d | scanner oriented with table extending to the right | table extends from the left |
| pretraining-1cf4d7f4-7836-4563-a820-8f813e62fa8d | table extends to the right relative to the gantry | table extends from the left relative to the gantry |
| pretraining-202b4970-74fc-4797-bdf4-fafa2e555b0a | large yacht occupying right side of foreground | small white open boat in lower right foreground |
| pretraining-21dcac62-951a-4855-90ea-33cefedcbedb | the man is positioned on the right side of the frame | the man is centered |
| pretraining-22c83998-233b-4d0b-ab5a-bb617d4d2540 | light source comes strongly from a window in the background | light source originates from the left side of the frame |
| pretraining-25656603-5f2e-477c-9bca-e4332bab15b9 | background vehicles mostly on the left side | background vehicles in center, upper-right, far right edges |
| pretraining-2b7ac7c6-a8af-49c7-9ad1-24be307dae44 | electrical outlet directly in the center below the shirt | electrical outlet in the lower-left quadrant |
| pretraining-2e0ddd40-b374-4302-b3f0-e2a22694a480 | drinking hole on the top right of the lid | drinking hole in the lower-left quadrant |
| pretraining-2e0ddd40-b374-4302-b3f0-e2a22694a480 | drinking hole and tab both on the top right of the lid | drinking hole in the lower-left quadrant |
| pretraining-328c2bd4-a4c0-44a4-a56b-b9dd162ccc31 | cat located on the left page of the book | cat statue in front of the book |
| pretraining-330d209e-6a44-43ee-9f38-18f159781845 | shadows suggesting light coming from the left | light is coming from the right side |
| pretraining-33257966-4246-4609-bf35-cb6e483c0942 | officers standing close to the foreground | people in the midground |
| pretraining-33767618-c9a3-4f38-9a7e-adeb79157f8b | bright, direct sunlight hitting the pumpkins | overcast natural daylight |
| pretraining-35c50254-4891-4132-9e02-f43f0ab5ba95 | robot on left side of the frame | robot on center-to-right side of frame |
| pretraining-360afb33-00e6-4542-afa1-d26686dfc488 | cat on the rock positioned in the lower-left corner | cat on the rock in the lower-center area |
| pretraining-39122b6b-14db-4042-924b-1d322f1e1703 | rooster on the left fully visible in frame | left rooster in close-up; not stated as fully visible |
| pretraining-4455ab2e-3ef1-49d1-9eb1-8bed19163d20 | purple orchid directly behind the bottle | orchids pink/magenta and located upper center-left |
| pretraining-46ba0517-d3cb-42b0-89b3-f8d6defbf209 | hand gripping the tray from the right side | hand grips the rack on the left half of the frame |
| pretraining-46ba0517-d3cb-42b0-89b3-f8d6defbf209 | gloved hand visible on the right side | gloved hand on the left side of the frame |
| pretraining-46ba0517-d3cb-42b0-89b3-f8d6defbf209 | arm and hand enter the frame from the right edge | hand on the left side of the frame |
| pretraining-46ba0517-d3cb-42b0-89b3-f8d6defbf209 | hand positioned on the right side of the image | hand on the left side of the image |
| pretraining-46ba0517-d3cb-42b0-89b3-f8d6defbf209 | hand on the right side of the machine | hand on the left side of the frame |
| pretraining-48210849-41ec-4e7e-a071-34a860ff64f3 | foreground worker is centrally located | foreground worker is on the right |
| pretraining-48b72d3e-4f2f-4bf2-98e3-84ec574e6aa7 | light source is from the upper left | light source is from the upper right |
| pretraining-4884c72f-a218-4d7a-a0fa-cdcc88fbd852 | person's right hand holds the phone lower down | right hand and phone in center of frame, arm extending upwards |
| pretraining-502bc69d-b4f5-4d84-8259-e8187b898465 | ring on the left hand's index finger | ring on the ring finger of the left hand |
| pretraining-53322ad2-4c86-43f3-8d17-23ac36ca5bc9 | woman positioned to the left of the robot | woman on the right half of the image |
| pretraining-53322ad2-4c86-43f3-8d17-23ac36ca5bc9 | robot framed on the right side of composition | robot on the left side of the composition |
| pretraining-53322ad2-4c86-43f3-8d17-23ac36ca5bc9 | layout: woman left, robot right | caption: robot left, woman right |
| pretraining-8a31e2f0-1ba8-4c5c-a222-1f6481938e2c | hand holding the book is on the right side | hand is in the lower-center of the frame |
| pretraining-8a7bea95-08ee-4eb1-8776-1cafae8b2e3f | directional sunlight hitting the subject from the left | directional lighting originates from the right side |
| pretraining-a142d2c4-f9a1-4071-a47a-cd7d41ae4650 | lander is centered in the image | lander is in the lower-left quadrant |
| pretraining-a142d2c4-f9a1-4071-a47a-cd7d41ae4650 | the light comes from the upper right | sunlight originates from the upper left |
| pretraining-a142d2c4-f9a1-4071-a47a-cd7d41ae4650 | subject placed in the center | subject in the lower-left quadrant |
| pretraining-a9661e2e-b81b-4162-b890-39be7b063550 | fire truck on the left side of the field | vehicle on the right side; not identified as a fire truck |
| pretraining-ace55a1e-6b9c-49c7-9725-bea33111c973 | campfire centrally located in the foreground | campfire in the lower right quadrant |
| pretraining-b300cff6-25a2-42ec-9df6-d0e066ab2df6 | sign 'SEVIER COUNTY FAIR & RODEO' on a building in center background | sign in the upper-right quadrant, in front of the large building |
| pretraining-bc0beb26-62a8-4436-8718-a24cc27409a5 | storm is perfectly centered in the frame | storm in center-left to lower-center / lower-left quadrant |
| pretraining-bc0beb26-62a8-4436-8718-a24cc27409a5 | single, cohesive spiral centered in the frame | cyclonic structure in lower-left quadrant or center-left to lower-center |
| pretraining-c23eb83e-71bf-4ebd-acd6-a51e330cd72b | bright, vibrant green color scheme | palette is murky and low-contrast |
| pretraining-c4f22422-7b95-451b-a1bb-8c0e01b5840b | image centers on a single large flower | composition has multiple central flowers |
| pretraining-d5524fca-5a7a-4349-a7fb-7d4d9cfda2c9 | Earth positioned to the far left | planet in the middle third of the frame |
| pretraining-d5524fca-5a7a-4349-a7fb-7d4d9cfda2c9 | Earth on the left side of the image | planet in the middle third of the frame |
| pretraining-d5524fca-5a7a-4349-a7fb-7d4d9cfda2c9 | satellite on the right side of the image | satellite in the upper center of the frame |
| pretraining-df0bfa3d-f8dd-4510-9bdb-d5dad070c5bb | hand reaching in from the top left | hand and arm reaching from the upper-right |
| pretraining-df0bfa3d-f8dd-4510-9bdb-d5dad070c5bb | arm enters the frame from the top left | arm enters from the upper-right |
| pretraining-df0bfa3d-f8dd-4510-9bdb-d5dad070c5bb | arm oriented diagonally from top-left to center | arm oriented diagonally from the upper-right |
| pretraining-df0bfa3d-f8dd-4510-9bdb-d5dad070c5bb | the light is coming from the top right | light is coming from the upper left |
| pretraining-df0bfa3d-f8dd-4510-9bdb-d5dad070c5bb | cooler placed slightly left of center | cooler in the lower-center to lower-right foreground |
| pretraining-e106d7a0-7d41-4752-9a09-57de5a134348 | person on the left, monitors on the right | person on the right, monitors in center and center-left |
| pretraining-e71da176-f6f6-45e3-b81c-87152a182b55 | the man is facing left | the man is facing the right side of the frame |
| pretraining-e85e6cd3-c168-4f1a-a727-5c04c361ccfd | image has a heavily blurred background | mentions a soft blur, not heavy |
| pretraining-ee5e747e-a412-4890-ab26-d4a469b7c7ad | bright, direct overhead lighting | lighting described as diffuse |
| pretraining-fc89a08e-c67f-4145-923a-451e6179da2b | three prominent rock formations in the foreground | only a single small dark rock formation in the center foreground |
| pretraining-fc89a08e-c67f-4145-923a-451e6179da2b | large rock formation on the left occupies significant left foreground | large rock formation on the left is in the midground |
| pretraining-fc89a08e-c67f-4145-923a-451e6179da2b | rock formations clustered tightly in the center | rock formations scattered across midground/background spanning the width |
| seungjun-pexels-helena-nguyen-677907392-18183152 | lighting clear and vibrant like a bright sunset | soft, natural daylight |
| seungjun-pexels-kateryna-tartachna-398669386-36535644 | large bush of purple flowers dominating the left foreground | purple flowers in lower-center and lower-right; left foreground is a patio |
| shitao-Zugpsitze_mountain | mountain range with broad, flat-topped peak on the left | mountain on the left has sharp peaks |
| shitao-forest_red_fox_l | fox's head turned back looking to the left | fox's head turned to the right |
| xingqian-07_happy_birthday_simple_version | spool of twine in the bottom left corner | spool of twine in the upper-left corner |
| xingqian-11_sign_board | arrows are on the right side of the panels | arrows on the left side of the panels |
| xingqian-11_sign_board | sign viewed from an angle looking up and to the left | sign receding from the left across the frame → view to the right |
| xingqian-11_sign_board | structural beams of a covered walkway/terminal entrance above | canopy structure described at the bottom edge of the image |
| xingqian-11_sign_board | background grey and cloudy | sky clear, bright, almost white / pale, washed-out |

**Pattern.** The dominant left/right errors are clustered in a small set of images (`pretraining-46ba0517-...` ×5, `pretraining-df0bfa3d-...` ×5, `pretraining-53322ad2-...` ×3, `pretraining-d5524fca-...` ×3, `pretraining-1cf4d7f4-...` ×3, `pretraining-fc89a08e-...` ×3) — i.e. when an image goes wrong on a left-right call, every assertion that depends on that call fails together. Fixing the spatial grounding for those few images would recover ~25 assertions.

---

## 5. OCR / text / logo / number transcription failures (70 cases, by file)

The judge's reason often quotes both the asserted text and what the caption transcribed instead, which makes these straightforward OCR errors. Where the caption simply did not mention the text/logo at all, the row says "caption omits".

| Image | Assertion | What v16s4s caption said |
|---|---|---|
| andrew-pexels-alisa-skripina-2147548092-35518191 | carved figure of a centaur holding a bow | caption omits any centaur or bow |
| andrew-pexels-chris-wade-ntezicimpa-564856410-32463996 | prominent number '14' on the white jersey of the ball handler | '14' is on the shorts of a seated player on the far right |
| andrew-pexels-chris-wade-ntezicimpa-564856410-32463996 | Nike swoosh on the white jersey | Nike swooshes on the socks and shoes; nothing on the white jersey |
| andrew-pexels-chris-wade-ntezicimpa-564856410-32463996 | number '14' visible on the ball handler's jersey | '14' is on the shorts of a seated player on the sideline |
| andrew-pexels-zekai-zhu-214984943-11831851 | text 'OXFORD CSSA' split onto two lines on the dark jersey | text rendered as a single word 'OXFORDCSSA', no line-break call |
| jiashu-pexels-pattvielma-1474928 | textured ground with leaves and ash is in the background | textured ground is in the foreground (foreground/background flip) |
| pretraining-0887005f-563c-4b37-9cb1-812b0af38e56 | text 'AV' is visible on the wood | caption omits 'AV' |
| pretraining-0887005f-563c-4b37-9cb1-812b0af38e56 | text 'G20M' is visible on the wood | caption transcribed 'GROM' |
| pretraining-0adb2517-5dc5-41f5-9559-300e8d3f91ce | image looks like live-action cosplay photography | caption: digital illustration or 3D render |
| pretraining-0d1a6c87-0d1d-4ebe-8c4c-b05fd6d624b0 | text 'miniLab 150' on a flat white label on a flat blue top | silver-coloured label on an angled surface |
| pretraining-172655c7-1112-4a7a-b1a4-2879b085982f | logo on the vans is a 3D box | logo described as angular geometric shapes and diagonal stripes |
| pretraining-180f7fd8-e38f-4844-bd50-484ef78f6dfb | creature has a dark grey chest plate | caption: chest plate is white or light grey |
| pretraining-202b4970-74fc-4797-bdf4-fafa2e555b0a | boat named 'SKY HOOKER' | text on boat transcribed as 'ENSKY HOOMER' |
| pretraining-202b4970-74fc-4797-bdf4-fafa2e555b0a | boat named 'BENNY HOOKER' | text on boat transcribed as 'ENSKY HOOMER' |
| pretraining-23fa31d5-0bc7-4a7d-b74e-890106da9157 | text on the plaque on the seat is plain blocky text | text described as cursive-style font |
| pretraining-2a2e4843-d2d4-4b6d-b40f-c7854a00658d | text 'GIANT ROBERT COMPETITION' stamped on derailleur | caption transcribed 'GIAN ROBERT' |
| pretraining-2a2e4843-d2d4-4b6d-b40f-c7854a00658d | text reads 'GIANT ROBERT COMPETITION' | caption transcribed 'GIAN ROBERT' |
| pretraining-328c2bd4-a4c0-44a4-a56b-b9dd162ccc31 | word 'again' on the right page partially occluded by the cat statue | caption mentions a shadow over text but no 'again' / occlusion call |
| pretraining-33257966-4246-4609-bf35-cb6e483c0942 | caution tape in the foreground has clear, readable text | caption: text on tape is blurry |
| pretraining-35c50254-4891-4132-9e02-f43f0ab5ba95 | robot has a smooth white chest plate without rivets | caption explicitly mentions metallic studs/fasteners bordering the chest plate |
| pretraining-4455ab2e-3ef1-49d1-9eb1-8bed19163d20 | text 'Maker's Mark' is printed on the glass of the bottle | caption: 'Maker's Mark' on glass is embossed (print is only on the paper label) |
| pretraining-46ba0517-d3cb-42b0-89b3-f8d6defbf209 | label reading 'B - SAMPLE SET 04' | caption omits 'B - SAMPLE SET 04' |
| pretraining-4aa5d8c9-37b2-488c-afbe-8d40258073aa | smaller bear following behind the main bear in the logo | caption omits the smaller bear |
| pretraining-4aa5d8c9-37b2-488c-afbe-8d40258073aa | diagonal seam below the bear logo | caption: diagonal raglan sleeve seam is to the left of the logo, not below |
| pretraining-4e1ba6a0-f62c-4d79-a628-1aebbc212ad4 | rings have clear 'TOP 73.00MM' markings | caption transcribed 'TOP 142X1053' |
| pretraining-4e1ba6a0-f62c-4d79-a628-1aebbc212ad4 | text on the rings reads 'TOP 73.00MM' | caption transcribed 'TOP 142X1053' |
| pretraining-53322ad2-4c86-43f3-8d17-23ac36ca5bc9 | image looks like a traditional vintage photograph | caption: highly detailed digital illustration or 3D render style |
| pretraining-67116534-5f8d-40e0-83c0-438b8a91136a | patient wearing lanyard with 'UT Health Austin' text | caption places 'UT Health Austin' on the patient's scrub top, not the lanyard |
| pretraining-6862eeeb-813e-4ce1-9f6c-c185a858aefd | face is a photorealistic human face | caption: 'stylized skin textures' in 'digital art or 3D CGI style' |
| pretraining-6862eeeb-813e-4ce1-9f6c-c185a858aefd | image is photorealistic live-action style | caption: digital art or 3D CGI style |
| pretraining-6adc4f14-b2d0-4672-8961-885ef29e528c | text printed on the large white plates | caption omits text on the large white plates |
| pretraining-6adc4f14-b2d0-4672-8961-885ef29e528c | text 'hello lovely' written on several large plates | caption: 'hello lovely' is on the small oval dish |
| pretraining-6adc4f14-b2d0-4672-8961-885ef29e528c | text 'hello lovely' printed in blue on multiple large plates | caption: 'hello lovely' is on a small oval dish |
| pretraining-73067986-3d1b-4f0f-88f5-129d8fe20155 | 'BAN 2' sticker on the base | caption transcribed 'BAA Z' |
| pretraining-73067986-3d1b-4f0f-88f5-129d8fe20155 | neon text at the top | caption: text is printed and blurry, not neon |
| pretraining-a25bd4d5-ae68-482b-859c-01a24237253a | control knobs mounted directly on the pickguard | caption: knobs mounted on a metallic control plate |
| pretraining-a8a40687-d3b5-4399-b8c3-3291549158ac | text 'NISSAN CONCEPT 2020' in a specific sans-serif font | caption mentions the text but omits font-style call |
| pretraining-ace55a1e-6b9c-49c7-9725-bea33111c973 | visible logo on the side of the boots | caption omits logo on boots |
| pretraining-b0ef2acc-5987-485d-97ac-b3d1564f2a5e | small silver stylized T or L logo on the side sensor | caption: symbol is a circle intersected by a diagonal line |
| pretraining-b0ef2acc-5987-485d-97ac-b3d1564f2a5e | Tesla logo on the side of the helmet | caption omits Tesla logo |
| pretraining-c016715e-4827-4f94-bd46-bfce4d77b1ca | cookies on a plate on the book cover | caption: printed graphic of cookies, no plate |
| pretraining-c263daa3-6631-4e96-a3b8-4f963d8114f3 | text '10.04.2.2' written on the side panel | caption transcribed 'DAW' |
| pretraining-c263daa3-6631-4e96-a3b8-4f963d8114f3 | image looks like a watercolor and ink sketch | caption: digital illustration with digital ink/pencil strokes |
| pretraining-c263daa3-6631-4e96-a3b8-4f963d8114f3 | text 'BAW' written on the side of the drone | caption transcribed 'DAW' |
| pretraining-c263daa3-6631-4e96-a3b8-4f963d8114f3 | text '10.04.2.2' written on the drone | caption transcribed 'DAW' |
| pretraining-c94cb238-7468-46d0-b734-fc7aef6ab806 | writing on the green chalkboard | caption omits chalkboard writing |
| pretraining-c94cb238-7468-46d0-b734-fc7aef6ab806 | writing on the chalkboard | caption omits chalkboard writing |
| pretraining-d4c1746f-1014-4dcc-9ba4-a410dd07f989 | signature sticker has holographic pattern of repeating circles | caption: repeating pattern of the word 'VALID' |
| pretraining-d6ce5a4b-d682-4754-b9be-90e5f814d8b9 | pronounced shadows on textured surfaces | caption: dark unlit areas in the gaps |
| pretraining-d8ca2ccf-5597-4939-972c-dba3bfa97959 | map contains text 'Sellaar Rard Hiden' | caption transcribed 'Selisu Pard liden' |
| pretraining-d8ca2ccf-5597-4939-972c-dba3bfa97959 | map contains text 'Lgmissa bodoore' | caption transcribed 'Lguisa bodore' |
| pretraining-df00ec71-0ea6-43f5-8cec-a8ef76a0d325 | words 'positive' and 'negative' are side-by-side | caption: 'positive' and 'negative' are stacked vertically |
| pretraining-e48360bd-98f7-429e-b137-17cf7e58b927 | word 'EXPERIMENTAL' written in bold black letters | caption: 'EXPERIMENTAL' is faint grey |
| pretraining-e984501b-1153-469b-bdad-8abb50331116 | front of the train has a domed shape with rivets | caption: front is a flat, circular metal plate |
| pretraining-ef4591d5-032a-45f4-a863-40e7a499c78d | text on the lab coat says 'Viktor' | caption transcribed 'Visitor' |
| pretraining-ef4591d5-032a-45f4-a863-40e7a499c78d | text on the lab coat says 'BIOFIL USA' | caption omits 'BIOFIL USA' |
| seungjun-pexels-chris-wade-ntezicimpa-564856410-32370453 | text 'KALGIC' on white jersey is partially obscured | caption transcribed 'KIGALI C' |
| seungjun-pexels-chris-wade-ntezicimpa-564856410-32370453 | number '20' on white jersey fully visible and unobscured | caption: '20' is partially folded |
| seungjun-pexels-jacint-bofill-1745787-31426908 | sign reads 'CS B2B' above a door on the left | caption transcribed 'CS 628' |
| seungjun-pexels-mart-production-7330759 | text on the wine bottle label | caption omits any label or text on wine bottle |
| seungjun-pexels-pili-toro-126287569-12946084 | reflections visible on the window | caption omits reflections (mentions views through and text on the window) |
| seungjun-pexels-pili-toro-126287569-12946084 | clear 'Café Triciclo' sign on the wall | caption: 'CAFE TRICICLO' is on a red mug, not a wall sign |
| shitao-What-We-Do_program-area_People-to-People-Exchanges_ | logos visible on the lanyards | caption mentions ID badges on lanyards but no logos |
| shitao-final_women-in-sports_credit-alamy | player on the LEFT of the hugging pair wears a blue warm-up jacket | caption places the blue-jacket player on the right |
| shitao-final_women-in-sports_credit-alamy | player on the RIGHT of the hugging pair wears a red uniform | caption places the red-jersey player on the left |
| xingqian-03_highway_streetview | main car has a visible license plate | caption omits license plate |
| xingqian-07_happy_birthday_simple_version | word 'Happy' on first line, 'birthday!' on second line | caption omits the line-arrangement |
| xingqian-09_basketball_match | banner in the background that says 'BREVECTER BANDITS' | caption: 'BANDITS' banner, no 'BREVECTER' |
| xingqian-13_complex_road_sign_with_ch_and_en | white and blue logo on a square block | caption: gold logos on a blue octagonal cap |
| xingqian-14_text_crazy_coffee_cap | (top recall offender — 13/21 fail; OCR-heavy stack) | many specific small-text claims; caption transcribes coarsely |
| xingqian-04_biology_class | whiteboard only has letters pointing to the drawing | caption: full text labels (e.g. 'contractile vacuole A', 'food vacuole B') |

**Pattern.** OCR errors split into three sub-patterns:
1. **Letter substitution** under load: `BAW`→`DAW`, `BAN 2`→`BAA Z`, `KALGIC`→`KIGALI C`, `CS B2B`→`CS 628`, `Sellaar Rard Hiden`→`Selisu Pard liden`. The model reads, but flips letters/digits.
2. **Wrong surface attachment**: text correctly transcribed but localized to the wrong object — `'14'` on shorts vs jersey; `'hello lovely'` on a small dish vs the large plates; `'Café Triciclo'` on a mug vs a wall sign; `'UT Health Austin'` on scrub vs lanyard.
3. **Style/format calls**: photo-vs-render (`pretraining-0adb2517`, `pretraining-53322ad2`, `pretraining-c263daa3`, `pretraining-6862eeeb`) — the s4s caption confidently labels stylized images as "digital illustration / 3D render" while the assertion expects a different judgement of style.

---

## 6. Worst-recall images (by absolute fail count, v16s4s)

The bottom-25 images concentrate ~280 of the 1,688 fails (~17% of total). The bulk are pretraining IDs, with a strong over-representation of:

- Images dense with **small text / signage** (`pretraining-73067986`, `pretraining-d8ca2ccf`, `xingqian-14_text_crazy_coffee_cap`, `xingqian-09_basketball_match`)
- Images with **left-right ambiguity** (`pretraining-46ba0517` — 5 spatial fails, `pretraining-53322ad2`, `pretraining-d5524fca`)
- Images where the benchmark assertion set includes **adversarial negatives** (`pretraining-bc0beb26` storm-centered, `pretraining-fc89a08e` rock formations)

| Image | Fail / Total |
|---|---|
| mengyao-pexels-keeganjchecks-36423795 | 20/36 |
| pretraining-73067986-3d1b-4f0f-88f5-129d8fe20155 | 19/32 |
| jiashu-pexels-vadym-alyekseyenko-137433856-36656992 | 16/40 |
| pretraining-8a31e2f0-1ba8-4c5c-a222-1f6481938e2c | 16/30 |
| pretraining-a9661e2e-b81b-4162-b890-39be7b063550 | 15/30 |
| pretraining-d1af890b-3cc6-4ef6-8f23-cb64e7243242 | 15/36 |
| pretraining-53322ad2-4c86-43f3-8d17-23ac36ca5bc9 | 15/40 |
| pretraining-ba2ddd2a-64f4-42c8-98ab-7ef2596267e5 | 15/28 |
| pretraining-d54e4c45-b05b-4d05-882b-ce59a79b54a9 | 14/33 |
| pretraining-ee5e747e-a412-4890-ab26-d4a469b7c7ad | 14/25 |
| pretraining-b4099113-5259-4982-a556-07d32a2d84f9 | 14/18 |
| pretraining-eb0c9405-183d-4318-b738-3c7c6ca024df | 14/24 |
| pretraining-33257966-4246-4609-bf35-cb6e483c0942 | 14/36 |
| xingqian-14_text_crazy_coffee_cap | 13/21 |
| andrew-pexels-san-fermin-pamplona-549332-1298992 | 13/24 |
| boxiang-pexels-nolandlive-36817196 | 13/26 |
| seungjun-pexels-pili-toro-126287569-12946084 | 13/34 |
| boxiang-pexels-andi-saiful-sidik-2160410058-36823828 | 13/28 |
| pretraining-426971d2-29ca-4b9e-9d8d-5d2485cdc7b4 | 13/21 |
| pretraining-27a3ac6f-265d-4466-95cd-dd2f174b136d | 12/29 |
| pretraining-502bc69d-b4f5-4d84-8259-e8187b898465 | 12/26 |
| pretraining-a142d2c4-f9a1-4071-a47a-cd7d41ae4650 | 12/27 |
| andrew-pexels-alisa-skripina-2147548092-35518191 | 12/23 |
| jiashu-pexels-a-darmel-7322455 | 12/30 |
| pretraining-180f7fd8-e38f-4844-bd50-484ef78f6dfb | 12/23 |

---

## 7. Recommendations specific to v16s4s

To move recall on s4s:

- **Patch the camera-format hole.** Stage 4 already emits camera/style — add a single field that names **portrait/landscape** and **aspect-ratio class** (vertical / horizontal / square). Cheap; recovers ~80–100 assertions across the ~50 most-frequently-failed orientation/aspect templates.
- **Audit adversarial-negative assertions in V1.** A recall ceiling above ~0.70 likely requires either filtering "exclusively / only / perfectly / no X visible" assertions before scoring, or scoring them as a separate adversarial bucket. Without that, no caption can pass.
- **Targeted OCR pass.** ~30+ of the 70 OCR fails are letter-flip transcriptions. A second pass with an OCR-specialized model (e.g. text crops sent to a dedicated OCR call) and a string-match override on text claims would directly recover them. Fixing only the top-3 OCR-heavy images (`pretraining-c263daa3` ×3, `pretraining-2a2e4843` ×2, `pretraining-d8ca2ccf` ×2, `pretraining-202b4970` ×2, `pretraining-4e1ba6a0` ×2, `pretraining-6adc4f14` ×3) recovers ~14 assertions from a single fix.
- **Left/right grounding fix on a small set of images.** The top 6 left/right-confused images (`pretraining-46ba0517`, `pretraining-df0bfa3d`, `pretraining-53322ad2`, `pretraining-d5524fca`, `pretraining-1cf4d7f4`, `pretraining-fc89a08e`) account for ~25 spatial fails. A cheap intervention: stage 2 grounding double-check for left/right anchors when the entity has lateralized features.
- **Foreground/background reconciliation.** A small but consistent class of fails (≈10) is f/g/b confusion (cat statue "in front of" vs "on the page", textured ground "background" vs "foreground"). Stage 3 should pick a single f/m/b for each entity and stick to it.

---

## 8. Pointers

- Slurm jobs (gcpcode):
  - precision: `~/log/slurm/coscapbench_precision_v16s4s.1017972.{e,o}` — done 2026-04-29 04:42:47
  - recall:    `~/log/slurm/coscapbench_recall_v16s4s.1017975.{e,o}`    — done 2026-04-29 04:14:52
- Result JSONs (GCS):
  - `gs://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_v16_g3pg3p/results/recall_eval_result_v16s4s.json`
  - `gs://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_v16_g3pg3p/results/recall_eval_claims_v16s4s.json` (per-assertion judge output, source for §3–§6 of this report)
  - `gs://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_v16_g3pg3p/results/precision_eval_{result,claims}_v16s4s.json`
- Caption source (GCS): `gs://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_v16_g3pg3p/stage4_structured_caption/`
- Companion reports:
  - `~/Project/trichord/reports/golden_caption_analysis.md` (v2 → v15 summary)
  - `~/Project/trichord/reports/golden_caption_analysis_v15vsv14.md` (v14 → v15 root-cause)
