/**
 * Safety-first, non-clinical voice exploration manual.
 * Copy is intentionally conservative: VPA cannot examine vocal folds or prescribe therapy.
 */

const SOURCE_URLS = {
  asha: "https://www.asha.org/practice-portal/professional-issues/gender-affirming-voice-and-communication/",
  nidcdCare: "https://www.nidcd.nih.gov/health/taking-care-your-voice",
  nidcdHoarseness: "https://www.nidcd.nih.gov/health/hoarseness",
  ucsf: "https://transcare.ucsf.edu/guidelines/vocal-health",
};

const COPY = {
  "zh-Hant": {
    title: "女性化聲音手冊",
    eyebrow: "安全優先 · 非臨床聲音探索",
    intro: "這份手冊幫你比較錄音與找到自己喜歡、舒服、可重複的表達方式。它不是醫療建議或聲音治療方案，也沒有任何單一數值能定義女性聲音。",
    firstTitle: "開始前先知道",
    first: [
      "目標由你決定：女性化可以包含音高、語調、清晰度、節奏、音量與互動風格，也可以完全不追求其中任何一項。",
      "VPA 的百分比與細項是受語句、語言、母音、麥克風和環境影響的模型／聲學估計；不能判斷性別認同、聲帶健康或正確發聲法。",
      "App 適合比較相同條件下的錄音，不能取代具性別肯認聲音經驗的語言治療師。",
    ],
    stopTitle: "停止與尋求專業協助",
    stopLead: "出現疼痛、沙啞、緊繃、明顯費力、音域突然改變或聲音突然消失時，立即停止發聲練習並休息；不要改做哼聲、耳語或其他發聲練習。",
    stopCare: "沙啞超過三週，或伴隨說話／吞嚥疼痛、呼吸或吞嚥困難、頸部腫塊、咳血等情況，應尋求耳鼻喉科評估。需要調整聲音時，可找具聲音專長及性別肯認經驗的語言治療師。",
    sessionTitle: "一次安全的自我比較流程",
    session: [
      ["1. 自然基準", "用日常音量說一小段話，不先改變聲音。重播後記錄：舒服嗎、自然嗎、容易重複嗎？"],
      ["2. 只改一個小元素", "喉嚨完全舒服時，可只比較一個風格元素，例如說慢一點、咬字更清楚，或用不同語調讀同一句。不要同時追多個數值。"],
      ["3. 立即回聽", "先依自己的感受與喜好判斷，再看 VPA。若 App 分數變高但聲音更費力，那次就不是更好的結果。"],
      ["4. 留下可重複的版本", "以舒服、自然、符合自己的目標及隔天仍無不適為準；分數只記錄模型差異。"],
    ],
    avoidTitle: "不要自行嘗試",
    avoid: [
      "不要用吞嚥、手壓喉部或刻意固定喉頭位置來找聲音。",
      "不要硬拉極高／極低音、刻意擠壓、製造漏氣感、喊叫或耳語來追 App 指標。",
      "不要把『胸、口罩、頭部共鳴』當成真實解剖位置；VPA 顯示的是低／中／高頻能量代理指標。",
      "不要依混合語音的 F1–F3 數值自行判斷舌頭、下顎、聲帶閉合或疾病。",
    ],
    metricsTitle: "正確閱讀 VPA 細項",
    metrics: [
      ["女性化傾向", "模型對這次錄音的分類傾向，不代表身分，也沒有理想分數。"],
      ["音高", "偵測到的有聲片段統計；語句、情緒與麥克風都會影響，沒有人人適用的女性音高。"],
      ["F1–F3／母音焦點", "受母音、語言與說話者影響的描述；混合語音無法精確推斷口腔或喉部位置。"],
      ["共振／亮度／頻譜傾斜", "頻率能量的代理描述，不是身體共鳴位置、聲帶狀態或健康檢查。"],
      ["氣聲", "頻譜代理估計，不能測量漏氣或聲帶閉合，也沒有安全的通用目標區間。"],
      ["語速／停頓／語調", "與語言、語意、情緒和個人風格高度相關，不是性別或健康標準。"],
      ["聲齡", "娛樂性的模型印象，不是實際年齡、生理年齡或健康評估。"],
    ],
    logTitle: "怎樣才算進步",
    log: "每次只記錄日期、語句、設備，以及舒服度、自然度、喜歡程度、可重複性。用相同設備和距離比較趨勢；若任何不適增加，就停止並回到自然說話，不追分數。",
    sourcesTitle: "專業依據與延伸閱讀",
    sources: ["ASHA：性別肯認聲音與溝通", "NIDCD：聲音照護", "NIDCD：沙啞警訊", "UCSF：聲音健康指南"],
  },
  "zh-Hans": {
    title: "女性化声音手册", eyebrow: "安全优先 · 非临床声音探索",
    intro: "这份手册帮助你比较录音并找到自己喜欢、舒服、可重复的表达方式。它不是医疗建议或声音治疗方案，也没有任何单一数值能定义女性声音。",
    firstTitle: "开始前先知道",
    first: ["目标由你决定：女性化可以包含音高、语调、清晰度、节奏、音量与互动风格，也可以完全不追求其中任何一项。", "VPA 的百分比与细项是受语句、语言、元音、麦克风和环境影响的模型／声学估计；不能判断性别认同、声带健康或正确发声法。", "App 适合比较相同条件下的录音，不能取代有性别肯定声音经验的言语治疗师。"],
    stopTitle: "停止与寻求专业协助", stopLead: "出现疼痛、沙哑、紧绷、明显费力、音域突然改变或声音突然消失时，立即停止发声练习并休息；不要改做哼声、耳语或其他发声练习。", stopCare: "沙哑超过三周，或伴随说话／吞咽疼痛、呼吸或吞咽困难、颈部肿块、咳血等情况，应寻求耳鼻喉科评估。需要调整声音时，可找有声音专长及性别肯定经验的言语治疗师。",
    sessionTitle: "一次安全的自我比较流程",
    session: [["1. 自然基准", "用日常音量说一小段话，不先改变声音。回放后记录：舒服吗、自然吗、容易重复吗？"], ["2. 只改一个小元素", "喉咙完全舒服时，可只比较一个风格元素，例如说慢一点、咬字更清楚，或用不同语调读同一句。不要同时追多个数值。"], ["3. 立即回听", "先依自己的感受与喜好判断，再看 VPA。若 App 分数变高但声音更费力，那次就不是更好的结果。"], ["4. 留下可重复的版本", "以舒服、自然、符合自己的目标及隔天仍无不适为准；分数只记录模型差异。"]],
    avoidTitle: "不要自行尝试", avoid: ["不要用吞咽、手压喉部或刻意固定喉头位置来找声音。", "不要硬拉极高／极低音、刻意挤压、制造漏气感、喊叫或耳语来追 App 指标。", "不要把‘胸、口罩、头部共鸣’当成真实解剖位置；VPA 显示的是低／中／高频能量代理指标。", "不要依混合语音的 F1–F3 数值自行判断舌头、下颌、声带状态或疾病。"],
    metricsTitle: "正确阅读 VPA 细项",
    metrics: [["女性化倾向", "模型对这次录音的分类倾向，不代表身份，也没有理想分数。"], ["音高", "检测到的有声片段统计；语句、情绪与麦克风都会影响，没有人人适用的女性音高。"], ["F1–F3／元音焦点", "受元音、语言与说话者影响的描述；混合语音无法精确推断口腔或喉部位置。"], ["共振／亮度／频谱倾斜", "频率能量的代理描述，不是身体共鸣位置、声带状态或健康检查。"], ["气声", "频谱代理估计，不能测量漏气或声带状态，也没有安全的通用目标区间。"], ["语速／停顿／语调", "与语言、语义、情绪和个人风格高度相关，不是性别或健康标准。"], ["声龄", "娱乐性的模型印象，不是实际年龄、生理年龄或健康评估。"]],
    logTitle: "怎样才算进步", log: "每次只记录日期、语句、设备，以及舒适度、自然度、喜欢程度、可重复性。用相同设备和距离比较趋势；若任何不适增加，就停止并回到自然说话，不追分数。",
    sourcesTitle: "专业依据与延伸阅读", sources: ["ASHA：性别肯定声音与沟通", "NIDCD：声音照护", "NIDCD：沙哑警讯", "UCSF：声音健康指南"],
  },
  en: {
    title: "Feminine Voice Manual", eyebrow: "SAFETY FIRST · NON-CLINICAL VOICE EXPLORATION",
    intro: "This manual helps you compare recordings and find a presentation that feels comfortable, authentic, and repeatable. It is not medical advice or a voice-therapy program, and no single value defines a feminine voice.",
    firstTitle: "Before you begin",
    first: ["You choose the goal. Feminine presentation may involve pitch, intonation, clarity, pacing, loudness, and interaction style—or none of these.", "VPA percentages and details are model/acoustic estimates affected by wording, language, vowels, microphone, and room. They cannot assess gender identity, vocal-fold health, or correct technique.", "The app is useful for like-for-like recording comparison; it cannot replace a speech-language pathologist experienced in gender-affirming voice care."],
    stopTitle: "When to stop and seek care", stopLead: "Stop voice practice and rest your voice if you notice pain, hoarseness, tightness, marked effort, sudden range change, or sudden voice loss. Do not switch to humming, whispering, or another voiced exercise.", stopCare: "Seek an ENT/laryngologist for hoarseness lasting more than three weeks, or sooner for pain with speaking or swallowing, breathing or swallowing difficulty, a neck lump, or coughing blood. For voice change, consider a voice-specialized speech-language pathologist with gender-affirming experience.",
    sessionTitle: "A safe self-comparison session",
    session: [["1. Capture a natural baseline", "Speak a short passage at your everyday volume without changing your voice first. On replay, ask: Was it comfortable, authentic, and easy to repeat?"], ["2. Change one small style element", "Only while fully comfortable, compare one element—such as clearer articulation, a slower pace, or a different intonation on the same sentence. Do not chase several metrics at once."], ["3. Listen before looking", "Judge the take by your own comfort and preference before opening VPA. A higher app score with more effort is not a better result."], ["4. Keep what is repeatable", "Prioritize comfort, authenticity, your own goal, and no later discomfort. The score records only a model difference."]],
    avoidTitle: "Do not self-prescribe",
    avoid: ["Do not use swallowing, manual pressure on the neck, or a deliberately fixed larynx position to find a voice.", "Do not force extreme pitch, squeeze, add airiness, shout, or whisper to chase an app metric.", "Do not treat chest/mask/head resonance as literal anatomy; VPA displays low/mid/high spectral-energy proxies.", "Do not use F1–F3 from mixed speech to diagnose tongue, jaw, vocal-fold behavior, or disease."],
    metricsTitle: "How to read VPA details",
    metrics: [["Feminine tendency", "A model classification tendency for this recording, not identity and not an ideal score."], ["Pitch", "A statistic from detected voiced frames. Wording, emotion, and microphone matter; there is no universal feminine pitch."], ["F1–F3 / vowel focus", "Descriptors affected by vowels, language, and speaker. One mixed-speech value cannot reveal precise mouth or larynx position."], ["Resonance / brightness / spectral tilt", "Spectral-energy proxies, not body locations, vocal-fold status, or health checks."], ["Breathiness", "A spectral proxy that cannot measure airflow or vocal-fold behavior and has no universal safe target band."], ["Rate / pauses / intonation", "Strongly shaped by language, meaning, emotion, and personal style—not gender or health standards."], ["Voice age", "A playful model impression, not actual age, biological age, or a health assessment."]],
    logTitle: "What progress can mean", log: "Log the date, phrase, device, comfort, authenticity, preference, and repeatability. Compare under the same device and distance. If discomfort increases, stop and return to ordinary speech rather than chasing the score.",
    sourcesTitle: "Clinical sources and further reading", sources: ["ASHA: Gender-Affirming Voice and Communication", "NIDCD: Taking Care of Your Voice", "NIDCD: Hoarseness", "UCSF: Vocal Health Guidelines"],
  },
  ja: {
    title: "女性的な声のマニュアル", eyebrow: "安全第一 · 非臨床的な声の探索",
    intro: "録音を比較し、快適で自分らしく、再現しやすい表現を見つけるための手引きです。医療助言や音声治療プログラムではなく、単一の数値が女性的な声を定義することもありません。",
    firstTitle: "始める前に",
    first: ["目標は本人が決めます。女性的な表現にはピッチ、イントネーション、明瞭さ、速度、音量、対話スタイルなどが含まれることも、含まれないこともあります。", "VPA の割合と詳細は、文、言語、母音、マイク、部屋に左右されるモデル／音響推定です。性自認、声帯の健康、正しい発声法は判定できません。", "同条件の録音比較には使えますが、ジェンダー肯定的音声支援の経験がある言語聴覚士の代わりにはなりません。"],
    stopTitle: "中止と受診の目安", stopLead: "痛み、かすれ、締め付け、強い努力感、急な音域変化、急な声の消失があれば発声練習を中止し、声を休めてください。ハミング、ささやき声、別の発声練習に切り替えないでください。", stopCare: "かすれが3週間を超える場合、または発声・嚥下時の痛み、呼吸・嚥下困難、首のしこり、喀血がある場合は耳鼻咽喉科を受診してください。声を変えたい場合は、音声とジェンダー肯定的支援の経験がある言語聴覚士も検討してください。",
    sessionTitle: "安全な自己比較の流れ",
    session: [["1. 自然な基準を録る", "まず声を変えず、普段の音量で短い文を話します。再生して、快適・自分らしい・再現しやすいか確認します。"], ["2. 一つだけ小さく変える", "完全に快適な時だけ、明瞭さ、速度、同じ文のイントネーションなど一要素を比較します。複数の指標を同時に追わないでください。"], ["3. 数値より先に聴く", "VPA を見る前に自分の快適さと好みで判断します。点数が高くても努力感が増えた録音は良い結果ではありません。"], ["4. 再現できるものを残す", "快適さ、自分らしさ、自分の目標、後から不調がないことを優先します。点数はモデル差だけを記録します。"]],
    avoidTitle: "自己流で行わないこと", avoid: ["嚥下、首への手圧、喉頭を固定する方法で声を探さない。", "極端な高低音、圧迫、息漏れ、叫び、ささやきでアプリ指標を追わない。", "胸・マスク・頭部共鳴を実際の解剖位置と考えない。VPA は低・中・高域のスペクトル代理指標を表示します。", "連続発話の F1〜F3 から舌、顎、声帯の動き、病気を自己判断しない。"],
    metricsTitle: "VPA 詳細の読み方",
    metrics: [["女性的傾向", "今回の録音に対するモデル分類傾向。本人の性別でも理想点でもありません。"], ["ピッチ", "検出された有声音の統計。文、感情、マイクで変わり、普遍的な女性ピッチはありません。"], ["F1〜F3／母音フォーカス", "母音、言語、話者に左右される記述。連続発話の一値で口腔・喉頭位置は分かりません。"], ["共鳴／明るさ／スペクトル傾斜", "スペクトルエネルギーの代理指標。体内位置、声帯状態、健康検査ではありません。"], ["息漏れ感", "呼気や声帯動作を測れないスペクトル代理指標で、普遍的な安全目標帯はありません。"], ["速度／間／イントネーション", "言語、意味、感情、個人スタイルに強く左右され、性別や健康の基準ではありません。"], ["声年齢", "遊びとしてのモデル印象で、実年齢、生物学的年齢、健康評価ではありません。"]],
    logTitle: "進歩の考え方", log: "日付、文、端末、快適さ、自分らしさ、好み、再現性を記録します。同じ端末と距離で比較し、不快感が増えたら点数を追わず中止して普段の話し方に戻します。",
    sourcesTitle: "専門資料と参考文献", sources: ["ASHA：ジェンダー肯定的な声とコミュニケーション", "NIDCD：声のケア", "NIDCD：かすれ", "UCSF：声の健康ガイド"],
  },
};

function list(items) {
  return `<ul>${items.map((item) => `<li>${item}</li>`).join("")}</ul>`;
}

function cards(items) {
  return `<div class="manual-safe-grid">${items.map(([title, body]) => `<article><h3>${title}</h3><p>${body}</p></article>`).join("")}</div>`;
}

function sources(t) {
  const links = [
    [SOURCE_URLS.asha, t.sources[0]],
    [SOURCE_URLS.nidcdCare, t.sources[1]],
    [SOURCE_URLS.nidcdHoarseness, t.sources[2]],
    [SOURCE_URLS.ucsf, t.sources[3]],
  ];
  return `<section class="manual-source-panel"><h2>${t.sourcesTitle}</h2>${links.map(([url, label]) => `<a href="${url}" target="_blank" rel="noopener">${label}</a>`).join("")}</section>`;
}

function buildManual(t) {
  return `<div class="manual-safety-manual"><header class="manual-safe-hero"><span>${t.eyebrow}</span><h2>${t.title}</h2><p>${t.intro}</p></header><section><h2>${t.firstTitle}</h2>${list(t.first)}</section><section class="manual-stop-panel"><h2>${t.stopTitle}</h2><p><strong>${t.stopLead}</strong></p><p>${t.stopCare}</p></section><section><h2>${t.sessionTitle}</h2>${cards(t.session)}</section><section class="manual-avoid-panel"><h2>${t.avoidTitle}</h2>${list(t.avoid)}</section><section><h2>${t.metricsTitle}</h2>${cards(t.metrics)}</section><section><h2>${t.logTitle}</h2><p>${t.log}</p></section>${sources(t)}</div>`;
}

export const MANUAL_DATA = Object.fromEntries(
  Object.entries(COPY).map(([locale, copy]) => [locale, { title: copy.title, html: buildManual(copy) }]),
);

export { SOURCE_URLS as MANUAL_SOURCE_URLS };
