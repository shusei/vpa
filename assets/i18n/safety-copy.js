import { buildAnalysisGuidance } from "./analysis-guidance.js";

const SOURCES = {
  asha: "https://www.asha.org/practice-portal/professional-issues/gender-affirming-voice-and-communication/",
  nidcdCare: "https://www.nidcd.nih.gov/health/taking-care-your-voice",
  nidcdHoarseness: "https://www.nidcd.nih.gov/health/hoarseness",
  ucsf: "https://transcare.ucsf.edu/guidelines/vocal-health",
};

const AUTHOR_PORTRAIT_URL = new URL("../avatar-evelyn.jpg", import.meta.url).href;

const HELP_UI = {
  "zh-Hant": {
    title: "Voice Presentation Analyzer 使用指南",
    lead: "從錄音、即時監控到結果閱讀，一次看懂每個畫面；所有數值都只用於比較錄音，不是性別或健康診斷。",
    top: "回到頁首", close: "關閉使用指南", toc: "快速導覽",
    quick: "快速開始", interface: "介面導覽與即時監控", panels: "結果面板怎麼看", scope: "用途與界線",
    step1: "在安靜環境用自然音量錄一小段話，或上傳裝置中的音訊／影片；原始音訊不會上傳。",
    step2: "錄音時可看音高走勢、音量與頻譜代理指標；停止後由瀏覽器在裝置上分析。",
    step3: "先重播並確認聲音舒服，再看模型傾向與統計；不要為了提高分數勉強改變聲音。",
    record: "錄音與上傳：停止錄音或選好檔案後會自動分析；較長錄音需要更多裝置記憶體與時間。",
    live: "即時監控：音高、音量、F1–F3 與頻段能量都會受語句、母音、麥克風與背景噪音影響。",
    privacy: "隱私：分析與重播留在目前裝置；只有使用者主動分享時才會產生分享內容。",
    model: "女性化傾向是模型對這次錄音的分類，不代表性別認同，也沒有人人適用的理想分數。",
    stats: "統計與進階細項適合在相同設備、距離與語句下比較趨勢，不能判斷聲帶健康或正確發聲法。",
    age: "聲齡是娛樂性的非臨床模型印象，不是實際年齡、生理年齡或健康評估。",
  },
  "zh-Hans": {
    title: "Voice Presentation Analyzer 使用指南",
    lead: "从录音、实时监控到结果阅读，一次看懂每个画面；所有数值都只用于比较录音，不是性别或健康诊断。",
    top: "回到页首", close: "关闭使用指南", toc: "快速导览",
    quick: "快速开始", interface: "界面导览与实时监控", panels: "结果面板怎么看", scope: "用途与界限",
    step1: "在安静环境用自然音量录一小段话，或上传设备中的音频／视频；原始音频不会上传。",
    step2: "录音时可看音高走势、音量与频谱代理指标；停止后由浏览器在设备上分析。",
    step3: "先回放并确认声音舒服，再看模型倾向与统计；不要为了提高分数勉强改变声音。",
    record: "录音与上传：停止录音或选好文件后会自动分析；较长录音需要更多设备内存与时间。",
    live: "实时监控：音高、音量、F1–F3 与频段能量都会受语句、元音、麦克风与背景噪音影响。",
    privacy: "隐私：分析与回放留在当前设备；只有用户主动分享时才会生成分享内容。",
    model: "女性化倾向是模型对这次录音的分类，不代表性别认同，也没有人人适用的理想分数。",
    stats: "统计与进阶细项适合在相同设备、距离与语句下比较趋势，不能判断声带健康或正确发声方法。",
    age: "声龄是娱乐性的非临床模型印象，不是实际年龄、生理年龄或健康评估。",
  },
  en: {
    title: "Voice Presentation Analyzer – Help",
    lead: "A guided tour from recording and live monitors to result reading. Every value compares recordings; none diagnoses gender or vocal health.",
    top: "Back to top", close: "Close help", toc: "Quick navigation",
    quick: "Quick start", interface: "Interface tour & live monitors", panels: "How to read the result panels", scope: "Purpose and limits",
    step1: "Record a short sample at a natural volume in a quieter space, or choose an audio/video file on your device. Raw audio is not uploaded.",
    step2: "During recording you can view pitch, level, and spectral proxies. After stopping, analysis runs in the browser on your device.",
    step3: "Replay first and check that the take felt comfortable, then read model tendencies and statistics. Never force a change to raise a score.",
    record: "Record and upload: analysis starts after recording stops or a file is selected. Longer recordings need more device memory and time.",
    live: "Live monitors: pitch, level, F1–F3, and band energy all change with wording, vowels, microphone, and background noise.",
    privacy: "Privacy: analysis and replay stay on this device. Share content is created only when the user actively chooses to share.",
    model: "The feminine-tendency percentage is a model classification of this recording—not gender identity—and has no universal ideal score.",
    stats: "Statistics and advanced details are useful for like-for-like trends. They cannot assess vocal-fold health or identify a correct vocal technique.",
    age: "Voice age is an entertainment-oriented, non-clinical model impression—not actual, biological, or health age.",
  },
  ja: {
    title: "Voice Presentation Analyzer 使い方ガイド",
    lead: "録音、ライブ表示、結果の読み方を順に案内します。すべての数値は録音比較用で、性別や声の健康の診断ではありません。",
    top: "ページ上部へ", close: "使い方ガイドを閉じる", toc: "クイックナビ",
    quick: "クイックスタート", interface: "画面ガイドとライブモニター", panels: "パネルの見方", scope: "用途と限界",
    step1: "静かな場所で自然な音量の短い発話を録音するか、端末内の音声／動画を選びます。元の音声はアップロードされません。",
    step2: "録音中はピッチ、レベル、スペクトル代理指標を確認できます。停止後、端末のブラウザー内で分析します。",
    step3: "まず再生して無理なく発声できたか確認し、その後にモデル傾向と統計を見ます。点数のために声を無理に変えないでください。",
    record: "録音とアップロード：録音停止またはファイル選択後に分析が始まります。長い録音ほど端末のメモリと時間が必要です。",
    live: "ライブモニター：ピッチ、レベル、F1〜F3、帯域エネルギーは文、母音、マイク、背景雑音で変化します。",
    privacy: "プライバシー：分析と再生はこの端末内に残ります。共有内容は利用者が共有を選んだときだけ作成されます。",
    model: "女性らしさ傾向は今回の録音に対するモデル分類で、性自認ではなく、万人共通の理想点もありません。",
    stats: "統計と詳細項目は同じ条件での傾向比較に使えますが、声帯の健康や正しい発声法は判断できません。",
    age: "声年齢は娯楽的な非臨床モデル印象で、実年齢、生物学的年齢、健康評価ではありません。",
  },
};

const PREFLIGHT_UI = {
  "zh-Hant": {
    title: "錄音前舒適檢查",
    summary: "約 30 秒，先確認聲音與環境都適合。",
    intro: "錄音前依序確認三件事；這不是發聲訓練，也不需要改變你的聲音。",
    callout: "提示：可錄音或上傳 mp3 / m4a / mp4 / mov，也能直接拖曳檔案。停止後會自動分析並提供重播、模型傾向與統計；錄音前可展開「錄音前舒適檢查」，需要導覽請點右上角 ❓。",
    shortcutHtml: '回到錄音鍵下方的<a href="#warmupCard">錄音前舒適檢查</a>，每次測驗前都能快速確認。',
    steps: {
      hum: { name: "平常說一句", desc: "：用日常音量說一小句，不先調整音高、共鳴或氣聲。" },
      yi: { name: "確認感受", desc: "：確認沒有疼痛、沙啞、緊繃或明顯費力；有任何不適就停止。" },
      hu: { name: "固定環境", desc: "：安靜環境中將手機放在前方約 30–60 公分，避免遮住麥克風，前後保持相近位置。" },
    },
  },
  "zh-Hans": {
    title: "录音前舒适检查",
    summary: "约 30 秒，先确认声音与环境都合适。",
    intro: "录音前依次确认三件事；这不是发声训练，也不需要改变你的声音。",
    callout: "提示：可录音或上传 mp3 / m4a / mp4 / mov，也能直接拖放文件。停止后会自动分析并提供回放、模型倾向与统计；录音前可展开“录音前舒适检查”，需要导览请点右上角 ❓。",
    shortcutHtml: '回到录音键下方的<a href="#warmupCard">录音前舒适检查</a>，每次测试前都能快速确认。',
    steps: {
      hum: { name: "平常说一句", desc: "：用日常音量说一小句，不先调整音高、共鸣或气声。" },
      yi: { name: "确认感受", desc: "：确认没有疼痛、沙哑、紧绷或明显费力；有任何不适就停止。" },
      hu: { name: "固定环境", desc: "：安静环境中将手机放在前方约 30–60 厘米，避免遮住麦克风，前后保持相近位置。" },
    },
  },
  en: {
    title: "Pre-recording comfort check",
    summary: "About 30 seconds to check your voice and setup.",
    intro: "Check these three items before recording. This is not a vocal exercise, and you do not need to change your voice.",
    callout: "Tip: record, upload an mp3 / m4a / mp4 / mov, or drop a file here. Analysis starts after stopping and provides replay, model tendencies, and statistics. Open the pre-recording comfort check first, or use ❓ for a guided tour.",
    shortcutHtml: 'Return to the <a href="#warmupCard">pre-recording comfort check</a> below the record button before any take.',
    steps: {
      hum: { name: "Speak one natural line", desc: ": Use your everyday volume without first changing pitch, resonance, or breathiness." },
      yi: { name: "Check how it feels", desc: ": Make sure there is no pain, hoarseness, tightness, or marked effort. Stop if any appears." },
      hu: { name: "Match the setup", desc: ": In a quiet room, place the phone about 30–60 cm in front of you, keep its microphones uncovered, and match the position between takes." },
    },
  },
  ja: {
    title: "録音前の快適さチェック",
    summary: "約30秒で、声と録音環境を確認します。",
    intro: "録音前に3点を確認します。これは発声練習ではなく、声を変える必要もありません。",
    callout: "ヒント：録音、mp3 / m4a / mp4 / mov の選択、またはドラッグ＆ドロップが使えます。停止後に自動分析し、再生、モデル傾向、統計を表示します。先に「録音前の快適さチェック」を開くか、❓でガイドを確認できます。",
    shortcutHtml: '録音ボタン下の<a href="#warmupCard">録音前の快適さチェック</a>へ戻り、各テイクの前に確認できます。',
    steps: {
      hum: { name: "普段の声で一文", desc: "：ピッチ、共鳴、息漏れ感を先に変えず、日常の音量で短く話します。" },
      yi: { name: "感覚を確認", desc: "：痛み、かすれ、締め付け、強い努力感がないか確認し、一つでもあれば中止します。" },
      hu: { name: "条件をそろえる", desc: "：静かな場所で端末を正面約 30〜60 cm に置き、マイクを塞がず、テイク間で位置をそろえます。" },
    },
  },
};

const MANUAL_TITLES = {
  "zh-Hant": "女性化聲音手冊",
  "zh-Hans": "女性化声音手册",
  en: "Feminine Voice Manual",
  ja: "女性的な声のマニュアル",
};

const COPY = {
  "zh-Hant": {
    guideTagline: "安全優先的聲音探索；結果是聲學與模型估計，不是醫療判斷。",
    helpSubtitle: "所有分析都在瀏覽器內完成。請把數值當作比較不同錄音的參考，而不是性別、健康或發聲方式的診斷。",
    helpStep3: "先重播並確認聲音舒服，再查看模型傾向、統計與非臨床觀察；不必為了提高分數勉強改變聲音。",
    helpFormant: "顯示 F1–F3 與頻譜代理指標；它們受母音、語言、麥克風與錄音環境影響。",
    helpStats: "整理音高、音量、訊噪比與聲學代理指標；內容僅供自我比較，不是治療處方。",
    helpPosition: "用途：個人聲音探索、錄音比較、教育示範與研究原型。目標應由使用者自己決定。",
    helpWarning: "不能判定性別認同、聲帶健康、實際年齡或任何醫療狀況，也不能取代耳鼻喉科醫師或具聲音專長的語言治療師。",
    safetyBadge: "安全使用",
    safetyTitle: "先以舒適為準，不要追著分數練",
    safetyScope: "百分比、音高、共振、氣聲與聲齡都是受錄音內容和設備影響的非臨床估計；沒有任何單一數值能定義女性聲音或你的性別。",
    safetyStop: "若出現疼痛、沙啞、緊繃、明顯費力或突然的聲音改變，請停止發聲練習並休息，不要用任何提示硬撐。",
    safetyCare: "沙啞超過三週，或伴隨說話／吞嚥疼痛、呼吸或吞嚥困難、頸部腫塊、咳血等情況，應尋求耳鼻喉科評估；需要調整聲音時，優先尋找具性別肯認聲音經驗的語言治療師。",
    sourcePrefix: "依據與延伸閱讀",
    sourceAsha: "ASHA 性別肯認聲音與溝通",
    sourceNidcd: "NIDCD 聲音照護",
    sourceHoarseness: "NIDCD 沙啞警訊",
    sourceUcsf: "UCSF 聲音健康指南",
    advBeta: "ADVANCED BETA · 聲學估計",
    advTitle: "進階聲學分析",
    advScore: "模型女性化傾向",
    confidence: "估計可信度",
    disclaimer: "這是實驗性的聲學與模型回饋，不是醫療診斷、治療處方、實際年齡或性別認定。錄音只在你的裝置上處理。",
    pitchHint: "錄音中偵測到的有聲片段之中位音高；受語句、情緒與麥克風影響，並非目標值。",
    ageTitle: "實驗性聲齡印象",
    ageEyebrow: "VOICE AGE 2.0 · 非臨床研究預覽",
    ageEvidence: "聲齡印象的聲學證據",
    ageUsed: "通過品質門檻；僅供模型估計",
    ageReference: "語音參考；不代表實際年齡或健康",
    ageNote: "結果會受語句、母音、麥克風、噪音與當下發聲狀態影響；不能推斷實際年齡、聲帶狀況或疾病。",
    insightLabel: "非臨床觀察",
    insights: {
      balancedGrowth: "這次錄音的模型與聲學指標較一致。若感覺自然舒適，可把它留作日後比較的參考。",
      consistencyOpportunity: "不同片段的指標有變化。可在相同距離、自然音量與相近語句下再錄一次，以判斷是表現差異還是錄音條件造成。",
      falsettoContrast: "音高與頻譜呈現不同方向；母音、語句、麥克風角度都可能造成這種差異。不要為了讓它們一致而勉強發聲。",
      insufficient: "可用的有聲資料不足。若喉嚨舒適，可在較安靜的環境以自然音量連續說 5–8 秒。",
      pitchOpportunity: "頻譜指標較穩定，但音高會隨語句自然改變。只需觀察是否符合自己的溝通目標，不必追求固定範圍。",
      resonanceOpportunity: "音高與頻譜趨勢略有差異，可能來自母音或錄音條件。可用同一句話、相同設備再比較。",
      strongIntegration: "這次的音高與頻譜趨勢較一致。記錄當時是否舒服、自然與可重複，比追求更高分重要。",
    },
    quickFooter: "分析只在你的裝置上進行。結果供娛樂與自我比較，不定義身分，也不是醫療或訓練處方。",
    quickRefineEyebrow: "聽聽並比較",
    quickRefineTitle: "保留自然、舒服的一次",
    quickRetryAria: "重新測試並比較不同錄音",
    quickRetryHint: "換一句或在相同條件下比較",
    quickBehind: "這次比朋友的模型傾向低 {{difference}} 個百分點；錄音內容和設備也會影響結果。",
    quickTrust: "實驗性綜合估計",
    quickInsight: "非臨床觀察",
    quickSafetyTitle: "舒服比高分重要",
    quickSafetyBody: "小建議只解釋這次錄音的聲學差異，不是醫療或訓練指示；沒有固定的女性聲音數值。",
    quickSafetyStop: "疼痛、沙啞、緊繃或費力時請停止，不要為了分數硬撐。",
    modelHint: "這是模型在本次錄音上的傾向，不是身分或健康判定，也沒有通用的理想分數。",
    proxyHint: "這是受語句、母音、設備與環境影響的聲學代理指標；適合在相同條件下比較，不能判斷身體中的共鳴位置或發聲健康。",
    formantHint: "F1–F3 會隨母音、語言與說話者改變；混合語音無法用單一數值精確推斷舌頭、下顎或喉部位置。",
    breathHint: "此數值只是頻譜代理指標，不能測量漏氣、聲帶閉合或健康，也沒有通用的理想區間。不要刻意擠壓或增加氣聲來追數值。",
    paceHint: "語速與停頓受語言、語意、情緒與個人風格影響，不是性別或健康指標；以清楚、自然、能舒服呼吸為準。",
    intonationHint: "語調會隨句型、語意、語言與個人風格改變；上揚或下降都不是女性化的必要條件。",
    focusHeading: "本次錄音的主要差異",
    focusEmpty: "目前沒有特別突出的模型差異；可依自然度、舒適度與可重複性決定是否保留。",
    focusSafety: "只比較錄音條件與聲學趨勢；不要依卡片自行做聲帶閉合、喉位或醫療訓練。",
    severity: ["較明顯的模型訊號", "次要模型訊號", "細微模型訊號"],
  },
  "zh-Hans": {
    guideTagline: "安全优先的声音探索；结果是声学与模型估计，不是医疗判断。",
    helpSubtitle: "所有分析都在浏览器内完成。请把数值当作比较不同录音的参考，而不是性别、健康或发声方式的诊断。",
    helpStep3: "先回放并确认声音舒服，再查看模型倾向、统计与非临床观察；不必为了提高分数勉强改变声音。",
    helpFormant: "显示 F1–F3 与频谱代理指标；它们受元音、语言、麦克风与录音环境影响。",
    helpStats: "整理音高、音量、信噪比与声学代理指标；内容仅供自我比较，不是治疗处方。",
    helpPosition: "用途：个人声音探索、录音比较、教育演示与研究原型。目标应由用户自己决定。",
    helpWarning: "不能判断性别认同、声带健康、实际年龄或任何医疗状况，也不能取代耳鼻喉科医生或具声音专长的言语治疗师。",
    safetyBadge: "安全使用", safetyTitle: "先以舒适为准，不要追着分数练",
    safetyScope: "百分比、音高、共振、气声与声龄都是受录音内容和设备影响的非临床估计；没有任何单一数值能定义女性声音或你的性别。",
    safetyStop: "若出现疼痛、沙哑、紧绷、明显费力或突然的声音改变，请停止发声练习并休息，不要用任何提示硬撑。",
    safetyCare: "沙哑超过三周，或伴随说话／吞咽疼痛、呼吸或吞咽困难、颈部肿块、咳血等情况，应寻求耳鼻喉科评估；需要调整声音时，优先寻找有性别肯定声音经验的言语治疗师。",
    sourcePrefix: "依据与延伸阅读", sourceAsha: "ASHA 性别肯定声音与沟通", sourceNidcd: "NIDCD 声音照护", sourceHoarseness: "NIDCD 沙哑警讯", sourceUcsf: "UCSF 声音健康指南",
    advBeta: "ADVANCED BETA · 声学估计", advTitle: "进阶声学分析", advScore: "模型女性化倾向", confidence: "估计可信度",
    disclaimer: "这是实验性的声学与模型反馈，不是医疗诊断、治疗处方、实际年龄或性别认定。录音只在你的设备上处理。",
    pitchHint: "录音中检测到的有声片段之中位音高；受语句、情绪与麦克风影响，并非目标值。",
    ageTitle: "实验性声龄印象", ageEyebrow: "VOICE AGE 2.0 · 非临床研究预览", ageEvidence: "声龄印象的声学证据", ageUsed: "通过质量门槛；仅供模型估计", ageReference: "语音参考；不代表实际年龄或健康", ageNote: "结果会受语句、元音、麦克风、噪音与当下发声状态影响；不能推断实际年龄、声带状况或疾病。",
    insightLabel: "非临床观察",
    insights: {
      balancedGrowth: "这次录音的模型与声学指标较一致。若感觉自然舒适，可把它留作日后比较的参考。",
      consistencyOpportunity: "不同片段的指标有变化。可在相同距离、自然音量与相近语句下再录一次，以判断是表现差异还是录音条件造成。",
      falsettoContrast: "音高与频谱呈现不同方向；元音、语句、麦克风角度都可能造成这种差异。不要为了让它们一致而勉强发声。",
      insufficient: "可用的有声数据不足。若喉咙舒适，可在较安静的环境以自然音量连续说 5–8 秒。",
      pitchOpportunity: "频谱指标较稳定，但音高会随语句自然改变。只需观察是否符合自己的沟通目标，不必追求固定范围。",
      resonanceOpportunity: "音高与频谱趋势略有差异，可能来自元音或录音条件。可用同一句话、相同设备再比较。",
      strongIntegration: "这次的音高与频谱趋势较一致。记录当时是否舒服、自然与可重复，比追求更高分重要。",
    },
    quickFooter: "分析只在你的设备上进行。结果供娱乐与自我比较，不定义身份，也不是医疗或训练处方。",
    quickRefineEyebrow: "听听并比较", quickRefineTitle: "保留自然、舒服的一次", quickRetryAria: "重新测试并比较不同录音", quickRetryHint: "换一句或在相同条件下比较", quickBehind: "这次比朋友的模型倾向低 {{difference}} 个百分点；录音内容和设备也会影响结果。", quickTrust: "实验性综合估计", quickInsight: "非临床观察",
    quickSafetyTitle: "舒服比高分重要", quickSafetyBody: "小建议只解释这次录音的声学差异，不是医疗或训练指示；没有固定的女性声音数值。", quickSafetyStop: "疼痛、沙哑、紧绷或费力时请停止，不要为了分数硬撑。",
    modelHint: "这是模型在本次录音上的倾向，不是身份或健康判断，也没有通用的理想分数。",
    proxyHint: "这是受语句、元音、设备与环境影响的声学代理指标；适合在相同条件下比较，不能判断身体中的共鸣位置或发声健康。",
    formantHint: "F1–F3 会随元音、语言与说话者改变；混合语音无法用单一数值精确推断舌头、下颌或喉部位置。",
    breathHint: "此数值只是频谱代理指标，不能测量漏气、声带状态或健康，也没有通用的理想区间。不要刻意挤压或增加气声来追数值。",
    paceHint: "语速与停顿受语言、语义、情绪与个人风格影响，不是性别或健康指标；以清楚、自然、能舒服呼吸为准。",
    intonationHint: "语调会随句型、语义、语言与个人风格改变；上扬或下降都不是女性化的必要条件。",
    focusHeading: "本次录音的主要差异", focusEmpty: "目前没有特别突出的模型差异；可依自然度、舒适度与可重复性决定是否保留。", focusSafety: "只比较录音条件与声学趋势；不要依卡片自行做声带、喉位或医疗训练。", severity: ["较明显的模型信号", "次要模型信号", "细微模型信号"],
  },
  en: {
    guideTagline: "Safety-first voice exploration: results are acoustic and model estimates, not medical findings.",
    helpSubtitle: "All analysis runs in your browser. Treat values as comparisons between recordings—not diagnoses of gender, vocal health, or vocal technique.",
    helpStep3: "Replay the take and check that it felt comfortable before reading model tendencies, statistics, and non-clinical observations. Never force a change to raise a score.",
    helpFormant: "Shows F1–F3 and spectral proxies, all influenced by vowels, language, microphone, and recording conditions.",
    helpStats: "Summarizes pitch, level, signal-to-noise ratio, and acoustic proxies for self-comparison—not treatment instructions.",
    helpPosition: "Purpose: self-directed voice exploration, recording comparison, education, and research prototyping. The user defines their own goals.",
    helpWarning: "Cannot determine gender identity, vocal-fold health, actual age, or any medical condition, and cannot replace an ENT/laryngologist or a voice-specialized speech-language pathologist.",
    safetyBadge: "USE SAFELY", safetyTitle: "Comfort comes first—do not train to the score",
    safetyScope: "Percentages, pitch, resonance, breathiness, and voice-age impressions are non-clinical estimates affected by the recording and device. No single value defines a feminine voice or your gender.",
    safetyStop: "Stop voice practice and rest your voice if you notice pain, hoarseness, tightness, marked effort, or a sudden voice change. Do not push through a warning sign to follow a card.",
    safetyCare: "Seek an ENT/laryngologist for hoarseness lasting more than three weeks, or sooner for pain with speaking or swallowing, breathing or swallowing difficulty, a neck lump, or coughing blood. For voice change, consider a speech-language pathologist experienced in gender-affirming voice care.",
    sourcePrefix: "Evidence and further reading", sourceAsha: "ASHA: Gender-Affirming Voice and Communication", sourceNidcd: "NIDCD: Taking Care of Your Voice", sourceHoarseness: "NIDCD: Hoarseness", sourceUcsf: "UCSF: Vocal Health Guidelines",
    advBeta: "ADVANCED BETA · ACOUSTIC ESTIMATE", advTitle: "Advanced acoustic analysis", advScore: "Model feminine tendency", confidence: "Estimate confidence",
    disclaimer: "Experimental acoustic and model feedback—not a medical diagnosis, treatment prescription, actual-age estimate, or gender determination. Audio is processed only on your device.",
    pitchHint: "Median pitch among detected voiced frames in this recording. It changes with wording, emotion, and microphone conditions and is not a target.",
    ageTitle: "Experimental voice-age impression", ageEyebrow: "VOICE AGE 2.0 · NON-CLINICAL RESEARCH PREVIEW", ageEvidence: "Acoustic evidence for the voice-age impression", ageUsed: "Passed the quality gate; used only by the model", ageReference: "Speech reference only; not actual age or health", ageNote: "Wording, vowels, microphone, noise, and momentary voice state affect this result. It cannot reveal actual age, vocal-fold condition, or disease.",
    insightLabel: "Non-clinical observation",
    insights: {
      balancedGrowth: "The model and acoustic indicators are relatively aligned in this take. If it felt natural and comfortable, save it as a comparison reference.",
      consistencyOpportunity: "Indicators varied across the take. Record again at the same distance, natural volume, and similar wording to separate performance variation from recording conditions.",
      falsettoContrast: "Pitch and spectral indicators trend differently. Vowels, wording, and microphone angle can all cause this; do not force your voice to make them match.",
      insufficient: "There is not enough voiced data. If your throat feels comfortable, speak continuously for 5–8 seconds at a natural volume in a quieter setting.",
      pitchOpportunity: "Spectral indicators are steadier while pitch varies naturally with wording. Compare this with your own communication goal rather than chasing a fixed range.",
      resonanceOpportunity: "Pitch and spectral trends differ slightly, possibly because of vowels or recording conditions. Compare another take using the same sentence and device.",
      strongIntegration: "Pitch and spectral trends align more closely in this take. Note whether it felt comfortable, authentic, and repeatable—those matter more than a higher score.",
    },
    quickFooter: "Analysis stays on your device. Results are for play and self-comparison; they do not define identity and are not medical or training prescriptions.",
    quickRefineEyebrow: "LISTEN AND COMPARE", quickRefineTitle: "Keep the take that feels natural", quickRetryAria: "Test again and compare recordings", quickRetryHint: "Try another line or match the recording conditions", quickBehind: "This take is {{difference}} points lower than your friend's model tendency; wording and device conditions can also change the result.", quickTrust: "Experimental composite estimate", quickInsight: "Non-clinical observation",
    quickSafetyTitle: "Comfort matters more than the score", quickSafetyBody: "This note only describes acoustic differences in the recording. It is not medical or training guidance, and there is no fixed feminine-voice target.", quickSafetyStop: "Stop if you notice pain, hoarseness, tightness, or effort. Never push for a score.",
    modelHint: "This is the model tendency for this recording—not an identity or health judgment. There is no universal ideal score.",
    proxyHint: "This acoustic proxy changes with wording, vowels, device, and room. Use it for like-for-like comparisons; it cannot locate resonance in the body or assess vocal health.",
    formantHint: "F1–F3 vary with vowels, language, and speaker. Mixed speech cannot use one value to infer precise tongue, jaw, or larynx position.",
    breathHint: "This spectral proxy cannot measure airflow, vocal-fold behavior, or health, and has no universal ideal band. Do not force extra air or compression to chase it.",
    paceHint: "Rate and pauses depend on language, meaning, emotion, and personal style—not gender or health. Prioritize clarity, naturalness, and comfortable breathing.",
    intonationHint: "Intonation changes with sentence type, meaning, language, and personal style. Rising and falling endings are both valid and neither is required for femininity.",
    focusHeading: "Main differences in this take", focusEmpty: "No model difference stands out. Decide whether to keep the take by comfort, authenticity, and repeatability.", focusSafety: "Compare recording conditions and acoustic trends only. Do not use these cards for self-directed vocal-fold, larynx, or medical training.", severity: ["Stronger model signal", "Secondary model signal", "Subtle model signal"],
  },
  ja: {
    guideTagline: "安全を優先した声の探索。結果は音響・モデル推定であり、医療判断ではありません。",
    helpSubtitle: "分析はブラウザー内で完結します。数値は録音同士の比較用で、性別・声の健康・発声法の診断ではありません。",
    helpStep3: "まず再生して無理なく発声できたか確認し、その後にモデル傾向、統計、非臨床的な所見を見ます。点数を上げるために声を無理に変えないでください。",
    helpFormant: "F1〜F3 とスペクトル代理指標を表示します。母音、言語、マイク、録音環境の影響を受けます。",
    helpStats: "ピッチ、音量、S/N 比、音響代理指標を自己比較用にまとめます。治療の指示ではありません。",
    helpPosition: "目的：本人が決めた声の探索、録音比較、教育デモ、研究プロトタイプ。目標は利用者自身が決めます。",
    helpWarning: "性自認、声帯の健康、実年齢、病気を判定できず、耳鼻咽喉科医や音声を専門とする言語聴覚士の代わりにはなりません。",
    safetyBadge: "安全に使う", safetyTitle: "快適さを最優先し、点数を目標に練習しない",
    safetyScope: "割合、ピッチ、共鳴、息漏れ感、声年齢は録音や端末に左右される非臨床的推定です。単一の数値が女性らしい声や性別を定義することはありません。",
    safetyStop: "痛み、かすれ、締め付け、強い努力感、急な声の変化があれば発声練習を中止し、声を休めてください。カードに従うために無理をしないでください。",
    safetyCare: "かすれが3週間を超える場合、または発声・嚥下時の痛み、呼吸・嚥下困難、首のしこり、喀血がある場合は耳鼻咽喉科を受診してください。声を変えたい場合はジェンダー肯定的音声支援の経験がある言語聴覚士も検討してください。",
    sourcePrefix: "根拠と参考資料", sourceAsha: "ASHA：ジェンダー肯定的な声とコミュニケーション", sourceNidcd: "NIDCD：声のケア", sourceHoarseness: "NIDCD：かすれの受診目安", sourceUcsf: "UCSF：声の健康ガイド",
    advBeta: "ADVANCED BETA · 音響推定", advTitle: "高度な音響分析", advScore: "モデルの女性的傾向", confidence: "推定の信頼度",
    disclaimer: "実験的な音響・モデルフィードバックであり、医療診断、治療処方、実年齢推定、性別判定ではありません。音声は端末内だけで処理されます。",
    pitchHint: "検出された有声音フレームの中央値。文、感情、マイク条件で変化し、目標値ではありません。",
    ageTitle: "実験的な声年齢の印象", ageEyebrow: "VOICE AGE 2.0 · 非臨床研究プレビュー", ageEvidence: "声年齢印象の音響的根拠", ageUsed: "品質条件を通過。モデル推定にのみ使用", ageReference: "音声上の参考。実年齢や健康を示しません", ageNote: "文、母音、マイク、雑音、その時の声の状態に左右されます。実年齢、声帯の状態、病気は分かりません。",
    insightLabel: "非臨床的な所見",
    insights: {
      balancedGrowth: "この録音ではモデルと音響指標が比較的一致しています。自然で快適だった場合は比較用に保存できます。",
      consistencyOpportunity: "録音内で指標が変化しました。同じ距離、自然な音量、似た文で再録音すると、発声差と録音条件を分けて考えやすくなります。",
      falsettoContrast: "ピッチとスペクトル指標の方向が異なります。母音、文、マイク角度でも起こるため、一致させようと無理をしないでください。",
      insufficient: "有声音データが不足しています。喉が快適なら、静かな場所で自然な音量のまま5〜8秒続けて話してください。",
      pitchOpportunity: "スペクトル指標は比較的安定し、ピッチは文に応じて自然に変化しています。固定範囲ではなく自分の目標と比較してください。",
      resonanceOpportunity: "ピッチとスペクトル傾向が少し異なります。母音や録音条件の影響もあるため、同じ文と端末で比較できます。",
      strongIntegration: "この録音ではピッチとスペクトル傾向が近くなっています。高得点よりも、快適・自分らしい・再現しやすいかを記録してください。",
    },
    quickFooter: "分析は端末内だけで行われます。結果は遊びと自己比較用で、本人のアイデンティティを定義せず、医療・訓練処方でもありません。",
    quickRefineEyebrow: "聴いて比較", quickRefineTitle: "自然で快適なテイクを残す", quickRetryAria: "再テストして録音を比較", quickRetryHint: "別の文、または同じ録音条件で比較", quickBehind: "この録音は友だちのモデル傾向より {{difference}} ポイント低めです。文や端末条件でも結果は変わります。", quickTrust: "実験的な複合推定", quickInsight: "非臨床的な所見",
    quickSafetyTitle: "点数より快適さが大切", quickSafetyBody: "小さな提案は録音の音響差を説明するだけで、医療・訓練指示ではありません。女性的な声に固定目標はありません。", quickSafetyStop: "痛み、かすれ、締め付け、努力感があれば中止し、点数のために無理をしないでください。",
    modelHint: "今回の録音に対するモデル傾向で、本人の性別や健康の判断ではありません。普遍的な理想点はありません。",
    proxyHint: "文、母音、端末、部屋で変わる音響代理指標です。同条件の比較に使い、体内の共鳴位置や声の健康の判定には使えません。",
    formantHint: "F1〜F3 は母音、言語、話者で変化します。連続発話の単一値から舌、顎、喉頭の正確な位置は推定できません。",
    breathHint: "このスペクトル代理指標は呼気、声帯の動き、健康を測定できず、普遍的な理想範囲もありません。数値のために息や圧迫を強制しないでください。",
    paceHint: "話速と間は言語、意味、感情、話し方によるもので、性別や健康の指標ではありません。明瞭さ、自然さ、楽な呼吸を優先してください。",
    intonationHint: "イントネーションは文型、意味、言語、話し方で変わります。上昇・下降のどちらも有効で、女性らしさの必須条件ではありません。",
    focusHeading: "今回の録音で目立つ差", focusEmpty: "目立つモデル差はありません。快適さ、自分らしさ、再現しやすさで残すか決めてください。", focusSafety: "録音条件と音響傾向だけを比較してください。カードを声帯・喉頭・医療的な自己訓練に使わないでください。", severity: ["比較的強いモデル信号", "二次的なモデル信号", "小さなモデル信号"],
  },
};

function deepMerge(base, override) {
  if (!override || typeof override !== "object" || Array.isArray(override)) return override;
  const result = { ...(base || {}) };
  Object.entries(override).forEach(([key, value]) => {
    result[key] = value && typeof value === "object" && !Array.isArray(value)
      ? deepMerge(result[key], value)
      : value;
  });
  return result;
}

function metricStates(states, hint, labels = {}) {
  return Object.fromEntries(states.map((state) => [state, {
    ...(labels[state] ? { label: labels[state] } : {}),
    hint,
  }]));
}

function sourceLinks(t) {
  return `<div class="help-evidence-links"><strong>${t.sourcePrefix}</strong><a href="${SOURCES.asha}" target="_blank" rel="noopener">${t.sourceAsha}</a><a href="${SOURCES.nidcdCare}" target="_blank" rel="noopener">${t.sourceNidcd}</a><a href="${SOURCES.nidcdHoarseness}" target="_blank" rel="noopener">${t.sourceHoarseness}</a><a href="${SOURCES.ucsf}" target="_blank" rel="noopener">${t.sourceUcsf}</a></div>`;
}

function helpDialog(t, locale) {
  const h = HELP_UI[locale] || HELP_UI.en;
  return `<div class="help-close-affix"><button type="button" id="helpTop" class="help-top" aria-label="${h.top}">↑</button><button type="button" id="helpClose" class="help-close" aria-label="${h.close}">×</button></div><div class="help-hero"><h2 id="helpTitle">${h.title}</h2><p class="lead">${h.lead}</p><div class="help-pill-row"><span class="help-pill">${h.quick}</span><span class="help-pill">${h.interface}</span><span class="help-pill">${h.panels}</span></div></div><nav class="help-toc" aria-label="${h.toc}"><span>${h.toc}</span><a href="#help-start">${h.quick}</a><a href="#help-live">${h.interface}</a><a href="#help-panels">${h.panels}</a><a href="#help-scope">${h.scope}</a></nav><section class="help-safety-panel"><span>${t.safetyBadge}</span><h3>${t.safetyTitle}</h3><p>${t.safetyScope}</p><p><strong>${t.safetyStop}</strong></p><p>${t.safetyCare}</p>${sourceLinks(t)}</section><section id="help-start" class="help-section accent"><h3>${h.quick}</h3><ol><li>${h.step1}</li><li>${h.step2}</li><li>${h.step3}</li></ol></section><section id="help-live" class="help-section"><h3>${h.interface}</h3><ul><li>${h.record}</li><li>${h.live}</li><li>${h.privacy}</li></ul></section><section id="help-panels" class="help-section"><h3>${h.panels}</h3><ul><li>${h.model}</li><li>${h.stats}</li><li>${h.age}</li></ul></section><section id="help-scope" class="help-section accent"><h3>${h.scope}</h3><p>${t.helpPosition}</p><p><strong>${t.helpWarning}</strong></p></section><div class="help-author"><img src="${AUTHOR_PORTRAIT_URL}" alt="Evelyn" class="help-author-img" loading="lazy" /><div class="help-author-text"><h4>Evelyn</h4><p>Voice Presentation Analyzer</p><a href="https://www.instagram.com/evelynjoelle.lin/" target="_blank" rel="noopener" class="help-social-link"><span class="icon-ig"></span> Instagram · @evelynjoelle.lin</a></div></div><div class="help-support"><a class="help-support-link" href="https://buymeacoffee.com/shusei" target="_blank" rel="noopener"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee" loading="lazy" /></a></div>`;
}

function buildOverride(locale) {
  const t = COPY[locale] || COPY.en;
  const ui = {
    en: {
      appReference: "App reference · recording-dependent", spectralNoTarget: "Spectral proxy · no target",
      resonanceTitle: "Spectral-band balance", lowBand: "Low band {{value}}%", midBand: "Mid band {{value}}%", highBand: "High band {{value}}%",
      pitchComparison: "Pitch comparison", spectralComparison: "Spectral comparison", paceComparison: "Pacing comparison",
      comparePitch: "Compare pitch", compareSpectral: "Compare spectral trend", compareBreath: "Compare spectral proxy", comparePace: "Compare pacing", openGuide: "Open recording guide",
      formantTitle: "Formant & spectral proxies", intonationTitle: "Intonation & pacing descriptors", vowelTitle: "Vowel-reference & spectral proxies",
      resonanceLabels: { insufficient: "Not enough data", balanced: "Even spectral bands", headBright: "More high-band energy", chestHeavy: "More low-band energy", maskLead: "More mid-band energy" },
      vowelLabels: { insufficient: "Not enough data", strong: "More frames in app reference", medium: "Mixed relative pattern", weak: "Fewer frames in app reference" },
    },
    ja: {
      appReference: "アプリ参考帯 · 録音条件で変化", spectralNoTarget: "スペクトル代理指標 · 目標値なし",
      resonanceTitle: "周波数帯エネルギー比", lowBand: "低域 {{value}}%", midBand: "中域 {{value}}%", highBand: "高域 {{value}}%",
      pitchComparison: "ピッチ比較", spectralComparison: "スペクトル比較", paceComparison: "話速比較",
      comparePitch: "ピッチを比較", compareSpectral: "スペクトル傾向を比較", compareBreath: "代理指標を比較", comparePace: "話速を比較", openGuide: "録音ガイドを開く",
      formantTitle: "フォルマントとスペクトル代理指標", intonationTitle: "イントネーションと話速の記述", vowelTitle: "母音参考帯とスペクトル代理指標",
      resonanceLabels: { insufficient: "データ不足", balanced: "周波数帯が均等", headBright: "高域エネルギーが多め", chestHeavy: "低域エネルギーが多め", maskLead: "中域エネルギーが多め" },
      vowelLabels: { insufficient: "データ不足", strong: "参考帯のフレームが多め", medium: "混合した相対パターン", weak: "参考帯のフレームが少なめ" },
    },
    "zh-Hans": {
      appReference: "应用参考带 · 随录音条件改变", spectralNoTarget: "频谱代理指标 · 无目标值",
      resonanceTitle: "频段能量比例", lowBand: "低频 {{value}}%", midBand: "中频 {{value}}%", highBand: "高频 {{value}}%",
      pitchComparison: "音高比较", spectralComparison: "频谱比较", paceComparison: "语速比较",
      comparePitch: "比较音高", compareSpectral: "比较频谱趋势", compareBreath: "比较频谱代理指标", comparePace: "比较语速", openGuide: "打开录音指南",
      formantTitle: "共振峰与频谱代理指标", intonationTitle: "语调与语速描述", vowelTitle: "元音参考带与频谱代理指标",
      resonanceLabels: { insufficient: "数据不足", balanced: "频段能量较平均", headBright: "高频能量较多", chestHeavy: "低频能量较多", maskLead: "中频能量较多" },
      vowelLabels: { insufficient: "数据不足", strong: "较多帧位于应用参考带", medium: "混合的相对分布", weak: "较少帧位于应用参考带" },
    },
    "zh-Hant": {
      appReference: "應用參考帶 · 隨錄音條件改變", spectralNoTarget: "頻譜代理指標 · 無目標值",
      resonanceTitle: "頻段能量比例", lowBand: "低頻 {{value}}%", midBand: "中頻 {{value}}%", highBand: "高頻 {{value}}%",
      pitchComparison: "音高比較", spectralComparison: "頻譜比較", paceComparison: "語速比較",
      comparePitch: "比較音高", compareSpectral: "比較頻譜趨勢", compareBreath: "比較頻譜代理指標", comparePace: "比較語速", openGuide: "開啟錄音指南",
      formantTitle: "共振峰與頻譜代理指標", intonationTitle: "語調與語速描述", vowelTitle: "母音參考帶與頻譜代理指標",
      resonanceLabels: { insufficient: "資料不足", balanced: "頻段能量較平均", headBright: "高頻能量較多", chestHeavy: "低頻能量較多", maskLead: "中頻能量較多" },
      vowelLabels: { insufficient: "資料不足", strong: "較多幀位於應用參考帶", medium: "混合的相對分布", weak: "較少幀位於應用參考帶" },
    },
  }[locale] || null;
  const proxyStates = ["insufficient", "balanced", "headBright", "chestHeavy", "maskLead"];
  const tiltStates = ["insufficient", "warm", "gentleWarm", "balanced", "bright"];
  const breathStates = ["insufficient", "dense", "balanced", "airy", "style", "tooAiry"];
  const brightStates = ["insufficient", "balanced", "warm", "sparkle", "sweet", "sweetMasculine", "sparkleMasculine", "sharp"];
  const pitchBandLabels = {
    "zh-Hant": {
      male: "應用程式男性化音高參考帶（85–165Hz）",
      overlap: "應用程式重疊音高參考帶（165–180Hz）",
      female: "應用程式女性化音高參考帶（180–310Hz）",
    },
    "zh-Hans": {
      male: "应用程序男性化音高参考带（85–165Hz）",
      overlap: "应用程序重叠音高参考带（165–180Hz）",
      female: "应用程序女性化音高参考带（180–310Hz）",
    },
    en: {
      male: "App masculine-presentation pitch reference (85–165 Hz)",
      overlap: "App overlap pitch reference (165–180 Hz)",
      female: "App feminine-presentation pitch reference (180–310 Hz)",
    },
    ja: {
      male: "アプリのマスキュリン提示ピッチ参考帯（85〜165 Hz）",
      overlap: "アプリの重なりピッチ参考帯（165〜180 Hz）",
      female: "アプリのフェミニン提示ピッチ参考帯（180〜310 Hz）",
    },
  }[locale];
  const safetyOverride = {
    guide: { tagline: t.guideTagline, title: MANUAL_TITLES[locale] || MANUAL_TITLES.en },
    callout: { bodyHtml: `<p>${(PREFLIGHT_UI[locale] || PREFLIGHT_UI.en).callout}</p>` },
    practice: { warmup: PREFLIGHT_UI[locale] || PREFLIGHT_UI.en },
    help: {
      subtitle: t.helpSubtitle,
      quickStart: { step3: t.helpStep3 },
      interface: { formantDesc: t.helpFormant, statsDesc: t.helpStats },
      ethics: { position: t.helpPosition, warning: t.helpWarning },
    },
    helpDialog: { dialogHtml: helpDialog(t, locale) },
    pitchBands: pitchBandLabels,
    experiment: {
      advanced: {
        beta: t.advBeta, title: t.advTitle, strictScore: t.advScore,
        confidence: { label: t.confidence }, disclaimer: t.disclaimer,
        pitchMedianHint: t.pitchHint,
        safety: {
          badge: t.safetyBadge, title: t.safetyTitle, scope: t.safetyScope,
          stop: t.safetyStop, care: t.safetyCare, sourcePrefix: t.sourcePrefix,
          sources: { asha: t.sourceAsha, nidcd: t.sourceNidcd, hoarseness: t.sourceHoarseness, ucsf: t.sourceUcsf },
        },
        voiceAge: { title: t.ageTitle },
        voiceAgeV2: {
          eyebrow: t.ageEyebrow, title: t.ageEvidence, used: t.ageUsed,
          connectedReference: t.ageReference, note: t.ageNote,
        },
        insight: { label: t.insightLabel, ...t.insights },
      },
      quick: {
        footer: t.quickFooter,
        challenge: { behind: t.quickBehind },
        trust: { strict: t.quickTrust },
        reveal: { insight: t.quickInsight },
        refine: {
          eyebrow: t.quickRefineEyebrow, title: t.quickRefineTitle,
          retryAria: t.quickRetryAria, retryHint: t.quickRetryHint,
        },
        safety: { title: t.quickSafetyTitle, body: t.quickSafetyBody, stop: t.quickSafetyStop },
      },
    },
    realtime: {
      formantTitle: ui.formantTitle,
      formantNote: t.formantHint,
      formants: {
        f1Range: ui.appReference,
        f2Range: ui.appReference,
        f3Range: ui.appReference,
        breathRange: ui.spectralNoTarget,
      },
      resonance: { label: ui.resonanceTitle, chest: ui.lowBand, mask: ui.midBand, head: ui.highBand },
    },
    analysis: {
      safety: { detailsTitle: t.safetyTitle, detailsBody: t.safetyScope, focus: t.focusSafety },
      meter: { hint: t.modelHint },
      resonanceBalance: metricStates(proxyStates, t.proxyHint, ui.resonanceLabels),
      tilt: metricStates(tiltStates, t.proxyHint),
      breathiness: metricStates(breathStates, t.breathHint),
      brightness: metricStates(brightStates, t.proxyHint),
      formant: {
        trendLabels: { inRange: locale === "en" ? "Within app reference" : locale === "ja" ? "アプリ参考帯内" : locale === "zh-Hans" ? "在应用参考带内" : "在應用參考帶內" },
        moreSamplesHint: t.formantHint,
        inRange: `{{label}}: ${t.formantHint}`,
        lowMessage: `{{label}}: ${t.formantHint}`,
        highMessage: `{{label}}: ${t.formantHint}`,
        low: { F1: t.formantHint, F2: t.formantHint, F3: t.formantHint },
        high: { F1: t.formantHint, F2: t.formantHint, F3: t.formantHint },
      },
      vowelFocus: metricStates(["insufficient", "strong", "medium", "weak"], t.formantHint, {
        strong: locale === "en" ? "More frames in app reference" : locale === "ja" ? "参考帯のフレームが多め" : locale === "zh-Hans" ? "较多帧位于应用参考带" : "較多幀位於應用參考帶",
        medium: locale === "en" ? "Mixed relative pattern" : locale === "ja" ? "混合した相対パターン" : locale === "zh-Hans" ? "混合的相对分布" : "混合的相對分布",
        weak: locale === "en" ? "Fewer frames in app reference" : locale === "ja" ? "参考帯のフレームが少なめ" : locale === "zh-Hans" ? "较少帧位于应用参考带" : "較少幀位於應用參考帶",
      }),
      speechRate: metricStates(["insufficient", "tooSlow", "balanced", "fast"], t.paceHint),
      liaison: metricStates(["insufficient", "strong", "medium", "weak"], t.paceHint),
      intonation: {
        slope: metricStates(["rising", "flat", "falling"], t.intonationHint),
        range: metricStates(["rich", "medium", "narrow"], t.intonationHint),
      },
    },
    summary: {
      beginnerHighlights: {
        heading: t.focusHeading,
        empty: t.focusEmpty,
        items: {
          pitch: { title: ui.pitchComparison, tip: t.modelHint },
          resonance: { title: ui.spectralComparison, tip: t.proxyHint },
          speech: { title: ui.paceComparison, tip: t.paceHint },
        },
      },
      focus: {
        heading: t.focusHeading, empty: t.focusEmpty,
        safety: t.focusSafety,
        severity: { high: t.severity[0], medium: t.severity[1], low: t.severity[2] },
        cta: { pitch: ui.comparePitch, resonance: ui.compareSpectral, breath: ui.compareBreath, pace: ui.comparePace, clarity: ui.openGuide },
        items: {
          divergence: t.modelHint, noisy: t.proxyHint, pitchWide: t.modelHint,
          pitchModerate: t.modelHint, breathinessAiry: t.breathHint,
          breathinessDense: t.breathHint, vowelWeak: t.formantHint,
          speechFast: t.paceHint, speechSlow: t.paceHint,
          voicedLow: t.proxyHint, brightnessSharp: t.proxyHint,
        },
      },
      advanced: { formantTitle: ui.formantTitle, intonationTitle: ui.intonationTitle, vowelBreathTitle: ui.vowelTitle, referenceBand: ui.appReference },
    },
  };
  return deepMerge(safetyOverride, buildAnalysisGuidance(locale, ui));
}

export function applySafetyCopy(locale, dictionary) {
  return deepMerge(dictionary, buildOverride(locale));
}

export { SOURCES as MEDICAL_SAFETY_SOURCES };
