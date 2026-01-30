/**
 * Voice Manual Data (Multi-language)
 * Exporting a dictionary keyed by locale.
 */

export const MANUAL_DATA = {
  "zh-Hant": {
    title: "女聲訓練手冊",
    html: `
<div class="manual-intro">
  <blockquote>
    <p><strong>寫給誰：</strong> 寫給任何想把聲音更貼近「自己理想中的樣子」的人——包含跨女、非二元、偽聲練習者、配音/表演者，或只是想嘗試更女性化聲線的你。</p>
    <p><strong>先說清楚：</strong> 你的性別從來不需要靠聲音來「證明」。這份手冊只是在教一套 <strong>可練、可檢查、可穩定</strong> 的「聲音呈現」方法。</p>
    <p><strong>安全第一：</strong> 任何 <strong>疼痛、刺痛、卡住、練完沙啞</strong> → 立刻停止，休息 24–48 小時；若反覆發生，請找嗓音醫師/語言治療師。</p>
  </blockquote>
</div>

<hr />

<section>
  <h2>0) 你每天照做什麼（超快版）</h2>
  <p>你只要固定做完「<strong>25 分鐘標準課表</strong>」，並遵守兩個規則：</p>
  <ol>
    <li><strong>不擠、不硬卡喉位、不追求大音量</strong>（越大聲越容易回到舊習慣）</li>
    <li><strong>先把女聲穩定起來，再進甜美段</strong>（甜美是加成，不是起點）</li>
  </ol>
  <p>你就會穩定把聲音往「更像女生、更自然、更不費力」推進。</p>
</section>

<section>
  <h2>1) 女聲的四個旋鈕（最重要的觀念）</h2>
  <p>把女聲想像成「調音台的四個旋鈕」：要一起調，才會像。</p>
  <ol>
    <li><strong>音高 Pitch（高度）</strong>：不必很高；要「落在女生常用區間，能講話、不費力」。</li>
    <li><strong>共鳴 Resonance（聲音放哪裡）｜最關鍵</strong>：把聲音從「胸口大箱子」搬到「嘴前小箱子」。</li>
    <li><strong>聲線重量 Vocal Weight（厚薄感）</strong>：變輕、變薄，但要有核心（不能虛飄）。</li>
    <li><strong>句尾與語調 Prosody（收尾方式）</strong>：女生常見「收住」或「微上揚」，比較少「斷崖式直墜」。</li>
  </ol>
  <blockquote>
    <p>一句話：<strong>像女生 ≈ 共鳴位置 + 聲線重量 + 句尾習慣</strong>；音高只是配角。</p>
  </blockquote>
</section>

<section>
  <h2>2) 安全規則（把它當「發聲交通規則」）</h2>
  <ul>
    <li><strong>疼痛/刺痛/卡住/練完沙啞</strong>：立刻停。只做「輕哼」與「放鬆嘆氣」。</li>
    <li><strong>音量 3–5/10</strong>：先求穩、求自然，不求大聲。</li>
    <li><strong>吞嚥硬卡</strong>只用來「感知位置」，不拿來「說話訓練」。</li>
    <li>口乾時最容易走偏：<strong>每 3–5 分鐘喝一口水</strong>。</li>
  </ul>
</section>

<section>
  <h2>3) 量測與追蹤（公開版友善做法）</h2>
  <h3>3.1 善用 VPA 內建追蹤</h3>
  <ul>
    <li><strong>使用內建錄音</strong>：點擊畫面中央的 🎙️ 按鈕，系統會自動處理雜訊並保持取樣一致。</li>
    <li><strong>查看即時回饋</strong>：錄音時，畫面上的波形與數值（如 Pitch）會即時跳動，讓你更有感。</li>
  </ul>

  <h3>3.2 固定測試文本（30 秒）</h3>
  <p>用你想要的「日常自然、甜而不做作」口吻念：</p>
  <div class="manual-script">
    <p>「嗨～早安。今天我想把事情慢慢做好，不用急。<br>
    我等等要出門一下，回來再跟你說。辛苦了，謝謝你喔。」</p>
    <div class="manual-practice-widget" data-phrase='{"id":"manual_test_script_tw"}'></div>
  </div>

  <h3>3.3 每週一次對比（超有效）</h3>
  <ul>
    <li>週一錄一次（當週基準）</li>
    <li>週五再錄一次（看進步與破功點）</li>
    <li>只要回聽覺得 <strong>更亮、更靠前、更不費力</strong>，方向就是對的。</li>
  </ul>

  <h3>3.4 如何看懂 VPA 分析報告？</h3>
  <p>錄音結束後，VPA 會生成詳細的儀表板，請重點關注這三個指針：</p>
  <ul>
    <li><strong>音高 (Pitch)</strong>：觀察是否落在「女生區間」內。如果不夠高，請嘗試爬階梯練習。</li>
    <li><strong>共鳴 (Resonance)</strong>：最核心指標。
      <ul>
        <li>偏左 (Chest/Mask)：像男生或感冒聲音，需要更多「嗯—」練習。</li>
        <li><strong>偏右 (Head/Bright)</strong>：這才是我們要的目標區域！</li>
      </ul>
    </li>
    <li><strong>重量 (Weight)</strong>：
      <ul>
        <li><strong>Heavy (厚)</strong>：聲音太重，像新聞男主播。</li>
        <li><strong>Light (輕)</strong>：正確方向，目標是讓指針往「Light」偏移。</li>
      </ul>
    </li>
  </ul>
  <blockquote>
    <p>小撇步：別只看總分，要看這三個指針有沒有「變綠色」或「往右邊跑」。</p>
  </blockquote>
</section>

<section>
  <h2>4) 四階段路線圖（新手 → 穩定女聲 → 甜美）</h2>
  <div class="manual-timeline">
    <h3>Stage 0：找位置（3–7 天）</h3>
    <p><strong>目標：</strong> 不傷喉、能把共鳴放到臉前、能說 1–2 句不掉回去。</p>

    <h3>Stage 1：女聲輪廓（2–4 週）</h3>
    <p><strong>目標：</strong> 能穩定短句、聲線變輕、句尾開始「收住」而不是直墜。</p>

    <h3>Stage 2：穩定口語（4–8 週）</h3>
    <p><strong>目標：</strong> 能聊天 3–5 分鐘不累、不緊、不崩回去。</p>

    <h3>Stage 3：甜美加成（達門檻後再進）</h3>
    <p><strong>目標：</strong> 在穩定女聲上增加「乾淨、亮而不刺、禮貌尾巴」的甜美感。</p>
    <p class="note"><strong>甜美門檻（建議）</strong>：你已經能用女聲連續說 1 分鐘、喉嚨完全不緊，回聽也不覺得像硬擠。<br>
    不到門檻先別做甜美，否則容易走成「嗲、虛、尖、假」。</p>
  </div>
</section>

<section>
  <h2>5) 六個核心模組（每天都會用到）</h2>

  <h3>模組 A：放鬆（30–90 秒）</h3>
  <p><strong>嘆氣哈— ×5</strong>（像放下肩膀的嘆息）</p>
  <ul>
    <li>你應該感覺：肩頸放下、喉嚨像「更開一點」</li>
    <li>如果變成用力吐氣：把音量降到更小，像「熱氣呼出」</li>
  </ul>

  <h3>模組 B：前置共鳴（2–4 分鐘）</h3>
  <ol>
    <li><strong>閉嘴哼「嗯—」</strong>：找「上唇、鼻翼、門牙後方」的震動感
      <div class="manual-practice-widget" data-phrase='{"id":"manual_hum_tw","text":"嗯——","tip":"閉嘴哼，找鼻樑震動"}'></div>
    </li>
    <li><strong>ㄋ 起音短句</strong>：每句前輕帶「ㄋ…」
      <ul>
        <li>例：ㄋ…你好、ㄋ…我知道、ㄋ…謝謝你</li>
      </ul>
    </li>
    <li><strong>微笑 30%</strong>：嘴角輕提（不是咧嘴），讓聲音自然更靠前</li>
  </ol>
  <p class="tip">小提醒：前置共鳴不是「變鼻音」。你要的是「前面亮」，不是「鼻腔塞滿」。</p>

  <h3>模組 C：爬階梯法（5–8 分鐘｜最推薦的主訓練）</h3>
  <p><strong>原則：</strong> 讓喉位「跟著音高自然上移」，不要硬把它按住。</p>
  <p>做法：</p>
  <ol>
    <li>先「嗯—」找到臉前震動</li>
    <li>做滑音（像警報器）：「嗯～～」從低慢慢滑到高（不必滑到破音）</li>
    <li>到高點時 <strong>切掉聲音</strong>，但保留那個「小箱子、靠前」的感覺 1–2 秒</li>
    <li>在那個感覺裡說一句超短句（1–3 秒）
      <div class="manual-practice-widget" data-phrase='{"id":"manual_stair_hello_tw","text":"「你好」「真的嗎」「辛苦了」","tip":"保留高點位置，輕輕說"}'></div>
      <ul>
        <li>例：「你好」「真的嗎」「辛苦了」</li>
      </ul>
    </li>
    <li>立刻回到「嗯—」放鬆，再來一次</li>
  </ol>
  <p>常見錯誤與修正：</p>
  <ul>
    <li><strong>滑太高破音、喉嚨緊</strong> → 只滑到「亮起來」就好，寧可不高也不要緊</li>
    <li><strong>以為要固定在最高</strong> → 你要的是「自然上移的工作點」，不是「卡死」</li>
  </ul>

  <h3>模組 D：聲線重量變輕（2–4 分鐘）</h3>
  <p>目標：<strong>輕，但不虛飄</strong>（薄但有芯）。</p>
  <ul>
    <li>用 4/10 音量說：
      <ul>
        <li>「我知道了。」×10
          <div class="manual-practice-widget" data-phrase='{"id":"manual_weight_iknow_tw","text":"我知道了","tip":"4/10 音量，輕但有芯"}'></div>
        </li>
        <li>「沒關係。」×10
          <div class="manual-practice-widget" data-phrase='{"id":"manual_weight_okay_tw","text":"沒關係","tip":"4/10 音量，輕但有芯"}'></div>
        </li>
      </ul>
    </li>
    <li>如果變得很虛：音量加一格（仍小聲），讓聲音站起來</li>
  </ul>

  <h3>模組 E：清楚亮度（2–3 分鐘）</h3>
  <p>目的：增加子音清楚與前置亮度，不是捏尖嗓子。</p>
  <ul>
    <li>「西、希、細、喜」每字 10 次（小聲、清楚、不用力）
      <div class="manual-practice-widget" data-phrase='{"id":"manual_clarity_xi_tw","text":"西、希、細、喜","tip":"小聲、清楚、不用力"}'></div>
    </li>
    <li>「七、氣、洗、喜」每字 10 次
      <div class="manual-practice-widget" data-phrase='{"id":"manual_clarity_qi_tw","text":"七、氣、洗、喜","tip":"增加子音清楚與前置亮度"}'></div>
    </li>
  </ul>

  <h3>模組 F：句尾不直墜（3–5 分鐘）</h3>
  <p>先練「收住」，再練「微上揚」。</p>
  <p><strong>1) 收住版（最自然、最不做作）</strong></p>
  <ul>
    <li>「我知道。」
      <div class="manual-practice-widget" data-phrase='{"id":"manual_prosody_iknow_tw","text":"我知道。","tip":"句尾收住"}'></div>
    </li>
    <li>「可以。」
      <div class="manual-practice-widget" data-phrase='{"id":"manual_prosody_can_tw","text":"可以。","tip":"句尾收住"}'></div>
    </li>
    <li>「沒關係。」
      <div class="manual-practice-widget" data-phrase='{"id":"manual_prosody_okay_tw","text":"沒關係。","tip":"句尾收住"}'></div>
    </li>
  </ul>
  <p>每句 10 次，句尾最後 <strong>0.2 秒</strong>像「點頭收尾」，不要摔下去。</p>
  <p><strong>2) 微上揚版（很小很小）</strong></p>
  <ul>
    <li>「真的嗎～」
      <div class="manual-practice-widget" data-phrase='{"id":"manual_prosody_really_tw","text":"真的嗎～","tip":"微上揚"}'></div>
    </li>
    <li>「好呀～」
      <div class="manual-practice-widget" data-phrase='{"id":"manual_prosody_good_tw","text":"好呀～","tip":"微上揚"}'></div>
    </li>
  </ul>
  <p>上揚要小到「不像唱歌」。</p>
</section>

<section>
  <h2>6) 每日課表（照表做就好）</h2>
  
  <div class="schedule-card">
    <h3>6.1 12 分鐘保底版（忙也要做）</h3>
    <ol>
      <li>放鬆哈—×5（1’）</li>
      <li>嗯—找臉前震動（2’）</li>
      <li>爬階梯滑音 6 次（4’）</li>
      <li>每次滑音後短句 1 句（4’）</li>
      <li><strong>使用 VPA 錄音 30 秒</strong>（1’）：確保共鳴指標有進步。</li>
    </ol>
  </div>

  <div class="schedule-card">
    <h3>6.2 25 分鐘標準版（最推薦）</h3>
    <ol>
      <li>放鬆（2’）</li>
      <li>前置共鳴（4’）：嗯— + ㄋ起音短句</li>
      <li>爬階梯（8’）：滑音 8 次；每次定格後短句 1 句</li>
      <li>聲線重量（3’）：兩句各 10 次</li>
      <li>咬字亮度（3’）</li>
      <li>句尾（4’）：收住為主</li>
      <li><strong>使用 VPA 錄音 30 秒</strong>（1’）：檢查 Pitch 與 Resonance 是否穩定。</li>
    </ol>
  </div>

  <div class="schedule-card">
    <h3>6.3 45 分鐘加速版（仍要「不累」）</h3>
    <p>在 25 分鐘後加：</p>
    <ul>
      <li>朗讀 2 分鐘（慢、清楚、輕）</li>
      <li>自問自答 3 分鐘（用女聲回答自己）</li>
      <li>句尾訓練加 5 分鐘</li>
      <li>若已達甜美門檻，再加甜美段 10 分鐘（見下一章）</li>
    </ul>
  </div>
</section>

<section>
  <h2>7) 甜美版（達門檻才做）</h2>
  <p>甜美不是「嗲」，是三件事：</p>
  <ol>
    <li><strong>更乾淨</strong>（少粗糙、少擠壓）</li>
    <li><strong>更亮但不刺</strong>（更靠前、更清楚）</li>
    <li><strong>更有禮貌的尾巴</strong>（收住或微上揚）</li>
  </ol>

  <h3>甜美段（10 分鐘）</h3>
  <ol>
    <li>微笑 30% + 嗯—（1’）</li>
    <li>甜美短句（4’）：每句 8 次
      <ul>
        <li>「謝謝你喔～」
          <div class="manual-practice-widget" data-phrase='{"id":"manual_sweet_thanks_tw","text":"謝謝你喔～","tip":"更乾淨、更亮"}'></div>
        </li>
        <li>「辛苦了耶。」
          <div class="manual-practice-widget" data-phrase='{"id":"manual_sweet_hardwork_tw","text":"辛苦了耶。","tip":"更乾淨、更亮"}'></div>
        </li>
        <li>「可以麻煩你嗎～」
          <div class="manual-practice-widget" data-phrase='{"id":"manual_sweet_trouble_tw","text":"可以麻煩你嗎～","tip":"更乾淨、更亮"}'></div>
        </li>
        <li>「真的嗎？好開心～」
          <div class="manual-practice-widget" data-phrase='{"id":"manual_sweet_happy_tw","text":"真的嗎？好開心～","tip":"更乾淨、更亮"}'></div>
        </li>
      </ul>
    </li>
    <li>甜美句尾（3’）：尾巴變輕 + 變亮 + 微收住</li>
    <li>甜美固定文本（2’）
      <div class="manual-script">
        <p>「嗨～今天謝謝你陪我一下。<br>
        我等一下就回來，路上小心喔～」</p>
        <div class="manual-practice-widget" data-phrase='{"id":"manual_sweet_script_tw"}'></div>
      </div>
    </li>
  </ol>

  <h3>甜美常見走偏與修正</h3>
  <ul>
    <li><strong>氣音太多</strong> → 變虛、變霧、很累：音量降一格、氣音減半</li>
    <li><strong>上揚太多</strong> → 變做作：上揚幅度砍半，回到收住版</li>
    <li><strong>捏尖</strong> → 變刺、卡通感：回到「嗯—前置」再說</li>
  </ul>
</section>

<section>
  <h2>8) 7 天打勾清單（最快進步期）</h2>
  <ul class="checklist">
    <li><strong>Day1：</strong> 臉前震動 + 6 次爬階梯 + 30 秒錄音</li>
    <li><strong>Day2：</strong> 同 Day1 + ㄋ起音短句 30 次</li>
    <li><strong>Day3：</strong> 加句尾收住 30 次</li>
    <li><strong>Day4：</strong> 加「西希細喜」</li>
    <li><strong>Day5：</strong> 加 3 秒定音「啊—」×10（小聲）</li>
    <li><strong>Day6：</strong> 自問自答 2 分鐘</li>
    <li><strong>Day7：</strong> 回聽 Day1 vs Day7，挑出最像女生的 10 秒，寫下「那個感覺是什麼」</li>
  </ul>
</section>

<section>
  <h2>9) 卡關排錯（超實用）</h2>
  <ul>
    <li><strong>一拉高就緊</strong>：你在推音高，不是在換位置 → 回到「嗯—」+「ㄋ起音」</li>
    <li><strong>音很高但仍偏厚</strong>：共鳴還在胸口、重量太重 → 加強 B + D</li>
    <li><strong>很虛很氣</strong>：漏氣太多 → 音量加一格、讓聲音有核心</li>
    <li><strong>鼻音太重</strong>：前置過頭 → 微笑減半、別用力「頂鼻」</li>
    <li><strong>句尾一直掉</strong>：先練「收住」，上揚是第二步</li>
  </ul>
</section>

<section>
  <h2>10) 你每天的成功標準（一句話）</h2>
  <blockquote>
    <p><strong>練完喉嚨更舒服、聲音更亮更靠前、回聽更像女生</strong><br>
    就算今天音高沒更高，你也在正確變強。</p>
  </blockquote>
  <p>（完）</p>
</section>
    `
  },
  "zh-Hans": {
    title: "女声训练手册",
    html: `
<div class="manual-intro">
  <blockquote>
    <p><strong>写给谁：</strong> 写给任何想把声音更贴近“自己理想中的样子”的人——包含跨女、非二元、伪声练习者、配音/表演者，或只是想尝试更女性化声线的你。</p>
    <p><strong>先说清楚：</strong> 你的性别从来不需要靠声音来“证明”。这份手册只是在教一套 <strong>可练、可检查、可稳定</strong> 的“声音呈现”方法。</p>
    <p><strong>安全第一：</strong> 任何 <strong>疼痛、刺痛、卡住、练完沙哑</strong> → 立刻停止，休息 24–48 小时；若反复发生，请找嗓音医师/语言治疗师。</p>
  </blockquote>
</div>

<hr />

<section>
  <h2>0) 你每天照做什么（超快版）</h2>
  <p>你只要固定做完“<strong>25 分钟标准课表</strong>”，并遵守两个规则：</p>
  <ol>
    <li><strong>不挤、不硬卡喉位、不追求大音量</strong>（越大声越容易回到旧习惯）</li>
    <li><strong>先把女声稳定起来，再进甜美段</strong>（甜美是加成，不是起点）</li>
  </ol>
  <p>你就会稳定把声音往“更像女生、更自然、更不费力”推进。</p>
</section>

<section>
  <h2>1) 女声的四个旋钮（最重要的观念）</h2>
  <p>把女声想象成“调音台的四个旋钮”：要一起调，才会像。</p>
  <ol>
    <li><strong>音高 Pitch（高度）</strong>：不必很高；要“落在女生常用区间，能讲话、不费力”。</li>
    <li><strong>共鸣 Resonance（声音放哪里）｜最关键</strong>：把声音从“胸口大箱子”搬到“嘴前小箱子”。</li>
    <li><strong>声线重量 Vocal Weight（厚薄感）</strong>：变轻、变薄，但要有核心（不能虚飘）。</li>
    <li><strong>句尾与语调 Prosody（收尾方式）</strong>：女生常见“收住”或“微上扬”，比较少“断崖式直坠”。</li>
  </ol>
  <blockquote>
    <p>一句话：<strong>像女生 ≈ 共鸣位置 + 声线重量 + 句尾习惯</strong>；音高只是配角。</p>
  </blockquote>
</section>

<section>
  <h2>2) 安全规则（把它当“发声交通规则”）</h2>
  <ul>
    <li><strong>疼痛/刺痛/卡住/练完沙哑</strong>：立刻停。只做“轻哼”与“放松叹气”。</li>
    <li><strong>音量 3–5/10</strong>：先求稳、求自然，不求大声。</li>
    <li><strong>吞咽硬卡</strong>只用来“感知位置”，不拿来“说话训练”。</li>
    <li>口干时最容易走偏：<strong>每 3–5 分钟喝一口水</strong>。</li>
  </ul>
</section>

<section>
  <h2>3) 量測與追蹤（公开版友善做法）</h2>
  <h3>3.1 善用 VPA 内建追踪</h3>
  <ul>
    <li><strong>使用内建录音</strong>：点击画面中央的 🎙️ 按钮，系统会自动处理杂讯并保持采样一致。</li>
    <li><strong>查看即时回馈</strong>：录音时，画面上的波形与数值（如 Pitch）会即时跳动，让你更有感。</li>
  </ul>

  <h3>3.2 固定测试文本（30 秒）</h3>
  <p>用你想要的“日常自然、甜而不做作”口吻念：</p>
  <div class="manual-script">
    <p>“嗨～早安。今天我想把事情慢慢做好，不用急。<br>
    我等等要出门一下，回来再跟你说。辛苦了，谢谢你喔。”</p>
    <div class="manual-practice-widget" data-phrase='{"id":"manual_test_script_cn","text":"嗨～早安...","tip":"30秒测试文本"}'></div>
  </div>

  <h3>3.3 每周一次对比（超有效）</h3>
  <ul>
    <li>周一录一次（当周基准）</li>
    <li>周五再录一次（看进步与破功点）</li>
    <li>只要回听觉得 <strong>更亮、更靠前、更不费力</strong>，方向就是对的。</li>
  </ul>

  <h3>3.4 如何看懂 VPA 分析报告？</h3>
  <p>录音结束后，VPA 会生成详细的仪表板，请重点关注这三个指标：</p>
  <ul>
    <li><strong>音高 (Pitch)</strong>：观察是否落在“女生区间”内。如果不够高，请尝试爬阶梯练习。</li>
    <li><strong>共鸣 (Resonance)</strong>：最核心指标。
      <ul>
        <li>偏左 (Chest/Mask)：像男生或感冒声音，需要更多“嗯—”练习。</li>
        <li><strong>偏右 (Head/Bright)</strong>：这才是我们要的目标区域！</li>
      </ul>
    </li>
    <li><strong>重量 (Weight)</strong>：
      <ul>
        <li><strong>Heavy (厚)</strong>：声音太重，像新闻男主播。</li>
        <li><strong>Light (轻)</strong>：正确方向，目标是让指针往“Light”偏移。</li>
      </ul>
    </li>
  </ul>
  <blockquote>
    <p>小撇步：别只看总分，要看这三个指标有没有“变绿色”或“往右边跑”。</p>
  </blockquote>
</section>

<section>
  <h2>4) 四阶段路线图（新手 → 稳定女声 → 甜美）</h2>
  <div class="manual-timeline">
    <h3>Stage 0：找位置（3–7 天）</h3>
    <p><strong>目标：</strong> 不伤喉、能把共鸣放到脸前、能说 1–2 句不掉回去。</p>

    <h3>Stage 1：女声轮廓（2–4 周）</h3>
    <p><strong>目标：</strong> 能稳定短句、声线变轻、句尾开始“收住”而不是直坠。</p>

    <h3>Stage 2：稳定口语（4–8 周）</h3>
    <p><strong>目标：</strong> 能聊天 3–5 分钟不累、不紧、不崩回去。</p>

    <h3>Stage 3：甜美加成（达门槛后再进）</h3>
    <p><strong>目标：</strong> 在稳定女声上增加“干净、亮而不刺、礼貌尾巴”的甜美感。</p>
    <p class="note"><strong>甜美门槛（建议）</strong>：你已经能用女声连续说 1 分钟、喉咙完全不紧，回听也不觉得像硬挤。<br>
    不到门槛先别做甜美，否则容易走成“嗲、虚、尖、假”。</p>
  </div>
</section>

<section>
  <h2>5) 六个核心模组（每天都会用到）</h2>

  <h3>模组 A：放松（30–90 秒）</h3>
  <p><strong>叹气哈— ×5</strong>（像放下肩膀的叹息）</p>
  <ul>
    <li>你应该感觉：肩颈放下、喉咙像“更开一点”</li>
    <li>如果变成用力吐气：把音量降到更小，像“热气呼出”</li>
  </ul>

  <h3>模组 B：前置共鸣（2–4 分钟）</h3>
  <ol>
    <li><strong>闭嘴哼“嗯—”</strong>：找“上唇、鼻翼、门牙后方”的震动感
      <div class="manual-practice-widget" data-phrase='{"id":"manual_hum_cn","text":"嗯——","tip":"闭嘴哼，找鼻梁震动"}'></div>
    </li>
    <li><strong>ㄋ 起音短句</strong>：每句前轻带“ㄋ…”
      <ul>
        <li>例：ㄋ…你好、ㄋ…我知道、ㄋ…谢谢你</li>
      </ul>
    </li>
    <li><strong>微笑 30%</strong>：嘴角轻提（不是咧嘴），让声音自然更靠前</li>
  </ol>
  <p class="tip">小提醒：前置共鸣不是“变鼻音”。你要的是“前面亮”，不是“鼻腔塞满”。</p>

  <h3>模组 C：爬阶梯法（5–8 分钟｜最推荐的主训练）</h3>
  <p><strong>原则：</strong> 让喉位“跟着音高自然上移”，不要硬把它按住。</p>
  <p>做法：</p>
  <ol>
    <li>先“嗯—”找到脸前震动</li>
    <li>做滑音（像警报器）：“嗯～～”从低慢慢滑到高（不必滑到破音）</li>
    <li>到高点时 <strong>切掉声音</strong>，但保留那个“小箱子、靠前”的感觉 1–2 秒</li>
    <li>在那个感觉里说一句超短句（1–3 秒）
      <div class="manual-practice-widget" data-phrase='{"id":"manual_stair_hello_cn","text":"你好","tip":"保留高点位置，轻轻说"}'></div>
      <ul>
        <li>例：“你好”“真的吗”“辛苦了”</li>
      </ul>
    </li>
    <li>立刻回到“嗯—”放松，再来一次</li>
  </ol>
  <p>常见错误与修正：</p>
  <ul>
    <li><strong>滑太高破音、喉咙紧</strong> → 只滑到“亮起来”就好，宁可不高也不要紧</li>
    <li><strong>以为要固定在最高</strong> → 你要的是“自然上移的工作点”，不是“卡死”</li>
  </ul>

  <h3>模组 D：声线重量变轻（2–4 分钟）</h3>
  <p>目标：<strong>轻，但不虚飘</strong>（薄但有芯）。</p>
  <ul>
    <li>用 4/10 音量说：
      <ul>
        <li>“我知道了。”×10
          <div class="manual-practice-widget" data-phrase='{"id":"manual_weight_iknow_cn","text":"我知道了","tip":"4/10 音量，轻但有芯"}'></div>
        </li>
        <li>“没关系。”×10
          <div class="manual-practice-widget" data-phrase='{"id":"manual_weight_okay_cn","text":"没关系","tip":"4/10 音量，轻但有芯"}'></div>
        </li>
      </ul>
    </li>
    <li>如果变得很虚：音量加一格（仍小声），让声音站起来</li>
  </ul>

  <h3>模组 E：清楚亮度（2–3 分钟）</h3>
  <p>目的：增加子音清楚与前置亮度，不是捏尖嗓子。</p>
  <ul>
    <li>“西、希、细、喜”每字 10 次（小声、清楚、不用力）
      <div class="manual-practice-widget" data-phrase='{"id":"manual_clarity_xi_cn","text":"西、希、细、喜","tip":"小声、清楚、不用力"}'></div>
    </li>
    <li>“七、气、洗、喜”每字 10 次
      <div class="manual-practice-widget" data-phrase='{"id":"manual_clarity_qi_cn","text":"七、气、洗、喜","tip":"增加子音清楚与前置亮度"}'></div>
    </li>
  </ul>

  <h3>模组 F：句尾不直坠（3–5 分钟）</h3>
  <p>先练“收住”，再练“微上扬”。</p>
  <p><strong>1) 收住版（最自然、最不做作）</strong></p>
  <ul>
    <li>“我知道。”
      <div class="manual-practice-widget" data-phrase='{"id":"manual_prosody_iknow_cn","text":"我知道。","tip":"句尾收住"}'></div>
    </li>
    <li>“可以。”
      <div class="manual-practice-widget" data-phrase='{"id":"manual_prosody_can_cn","text":"可以。","tip":"句尾收住"}'></div>
    </li>
    <li>“没关系。”
      <div class="manual-practice-widget" data-phrase='{"id":"manual_prosody_okay_cn","text":"没关系。","tip":"句尾收住"}'></div>
    </li>
  </ul>
  <p>每句 10 次，句尾最后 <strong>0.2 秒</strong>像“点头收尾”，不要摔下去。</p>
  <p><strong>2) 微上扬版（很小很小）</strong></p>
  <ul>
    <li>“真的吗～”
      <div class="manual-practice-widget" data-phrase='{"id":"manual_prosody_really_cn","text":"真的吗～","tip":"微上扬"}'></div>
    </li>
    <li>“好呀～”
      <div class="manual-practice-widget" data-phrase='{"id":"manual_prosody_good_cn","text":"好呀～","tip":"微上扬"}'></div>
    </li>
  </ul>
  <p>上扬要小到“不像唱歌”。</p>
</section>

<section>
  <h2>6) 每日课表（照表做就好）</h2>
  
  <div class="schedule-card">
    <h3>6.1 12 分钟保底版（忙也要做）</h3>
    <ol>
      <li>放松哈—×5（1’）</li>
      <li>嗯—找脸前震动（2’）</li>
      <li>爬阶梯滑音 6 次（4’）</li>
      <li>每次滑音后短句 1 句（4’）</li>
      <li><strong>使用 VPA 录音 30 秒</strong>（1’）：确保共鸣指标有进步。</li>
    </ol>
  </div>

  <div class="schedule-card">
    <h3>6.2 25 分钟标准版（最推荐）</h3>
    <ol>
      <li>放松（2’）</li>
      <li>前置共鸣（4’）：嗯— + ㄋ起音短句</li>
      <li>爬阶梯（8’）：滑音 8 次；每次定格后短句 1 句</li>
      <li>声线重量（3’）：两句各 10 次</li>
      <li>咬字亮度（3’）</li>
      <li>句尾（4’）：收住为主</li>
      <li><strong>使用 VPA 录音 30 秒</strong>（1’）：检查 Pitch 与 Resonance 是否稳定。</li>
    </ol>
  </div>

  <div class="schedule-card">
    <h3>6.3 45 分钟加速版（仍要“不累”）</h3>
    <p>在 25 分钟后加：</p>
    <ul>
      <li>朗读 2 分钟（慢、清楚、轻）</li>
      <li>自问自答 3 分钟（用女声回答自己）</li>
      <li>句尾训练加 5 分钟</li>
      <li>若已达甜美门槛，再加甜美段 10 分钟（见下一章）</li>
    </ul>
  </div>
</section>

<section>
  <h2>7) 甜美版（达门槛才做）</h2>
  <p>甜美不是“嗲”，是三件事：</p>
  <ol>
    <li><strong>更干净</strong>（少粗糙、少挤压）</li>
    <li><strong>更亮但不刺</strong>（更靠前、更清楚）</li>
    <li><strong>更有礼貌的尾巴</strong>（收住或微上扬）</li>
  </ol>

  <h3>甜美段（10 分钟）</h3>
  <ol>
    <li>微笑 30% + 嗯—（1’）</li>
    <li>甜美短句（4’）：每句 8 次
      <ul>
        <li>“谢谢你喔～”
          <div class="manual-practice-widget" data-phrase='{"id":"manual_sweet_thanks_cn","text":"谢谢你喔～","tip":"更干净、更亮"}'></div>
        </li>
        <li>“辛苦了耶。”
          <div class="manual-practice-widget" data-phrase='{"id":"manual_sweet_hardwork_cn","text":"辛苦了耶。","tip":"更干净、更亮"}'></div>
        </li>
        <li>“可以麻烦你吗～”
          <div class="manual-practice-widget" data-phrase='{"id":"manual_sweet_trouble_cn","text":"可以麻烦你吗～","tip":"更干净、更亮"}'></div>
        </li>
        <li>“真的吗？好开心～”
          <div class="manual-practice-widget" data-phrase='{"id":"manual_sweet_happy_cn","text":"真的吗？好开心～","tip":"更干净、更亮"}'></div>
        </li>
      </ul>
    </li>
    <li>甜美句尾（3’）：尾巴变轻 + 变亮 + 微收住</li>
    <li>甜美固定文本（2’）
      <div class="manual-script">
        <p>“嗨～今天谢谢你陪我一下。<br>
        我等一下就回来，路上小心喔～”</p>
        <div class="manual-practice-widget" data-phrase='{"id":"manual_sweet_script_cn"}'></div>
      </div>
    </li>
  </ol>

  <h3>甜美常见走偏与修正</h3>
  <ul>
    <li><strong>气音太多</strong> → 变虚、变雾、很累：音量降一格、气音减半</li>
    <li><strong>上扬太多</strong> → 变做作：上扬幅度砍半，回到收住版</li>
    <li><strong>捏尖</strong> → 变刺、卡通感：回到“嗯—前置”再说</li>
  </ul>
</section>

<section>
  <h2>8) 7 天打勾清单（最快进步期）</h2>
  <ul class="checklist">
    <li><strong>Day1：</strong> 脸前震动 + 6 次爬阶梯 + 30 秒录音</li>
    <li><strong>Day2：</strong> 同 Day1 + ㄋ起音短句 30 次</li>
    <li><strong>Day3：</strong> 加句尾收住 30 次</li>
    <li><strong>Day4：</strong> 加“西希细喜”</li>
    <li><strong>Day5：</strong> 加 3 秒定音“啊—”×10（小声）</li>
    <li><strong>Day6：</strong> 自问自答 2 分钟</li>
    <li><strong>Day7：</strong> 回听 Day1 vs Day7，挑出最像女生的 10 秒，写下“那个感觉是什么”</li>
  </ul>
</section>

<section>
  <h2>9) 卡关排错（超实用）</h2>
  <ul>
    <li><strong>一拉高就紧</strong>：你在推音高，不是在换位置 → 回到“嗯—”+“ㄋ起音”</li>
    <li><strong>音很高但仍偏厚</strong>：共鸣还在胸口、重量太重 → 加强 B + D</li>
    <li><strong>很虚很气</strong>：漏气太多 → 音量加一格、让声音有核心</li>
    <li><strong>鼻音太重</strong>：前置过头 → 微笑减半、别用力“顶鼻”</li>
    <li><strong>句尾一直掉</strong>：先练“收住”，上扬是第二步</li>
  </ul>
</section>

<section>
  <h2>10) 你每天的成功标准（一句话）</h2>
  <blockquote>
    <p><strong>练完喉咙更舒服、声音更亮更靠前、回听更像女生</strong><br>
    就算今天音高没更高，你也在正确变强。</p>
  </blockquote>
  <p>（完）</p>
</section>
    `
  },
  "en": {
    title: "Feminine Voice Manual",
    html: `
<div class="manual-intro">
  <blockquote>
    <p><strong>Who is this for:</strong> Anyone who wants their voice to match their ideal self—including trans women, non-binary individuals, voice actors, or anyone exploring a more feminine register.</p>
    <p><strong>Disclaimer:</strong> Your gender does not need a voice to "prove" it. This manual simply teaches a <strong>trainable, measurable, and sustainable</strong> method for voice feminization.</p>
    <p><strong>Safety First:</strong> If you feel <strong>pain, sharp discomfort, squeezing, or hoarseness</strong> → STOP immediately. Rest for 24–48 hours. If issues persist, consult a speech therapist or ENT.</p>
  </blockquote>
</div>

<hr />

<section>
  <h2>0) Daily Essentials (TL;DR)</h2>
  <p>Stick to the "<strong>25-minute Standard Routine</strong>" and follow two rules:</p>
  <ol>
    <li><strong>Don't Squeeze or Strain</strong> (Louder often means falling back to old habits).</li>
    <li><strong>Stability First, Sweetness Later</strong> ("Sweetness" is an add-on, not the foundation).</li>
  </ol>
  <p>This will steadily push your voice towards being "more feminine, natural, and effortless."</p>
</section>

<section>
  <h2>1) The Four Knobs of Feminine Voice</h2>
  <p>Think of it as a mixing console with four knobs that must be adjusted together.</p>
  <ol>
    <li><strong>Pitch</strong>: Doesn't need to be super high. Aim for the "androgynous/female overlapping range" where you can speak comfortably.</li>
    <li><strong>Resonance (The Key)</strong>: Move the sound from the "big chest box" to the "small mouth box."</li>
    <li><strong>Vocal Weight</strong>: Lighter and thinner, but with a core (not breathy or hollow).</li>
    <li><strong>Prosody (Intonation)</strong>: Feminine speech often focuses on "holding" the end of sentences or a "slight upward inflection," avoiding the "cliff-drop" low pitch ending.</li>
  </ol>
  <blockquote>
    <p>In short: <strong>Feminine Voice ≈ Resonance + Weight + Prosody</strong>; Pitch is secondary.</p>
  </blockquote>
</section>

<section>
  <h2>2) Safety Rules (Traffic Laws for Voice)</h2>
  <ul>
    <li><strong>Pain/Hoarseness</strong>: Stop. Only do light humming or gentle sighing.</li>
    <li><strong>Volume 3–5/10</strong>: Aim for stability and naturalness, not loudness.</li>
    <li><strong>Swallowing Method</strong>: Only use to "sense the position," NEVER for speaking practice.</li>
    <li><strong>Hydration</strong>: Drink water every 3–5 minutes to prevent dryness and strain.</li>
  </ul>
</section>

<section>
  <h2>3) Measurement & Tracking</h2>
  <h3>3.1 Use VPA Built-in Tracking</h3>
  <ul>
    <li><strong>Record with VPA</strong>: Click the central 🎙️ button. The system handles noise reduction and consistency.</li>
    <li><strong>Real-time Feedback</strong>: Watch the live waveform and Pitch gauge while recording.</li>
  </ul>

  <h3>3.2 Standard Test Script (30s)</h3>
  <p>Read this with a "natural, friendly, and sweet" tone:</p>
  <div class="manual-script">
    <p>"Hi, good morning. I want to take things slow and easy today.<br>
    I'll head out for a bit and talk to you when I'm back. Thanks for everything!"</p>
    <div class="manual-practice-widget" data-phrase='{"id":"manual_test_script_en","text":"Hi, good morning...","tip":"30s Test Script"}'></div>
  </div>

  <h3>3.3 Weekly Comparison</h3>
  <ul>
    <li>Record on Monday (Baseline)</li>
    <li>Record on Friday (Check progress/fatigue)</li>
    <li>If it sounds <strong>Brighter, More Forward, and Less Effortful</strong>, you are on the right track.</li>
  </ul>

  <h3>3.4 Understanding VPA Analysis</h3>
  <p>After recording, check the dashboard gauges:</p>
  <ul>
    <li><strong>Pitch</strong>: Is it in the female range? If low, try the "Staircase Siren" exercise.</li>
    <li><strong>Resonance</strong>: The heavy lifter.
      <ul>
        <li>Left (Chest/Mask): Male-leaning/Cold. Needs more "Mmm" humming.</li>
        <li><strong>Right (Head/Bright)</strong>: This is the target zone!</li>
      </ul>
    </li>
    <li><strong>Weight</strong>:
      <ul>
        <li><strong>Heavy</strong>: Thick, booming voice.</li>
        <li><strong>Light</strong>: The goal. Ideally centered or leaning towards Light.</li>
      </ul>
    </li>
  </ul>
</section>

<section>
  <h2>4) Four-Stage Roadmap</h2>
  <div class="manual-timeline">
    <h3>Stage 0: Finding the Place (3–7 Days)</h3>
    <p><strong>Goal:</strong> No pain. Ability to move resonance to the front (face) and hold it for 1-2 sentences.</p>

    <h3>Stage 1: The Contour (2–4 Weeks)</h3>
    <p><strong>Goal:</strong> Stable short sentences. Lighter weight. Sentence endings cease to "drop."</p>

    <h3>Stage 2: Stability (4–8 Weeks)</h3>
    <p><strong>Goal:</strong> Chatting for 3-5 minutes without fatigue, tightness, or cracking.</p>

    <h3>Stage 3: Sweetness (Advanced)</h3>
    <p><strong>Goal:</strong> Adding "clean, bright, and polite" qualities on top of the stable base.</p>
    <p class="note"><strong>Threshold</strong>: Don't attempt Stage 3 until you can speak for 1 minute comfortably without strain.</p>
  </div>
</section>

<section>
  <h2>5) Six Core Modules</h2>

  <h3>Module A: Relaxation (30–90s)</h3>
  <p><strong>Sigh "Haa—" ×5</strong> (Like dropping a heavy load)</p>
  <ul>
    <li>Feel your shoulders drop and throat open.</li>
    <li>If you are pushing air: quiet down, make it a "warm breath."</li>
  </ul>

  <h3>Module B: Frontal Resonance (2–4 min)</h3>
  <ol>
    <li><strong>Closed-mouth "Mmm—"</strong>: Find vibration in lips/nose/front teeth.
      <div class="manual-practice-widget" data-phrase='{"id":"manual_hum_en","text":"Mmm——","tip":"Feel the buzz in your lips"}'></div>
    </li>
    <li><strong>"N" Onset</strong>: Start words with a light "N..."
      <ul>
        <li>Ex: N...Nice, N...No, N...Now.</li>
      </ul>
    </li>
    <li><strong>30% Smile</strong>: Slight lift of corners to brighten the resonance.</li>
  </ol>

  <h3>Module C: Staircase Method (5–8 min)</h3>
  <p><strong>Principle:</strong> Let the larynx raise naturally with pitch.</p>
  <ol>
    <li>"Mmm—" to find buzz.</li>
    <li>Glide up ("Mmm~~") like a siren.</li>
    <li>Cut the sound at the top, hold the "small box" feeling.</li>
    <li>Say a short phrase in that position (1-3s).
      <div class="manual-practice-widget" data-phrase='{"id":"manual_stair_hello_en","text":"Hello","tip":"Keep the resonance high"}'></div></li>
    <li>Relax and repeat.</li>
  </ol>

  <h3>Module D: Light Weight (2–4 min)</h3>
  <p>Target: <strong>Thin but with a core</strong> (Not whispery).</p>
  <ul>
    <li>Volume 4/10: Say "I know." ×10, "It's okay." ×10.
      <div class="manual-practice-widget" data-phrase='{"id":"manual_weight_iknow_en","text":"I know","tip":"Light but with a core"}'></div>
      <div class="manual-practice-widget" data-phrase='{"id":"manual_weight_okay_en","text":"It is okay","tip":"Light but with a core"}'></div>
    </li>
    <li>If too breathy: Add a little volume to engage the cords.</li>
  </ul>

  <h3>Module E: Clarity & Brightness (2–3 min)</h3>
  <p>Target: Clarify consonants.</p>
  <ul>
    <li>"He, She, See, Tea" ×10 (Quiet, clear).
      <div class="manual-practice-widget" data-phrase='{"id":"manual_clarity_he_en","text":"He, She, See, Tea","tip":"Quiet, clear"}'></div>
    </li>
  </ul>

  <h3>Module F: Prosody (3–5 min)</h3>
  <p><strong>1) The "Hold" (Natural)</strong></p>
  <ul>
    <li>"I see." / "Okay."
      <div class="manual-practice-widget" data-phrase='{"id":"manual_prosody_isee_en","text":"I see.","tip":"Hold the pitch"}'></div>
      <div class="manual-practice-widget" data-phrase='{"id":"manual_prosody_okay_hold_en","text":"Okay.","tip":"Hold the pitch"}'></div>
    </li>
    <li>Don't let the pitch drop at the period. Keep it level.</li>
  </ul>
  <p><strong>2) The "Micro-lift" (Sweet)</strong></p>
  <ul>
    <li>"Really?~" / "Okay~"
      <div class="manual-practice-widget" data-phrase='{"id":"manual_prosody_really_lift_en","text":"Really?~","tip":"Micro-lift"}'></div>
      <div class="manual-practice-widget" data-phrase='{"id":"manual_prosody_okay_lift_en","text":"Okay~","tip":"Micro-lift"}'></div>
    </li>
    <li>Very subtle upward tail.</li>
  </ul>
</section>

<section>
  <h2>6) Daily Schedule</h2>
  
  <div class="schedule-card">
    <h3>6.1 The 12-Minute "busy" Routine</h3>
    <ol>
      <li>Relax Haa— ×5 (1’)</li>
      <li>Mmm— Resonance (2’)</li>
      <li>Staircase Sirens ×6 (4’)</li>
      <li>Short phrases after siren (4’)</li>
      <li><strong>VPA Record 30s</strong> (1’): Check metrics.</li>
    </ol>
  </div>

  <div class="schedule-card">
    <h3>6.2 The 25-Minute "Standard" Routine</h3>
    <ol>
      <li>Relax (2’)</li>
      <li>Resonance (4’): Mmm + N-words</li>
      <li>Staircase (8’)</li>
      <li>Weight (3’)</li>
      <li>Clarity (3’)</li>
      <li>Prosody (4’): Focus on "Holding"</li>
      <li><strong>VPA Record 30s</strong> (1’): Check Pitch & Resonance stability.</li>
    </ol>
  </div>

  <div class="schedule-card">
    <h3>6.3 The 45-Minute "Speedrun" Routine</h3>
    <p>Add after standard:</p>
    <ul>
      <li>Reading Aloud (2 min)</li>
      <li>Self-Q&A (3 min)</li>
      <li>Extra Prosody (5 min)</li>
      <li>Sweetness Module (10 min) - If ready.</li>
    </ul>
  </div>
</section>

<section>
  <h2>7) Sweetness (Advanced)</h2>
  <p>Sweetness is:</p>
  <ol>
    <li><strong>Cleaner</strong> (Less fry/strain)</li>
    <li><strong>Brighter</strong> (More forward)</li>
    <li><strong>Polite Tails</strong> (Hold or Lift)</li>
  </ol>

  <h3>Sweetness Module</h3>
  <ol>
    <li>Smile 30% + Mmm (1’)</li>
    <li>Sweet Phrases (4’):
      <ul>
        <li>"Thank you so much~"
          <div class="manual-practice-widget" data-phrase='{"id":"manual_sweet_thanks_en","text":"Thank you so much~","tip":"Cleaner, Brighter"}'></div>
        </li>
        <li>"Can you help me?~"
          <div class="manual-practice-widget" data-phrase='{"id":"manual_sweet_trouble_en","text":"Can you help me?~","tip":"Cleaner, Brighter"}'></div>
        </li>
      </ul>
    </li>
    <li>Sweet Script (2’)
      <div class="manual-script">
        <p>"Hi~ Thanks for keeping me company.<br>
        I'll be right back, drive safe okay?~"</p>
        <div class="manual-practice-widget" data-phrase='{"id":"manual_sweet_script_en"}'></div>
      </div>
    </li>
  </ol>
</section>

<section>
  <h2>8) 7-Day Checklist</h2>
  <ul class="checklist">
    <li><strong>Day 1:</strong> Vibration + 6 Sirens + Recording</li>
    <li><strong>Day 2:</strong> Day 1 + N-words</li>
    <li><strong>Day 3:</strong> Add Prosody (Holding)</li>
    <li><strong>Day 4:</strong> Add "E/See/Tee" Clarity</li>
    <li><strong>Day 5:</strong> Hold "Ah—" for 3s ×10</li>
    <li><strong>Day 6:</strong> Self Q&A</li>
    <li><strong>Day 7:</strong> Compare Day 1 vs Day 7 recordings.</li>
  </ul>
</section>

<section>
  <h2>9) Troubleshooting</h2>
  <ul>
    <li><strong>Tight at high pitch</strong>: You are pushing, not shifting resonance. Go back to "Mmm".</li>
    <li><strong>High pitch but thick</strong>: Resonance is still in chest. Focus on Modules B + D.</li>
    <li><strong>Breathy/Weak</strong>: Leaking air. Increase volume slightly, engage core.</li>
    <li><strong>Nasally</strong>: Too much nose. Reduce smile, focus on "bright mouth" not "nose buzz".</li>
  </ul>
</section>

<section>
  <h2>10) Success Criteria</h2>
  <blockquote>
    <p><strong>Throat feels better, sound is brighter/forward, and playback sounds more feminine.</strong><br>
    Even if pitch isn't higher today, you are improving.</p>
  </blockquote>
  <p>(End)</p>
</section>
    `
  }
};
