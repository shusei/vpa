const GUIDANCE = {
  "zh-Hant": {
    labels: ["這項數據", "這次結果", "下一步"],
    insufficient: "可用的有聲片段不足，目前還無法形成穩定趨勢。",
    retry: "在安靜處用自然音量連續說 5–8 秒，再以相同裝置與距離比較。",
    compare: "用同一句話、同一裝置與距離再錄一次；一次只改一個元素，回聽哪個版本更符合你的目標。",
    model: {
      purpose: "綜合模型從這段錄音估計聲音呈現的女性化傾向，適合比較不同次錄音。",
      result: "百分比是本次錄音的模型傾向；較高代表模型在這次聽到較多其學得的女性化特徵，並不是身分或健康判定。",
      next: "先重播確認自然與舒服；想比較時，固定語句和收音條件再錄一次，觀察哪個版本更接近你想要的呈現。",
    },
    pitch: {
      purpose: "代表音高是偵測到的有聲片段之中位數，用來比較整段聲音的音高中心。",
      single: "畫面顯示這一次錄音的音高中位數；它會隨句子、情緒與說話方式改變。",
      standard: "畫面顯示三句各自音高中位數再取中位數，可減少單一句子的偶然波動。",
      next: "先回聽是否自然；若想比較音高方向，用同一句話做一個略高或略低、但仍舒服的版本。",
    },
    metrics: {
      resonance: {
        purpose: "比較低、中、高頻段的相對能量，描述這次錄音偏厚、集中或明亮的頻譜平衡；它不是身體共鳴位置的量測。",
        next: "若想更明亮清楚，可把同一句的子音與母音說清楚一點；若想更溫暖柔和，則改用較柔和的語氣。保持自然音量，回聽比較。",
        result: {
          insufficient: "可用的頻譜片段不足，三個頻段的比例目前不穩定。",
          balanced: "低、中、高頻能量較接近，這次的頻譜分布較平均。",
          headBright: "高頻能量占比較高，這次音色可能聽來較明亮或輕。",
          chestHeavy: "低頻能量占比較高，這次音色可能聽來較厚或暖。",
          maskLead: "中頻能量占比較高，這次音色可能聽來較集中或清楚。",
        },
      },
      tilt: {
        purpose: "比較低頻到高頻能量下降的速度，用來描述整體音色偏暖、平衡或明亮。",
        next: "若目標更明亮，可試一版較清楚的咬字；若目標更溫暖，可試一版較柔和的語氣。只做小幅變化並回聽亮暗差異。",
        result: {
          insufficient: "穩定的有聲頻譜不足，暫時無法估計傾斜趨勢。",
          warm: "低頻相對更強，這次頻譜呈現較明顯的暖厚傾向。",
          gentleWarm: "低頻略強，這次音色呈現溫暖但不極端的傾向。",
          balanced: "低、高頻能量落差較居中，這次亮暗分布較平衡。",
          bright: "高頻相對更活躍，這次頻譜呈現較明亮的傾向。",
        },
      },
      breathiness: {
        purpose: "以諧波與雜訊的頻譜關係估計聲音的空氣感，適合比較錄音；它不直接量測漏氣或聲帶閉合。",
        next: "想比較空氣感時，用同一句各錄一個稍柔和與較清楚的版本；保持自然音量，不要刻意大量漏氣或擠壓聲音。",
        result: {
          insufficient: "可用的有聲頻譜不足，空氣感代理值目前不穩定。",
          dense: "代理值較低，這次聲音的空氣感較少、質地較實。",
          balanced: "代理值位於應用程式的中段，這次空氣感較適中。",
          airy: "代理值較高，這次可能聽到較多空氣感。",
          style: "代理值偏高且錄音品質可用，這次呈現較明顯的空氣感風格。",
          tooAiry: "代理值很高，這次可能有強烈空氣感，也可能受到噪音或收音條件影響。",
        },
      },
      brightness: {
        purpose: "綜合 F3、頻譜傾斜與空氣感代理值，描述這次音色的明亮度；不是女性化程度的單獨分數。",
        next: "若想更亮，可稍微加強咬字清晰度；若想更暖，可用較柔和的子音與語氣。回聽實際音色，不要只追分類名稱。",
        result: {
          insufficient: "穩定的 F3 資料不足，目前無法形成明亮度分類。",
          balanced: "綜合訊號位於應用程式的中段，這次明亮度較平衡。",
          warm: "F3 相對較低，這次音色傾向較暖、較圓。",
          sparkle: "F3 相對較高，這次音色呈現較活躍的明亮感。",
          sweet: "F3 偏高且其他代理訊號較穩定，這次呈現集中而明亮的音色。",
          sweetMasculine: "F3 偏高且其他代理訊號較穩定，這次呈現集中而明亮的音色。",
          sparkleMasculine: "高頻訊號較活躍，這次呈現清楚而明亮的音色。",
          sharp: "F3 與高頻訊號都偏強，這次的明亮感可能較尖銳，也可能受收音影響。",
        },
      },
      vowel: {
        purpose: "計算有多少可用片段落在應用程式設定的 F1／F2 參考帶，用來比較同一句話的母音頻譜一致性。",
        next: "把同一句稍微放慢、清楚說完每個母音，再與自然版本比較比例和聽感；不要為了進入參考帶扭曲字音。",
        result: {
          insufficient: "可用的 F1／F2 片段不足，比例目前沒有代表性。",
          strong: "較多片段落在參考帶內，這次的母音頻譜較常符合此應用參考。",
          medium: "一部分片段在參考帶內、一部分在外，這次呈現混合分布。",
          weak: "較少片段落在參考帶內；語句中的母音種類也可能造成這個結果。",
        },
      },
      speech: {
        purpose: "估計每秒音節數，描述這次錄音的說話速度；它主要反映語句與個人節奏。",
        next: "若覺得太趕，就用同一句慢約一成；若覺得拖沓，就快約一成。回聽哪個版本仍清楚、自然且呼吸舒服。",
        result: {
          insufficient: "連續語音不足，目前無法得到穩定語速。",
          tooSlow: "每秒音節數低於應用程式的中段，這次節奏較慢、停留較多。",
          balanced: "每秒音節數位於應用程式的中段，這次節奏較適中。",
          fast: "每秒音節數高於應用程式的中段，這次節奏較快。",
        },
      },
      liaison: {
        purpose: "計算有聲片段之間短停頓的比例，描述這次語流較連續或較分段；它不是呼吸能力測試。",
        next: "同一句先自然說一次，再依語意分段、稍微縮短不必要的停頓；比較哪個版本既連貫又不需要勉強一口氣說完。",
        result: {
          insufficient: "有聲片段不足，短停頓比例目前不穩定。",
          strong: "較多片段以短停頓相連，這次語流較連續。",
          medium: "短停頓與較長停頓並存，這次語流連續度居中。",
          weak: "較少片段以短停頓相連，這次語流較分段；標點與句型也會影響結果。",
        },
      },
      slope: {
        purpose: "比較句子前後的音高走勢，描述這次語調整體上揚、平穩或下降。",
        next: "用同一句分別表達陳述、疑問或不同情緒，回聽哪種句尾最符合你要傳達的意思；不必固定追求上揚。",
        result: {
          rising: "處理後曲線由前往後整體上升，這次句尾傾向上揚。",
          flat: "處理後曲線前後差異較小，這次整體走勢較平穩。",
          falling: "處理後曲線由前往後整體下降，這次句尾傾向下降。",
        },
      },
      range: {
        purpose: "計算處理後音高曲線的高低跨度，描述這次語調變化幅度。",
        next: "同一句各錄一個自然版與表情稍明顯的版本，回聽清楚度與真實感；在舒服範圍內選擇適合的變化幅度。",
        result: {
          rich: "音高跨度較大，這次語調變化較豐富。",
          medium: "音高跨度位於應用程式的中段，這次語調變化適中。",
          narrow: "音高跨度較小，這次語調變化較集中。",
        },
      },
    },
    formants: {
      F1: { purpose: "F1 主要跟母音開口度相關，用來比較同一母音或同一句話的開口頻譜變化。", next: "選同一個母音，各錄一個開口稍大與稍小、但字音仍自然的版本，回聽並比較 F1；不要只用混合語音的單一數字下結論。", low: "F1 低於應用參考帶，這次母音整體呈現較低的第一共振峰。", inRange: "F1 位於應用參考帶內，這次第一共振峰接近此應用的比較範圍。", high: "F1 高於應用參考帶，這次母音整體呈現較高的第一共振峰。" },
      F2: { purpose: "F2 主要跟母音前後度相關，用來比較同一母音或同一句話的音色位置變化。", next: "用同一組母音清楚但自然地各錄兩次，第二次只微調母音前後感，再回聽比較 F2 與可懂度。", low: "F2 低於應用參考帶，這次母音整體呈現較低的第二共振峰。", inRange: "F2 位於應用參考帶內，這次第二共振峰接近此應用的比較範圍。", high: "F2 高於應用參考帶，這次母音整體呈現較高的第二共振峰。" },
      F3: { purpose: "F3 是較高階的音色共振峰，適合在相同母音與收音條件下比較明亮度變化。", next: "固定母音、裝置與距離，連同明亮度卡和實際聽感一起比較；不要把單一 F3 當成必須升高或降低的目標。", low: "F3 低於應用參考帶，這次較高階的共振峰整體偏低。", inRange: "F3 位於應用參考帶內，這次第三共振峰接近此應用的比較範圍。", high: "F3 高於應用參考帶，這次較高階的共振峰整體偏高。" },
    },
    insights: {
      balancedGrowth: ["綜合比較模型、音高、頻譜與錄音品質是否朝相近方向。", "這次各項指標較一致，沒有明顯互相矛盾。", "重播確認自然與舒服；若喜歡這次，可把相同語句與收音距離記作之後的比較基準。"],
      consistencyOpportunity: ["檢查同一段錄音內的訊號是否穩定，避免用短暫波動代表整體。", "這次不同片段的變化較大，分數可能較受語句或收音影響。", "固定距離與自然音量，把同一句再錄一次，先比較兩次是否接近。"],
      falsettoContrast: ["比較音高與頻譜是否提供一致的聲音呈現線索。", "這次音高與頻譜走向不同，母音、語句或麥克風角度都可能造成。", "不要強迫兩者一致；用同一句、同距離再錄一版，回聽哪次更符合你的目標。"],
      insufficient: ["確認錄音是否有足夠的有聲片段供模型與聲學分析。", "這次可用資料不足，因此不適合解讀細項趨勢。", "在安靜處以自然音量連續說 5–8 秒，再重新分析。"],
      pitchOpportunity: ["比較音高線索與頻譜線索在綜合結果中的相對方向。", "這次頻譜訊號較明顯，音高方向相對不同。", "固定其他條件，只做一個略高或略低但舒服的版本，回聽哪個更符合你的呈現目標。"],
      resonanceOpportunity: ["比較頻譜／共振代理訊號與音高線索是否朝相近方向。", "這次音高訊號較明顯，頻譜方向相對不同；母音也會影響結果。", "用同一句與同裝置，只改一次母音清晰度，回聽並比較頻譜卡是否穩定。"],
      strongIntegration: ["綜合檢查模型、音高與頻譜是否同時提供一致訊號。", "這次多項訊號方向一致，綜合傾向較明確。", "先確認聲音自然、舒服且可重複；符合目標就保存，不必為更高分再勉強改動。"],
    },
  },
  "zh-Hans": {
    labels: ["这项数据", "这次结果", "下一步"],
    insufficient: "可用的有声片段不足，目前还无法形成稳定趋势。",
    retry: "在安静处用自然音量连续说 5–8 秒，再以相同设备与距离比较。",
    compare: "用同一句话、同一设备与距离再录一次；一次只改一个元素，回听哪个版本更符合你的目标。",
    model: { purpose: "综合模型从这段录音估计声音呈现的女性化倾向，适合比较不同次录音。", result: "百分比是本次录音的模型倾向；较高代表模型在这次听到较多其学到的女性化特征，并不是身份或健康判断。", next: "先回放确认自然与舒服；想比较时，固定语句和收音条件再录一次，观察哪个版本更接近你想要的呈现。" },
    pitch: { purpose: "代表音高是检测到的有声片段之中位数，用来比较整段声音的音高中心。", single: "画面显示这一次录音的音高中位数；它会随句子、情绪与说话方式改变。", standard: "画面显示三句各自音高中位数再取中位数，可减少单一句子的偶然波动。", next: "先回听是否自然；若想比较音高方向，用同一句话做一个略高或略低、但仍舒服的版本。" },
    metrics: {
      resonance: { purpose: "比较低、中、高频段的相对能量，描述这次录音偏厚、集中或明亮的频谱平衡；它不是身体共鸣位置的测量。", next: "若想更明亮清楚，可把同一句的辅音与元音说清楚一点；若想更温暖柔和，则改用较柔和的语气。保持自然音量，回听比较。", result: { insufficient: "可用的频谱片段不足，三个频段的比例目前不稳定。", balanced: "低、中、高频能量较接近，这次的频谱分布较平均。", headBright: "高频能量占比较高，这次音色可能听来较明亮或轻。", chestHeavy: "低频能量占比较高，这次音色可能听来较厚或暖。", maskLead: "中频能量占比较高，这次音色可能听来较集中或清楚。" } },
      tilt: { purpose: "比较低频到高频能量下降的速度，用来描述整体音色偏暖、平衡或明亮。", next: "若目标更明亮，可试一版更清楚的咬字；若目标更温暖，可试一版更柔和的语气。只做小幅变化并回听亮暗差异。", result: { insufficient: "稳定的有声频谱不足，暂时无法估计倾斜趋势。", warm: "低频相对更强，这次频谱呈现较明显的暖厚倾向。", gentleWarm: "低频略强，这次音色呈现温暖但不极端的倾向。", balanced: "低、高频能量落差较居中，这次亮暗分布较平衡。", bright: "高频相对更活跃，这次频谱呈现较明亮的倾向。" } },
      breathiness: { purpose: "以谐波与噪声的频谱关系估计声音的空气感，适合比较录音；它不直接测量漏气或声带闭合。", next: "想比较空气感时，用同一句各录一个稍柔和与较清楚的版本；保持自然音量，不要刻意大量漏气或挤压声音。", result: { insufficient: "可用的有声频谱不足，空气感代理值目前不稳定。", dense: "代理值较低，这次声音的空气感较少、质地较实。", balanced: "代理值位于应用程序的中段，这次空气感较适中。", airy: "代理值较高，这次可能听到较多空气感。", style: "代理值偏高且录音质量可用，这次呈现较明显的空气感风格。", tooAiry: "代理值很高，这次可能有强烈空气感，也可能受到噪声或收音条件影响。" } },
      brightness: { purpose: "综合 F3、频谱倾斜与空气感代理值，描述这次音色的明亮度；不是女性化程度的单独分数。", next: "若想更亮，可稍微加强咬字清晰度；若想更暖，可用更柔和的辅音与语气。回听实际音色，不要只追分类名称。", result: { insufficient: "稳定的 F3 数据不足，目前无法形成明亮度分类。", balanced: "综合信号位于应用程序的中段，这次明亮度较平衡。", warm: "F3 相对较低，这次音色倾向较暖、较圆。", sparkle: "F3 相对较高，这次音色呈现较活跃的明亮感。", sweet: "F3 偏高且其他代理信号较稳定，这次呈现集中而明亮的音色。", sweetMasculine: "F3 偏高且其他代理信号较稳定，这次呈现集中而明亮的音色。", sparkleMasculine: "高频信号较活跃，这次呈现清楚而明亮的音色。", sharp: "F3 与高频信号都偏强，这次的明亮感可能较尖锐，也可能受收音影响。" } },
      vowel: { purpose: "计算有多少可用片段落在应用程序设置的 F1／F2 参考带，用来比较同一句话的元音频谱一致性。", next: "把同一句稍微放慢、清楚说完每个元音，再与自然版本比较比例和听感；不要为了进入参考带扭曲字音。", result: { insufficient: "可用的 F1／F2 片段不足，比例目前没有代表性。", strong: "较多片段落在参考带内，这次的元音频谱较常符合此应用参考。", medium: "一部分片段在参考带内、一部分在外，这次呈现混合分布。", weak: "较少片段落在参考带内；语句中的元音种类也可能造成这个结果。" } },
      speech: { purpose: "估计每秒音节数，描述这次录音的说话速度；它主要反映语句与个人节奏。", next: "若觉得太赶，就用同一句慢约一成；若觉得拖沓，就快约一成。回听哪个版本仍清楚、自然且呼吸舒服。", result: { insufficient: "连续语音不足，目前无法得到稳定语速。", tooSlow: "每秒音节数低于应用程序的中段，这次节奏较慢、停留较多。", balanced: "每秒音节数位于应用程序的中段，这次节奏较适中。", fast: "每秒音节数高于应用程序的中段，这次节奏较快。" } },
      liaison: { purpose: "计算有声片段之间短停顿的比例，描述这次语流较连续或较分段；它不是呼吸能力测试。", next: "同一句先自然说一次，再按语意分段、稍微缩短不必要的停顿；比较哪个版本既连贯又不需要勉强一口气说完。", result: { insufficient: "有声片段不足，短停顿比例目前不稳定。", strong: "较多片段以短停顿相连，这次语流较连续。", medium: "短停顿与较长停顿并存，这次语流连续度居中。", weak: "较少片段以短停顿相连，这次语流较分段；标点与句型也会影响结果。" } },
      slope: { purpose: "比较句子前后的音高走势，描述这次语调整体上扬、平稳或下降。", next: "用同一句分别表达陈述、疑问或不同情绪，回听哪种句尾最符合你要传达的意思；不必固定追求上扬。", result: { rising: "处理后曲线由前往后整体上升，这次句尾倾向上扬。", flat: "处理后曲线前后差异较小，这次整体走势较平稳。", falling: "处理后曲线由前往后整体下降，这次句尾倾向下降。" } },
      range: { purpose: "计算处理后音高曲线的高低跨度，描述这次语调变化幅度。", next: "同一句各录一个自然版与表情稍明显的版本，回听清楚度与真实感；在舒服范围内选择适合的变化幅度。", result: { rich: "音高跨度较大，这次语调变化较丰富。", medium: "音高跨度位于应用程序的中段，这次语调变化适中。", narrow: "音高跨度较小，这次语调变化较集中。" } },
    },
    formants: {
      F1: { purpose: "F1 主要跟元音开口度相关，用来比较同一元音或同一句话的开口频谱变化。", next: "选同一个元音，各录一个开口稍大与稍小、但字音仍自然的版本，回听并比较 F1；不要只用混合语音的单一数字下结论。", low: "F1 低于应用参考带，这次元音整体呈现较低的第一共振峰。", inRange: "F1 位于应用参考带内，这次第一共振峰接近此应用的比较范围。", high: "F1 高于应用参考带，这次元音整体呈现较高的第一共振峰。" },
      F2: { purpose: "F2 主要跟元音前后度相关，用来比较同一元音或同一句话的音色位置变化。", next: "用同一组元音清楚但自然地各录两次，第二次只微调元音前后感，再回听比较 F2 与可懂度。", low: "F2 低于应用参考带，这次元音整体呈现较低的第二共振峰。", inRange: "F2 位于应用参考带内，这次第二共振峰接近此应用的比较范围。", high: "F2 高于应用参考带，这次元音整体呈现较高的第二共振峰。" },
      F3: { purpose: "F3 是较高阶的音色共振峰，适合在相同元音与收音条件下比较明亮度变化。", next: "固定元音、设备与距离，连同明亮度卡和实际听感一起比较；不要把单一 F3 当成必须升高或降低的目标。", low: "F3 低于应用参考带，这次较高阶的共振峰整体偏低。", inRange: "F3 位于应用参考带内，这次第三共振峰接近此应用的比较范围。", high: "F3 高于应用参考带，这次较高阶的共振峰整体偏高。" },
    },
    insights: {
      balancedGrowth: ["综合比较模型、音高、频谱与录音质量是否朝相近方向。", "这次各项指标较一致，没有明显互相矛盾。", "回放确认自然与舒服；若喜欢这次，可把相同语句与收音距离记作之后的比较基准。"],
      consistencyOpportunity: ["检查同一段录音内的信号是否稳定，避免用短暂波动代表整体。", "这次不同片段的变化较大，分数可能较受语句或收音影响。", "固定距离与自然音量，把同一句再录一次，先比较两次是否接近。"],
      falsettoContrast: ["比较音高与频谱是否提供一致的声音呈现线索。", "这次音高与频谱走向不同，元音、语句或麦克风角度都可能造成。", "不要强迫两者一致；用同一句、同距离再录一版，回听哪次更符合你的目标。"],
      insufficient: ["确认录音是否有足够的有声片段供模型与声学分析。", "这次可用数据不足，因此不适合解读细项趋势。", "在安静处以自然音量连续说 5–8 秒，再重新分析。"],
      pitchOpportunity: ["比较音高线索与频谱线索在综合结果中的相对方向。", "这次频谱信号较明显，音高方向相对不同。", "固定其他条件，只做一个略高或略低但舒服的版本，回听哪个更符合你的呈现目标。"],
      resonanceOpportunity: ["比较频谱／共鸣代理信号与音高线索是否朝相近方向。", "这次音高信号较明显，频谱方向相对不同；元音也会影响结果。", "用同一句与同一设备，只改一次元音清晰度，回听并比较频谱卡是否稳定。"],
      strongIntegration: ["综合检查模型、音高与频谱是否同时提供一致信号。", "这次多项信号方向一致，综合倾向较明确。", "先确认声音自然、舒服且可重复；符合目标就保存，不必为更高分再勉强改动。"],
    },
  },
  en: {
    labels: ["What it shows", "This take", "Next try"],
    insufficient: "There are too few usable voiced frames for a stable trend.",
    retry: "Speak continuously for 5–8 seconds at a natural volume in a quiet place, then compare with the same device and distance.",
    compare: "Record the same line again with the same device and distance. Change one element only, then replay both and choose what fits your goal.",
    model: { purpose: "The composite model estimates the feminine presentation tendency of this recording so you can compare takes.", result: "The percentage is the model tendency for this take. A higher value means the model detected more features it learned as feminine; it is not an identity or health judgment.", next: "Replay first and check that it sounds natural and felt comfortable. For comparison, match the wording and recording setup, then keep the take that better fits your goal." },
    pitch: { purpose: "Representative pitch is the median of detected voiced frames and describes the pitch center of the whole take.", single: "This is the median pitch of this recording; wording, emotion, and delivery can move it.", standard: "This is the median of the three sentence-level pitch medians, reducing the effect of one unusual line.", next: "Replay for naturalness. To explore pitch, record the same line slightly higher or lower while staying comfortable." },
    metrics: {
      resonance: { purpose: "Compares relative low-, mid-, and high-band energy to describe a thicker, focused, or brighter spectral balance; it does not measure a resonance location in the body.", next: "For a brighter, clearer result, try the same line with slightly clearer consonants and vowels. For a warmer result, try a gentler delivery. Keep natural volume and replay both.", result: { insufficient: "There are too few usable spectral frames for stable band proportions.", balanced: "Low-, mid-, and high-band energy are relatively close, giving this take a more even spectral balance.", headBright: "High-band energy has the largest share, so this take may sound brighter or lighter.", chestHeavy: "Low-band energy has the largest share, so this take may sound thicker or warmer.", maskLead: "Mid-band energy has the largest share, so this take may sound more focused or clear." } },
      tilt: { purpose: "Compares how quickly energy falls from low to high frequencies to describe an overall warm, balanced, or bright timbre.", next: "For a brighter goal, try clearer articulation; for a warmer goal, try a gentler delivery. Make only a small change and replay the bright–dark difference.", result: { insufficient: "There is not enough stable voiced spectrum to estimate a tilt trend.", warm: "Low frequencies are relatively stronger, giving this take a clearly warmer, thicker spectral trend.", gentleWarm: "Low frequencies are slightly stronger, giving this take a mildly warm trend.", balanced: "The low-to-high energy difference is more central, giving this take a balanced bright–dark distribution.", bright: "High frequencies are relatively more active, giving this take a brighter spectral trend." } },
      breathiness: { purpose: "Uses harmonic-to-noise spectral relationships as a proxy for perceived airiness. It does not directly measure airflow or vocal-fold closure.", next: "Record the same line once with a slightly gentler delivery and once with clearer voicing. Keep natural volume; do not force extra air or compression.", result: { insufficient: "There are too few usable voiced spectra for a stable airiness proxy.", dense: "The proxy is lower, suggesting less airiness and a denser texture in this take.", balanced: "The proxy is in the app's middle band, suggesting moderate airiness in this take.", airy: "The proxy is higher, so this take may have more audible airiness.", style: "The proxy is elevated while recording quality is usable, giving this take a noticeably airy style.", tooAiry: "The proxy is very high, suggesting strong airiness, though noise or recording conditions may also contribute." } },
      brightness: { purpose: "Combines F3, spectral tilt, and the airiness proxy to describe timbre brightness; it is not a stand-alone femininity score.", next: "For more brightness, try slightly clearer articulation; for more warmth, try gentler consonants and delivery. Judge the replay, not the category name alone.", result: { insufficient: "There is not enough stable F3 data to classify brightness.", balanced: "The combined signals are near the app's middle band, giving this take balanced brightness.", warm: "F3 is relatively lower, giving this take a warmer, rounder timbre.", sparkle: "F3 is relatively higher, giving this take a more active bright quality.", sweet: "F3 is high while the other proxies are steadier, giving this take a focused bright timbre.", sweetMasculine: "F3 is high while the other proxies are steadier, giving this take a focused bright timbre.", sparkleMasculine: "High-frequency signals are more active, giving this take a clear bright timbre.", sharp: "F3 and high-frequency signals are both strong, so brightness may sound sharp or may reflect the recording setup." } },
      vowel: { purpose: "Counts usable frames inside the app's F1/F2 reference band to compare vowel-spectrum consistency for the same line.", next: "Say the same line slightly slower with each vowel clear, then compare its ratio and sound with your natural take. Do not distort words to enter the reference band.", result: { insufficient: "There are too few usable F1/F2 frames for a representative ratio.", strong: "More frames fall inside the reference band, so this take more often matches the app reference.", medium: "Some frames are inside and some outside the reference band, giving this take a mixed distribution.", weak: "Fewer frames fall inside the reference band; the mix of vowels in the sentence can also cause this." } },
      speech: { purpose: "Estimates syllables per second to describe speaking pace; it mainly reflects the sentence and personal rhythm.", next: "If it feels rushed, record the same line about 10% slower; if it drags, try about 10% faster. Replay for clarity, naturalness, and comfortable breathing.", result: { insufficient: "There is not enough continuous speech for a stable rate.", tooSlow: "Syllables per second are below the app's middle band, so this take has a slower pace with more dwell time.", balanced: "Syllables per second are in the app's middle band, so this take has a moderate pace.", fast: "Syllables per second are above the app's middle band, so this take has a faster pace." } },
      liaison: { purpose: "Measures the share of short gaps between voiced segments to describe continuous versus segmented flow; it is not a breathing-capacity test.", next: "Say the line naturally, then repeat it in meaningful phrases with unnecessary gaps slightly shorter. Choose the version that flows without forcing one long breath.", result: { insufficient: "There are too few voiced segments for a stable short-gap ratio.", strong: "More segments are joined by short gaps, so this take has more continuous flow.", medium: "Short and longer gaps are both present, giving this take medium continuity.", weak: "Fewer segments are joined by short gaps, so this take is more segmented; punctuation and sentence type also matter." } },
      slope: { purpose: "Compares pitch near the beginning and end to describe an overall rising, level, or falling intonation trend.", next: "Say the same line as a statement, question, or with different emotion. Replay which ending communicates your meaning best; a rising ending is not a required target.", result: { rising: "The processed curve rises overall, so this take tends toward a rising ending.", flat: "The processed curve changes little from beginning to end, so this take is comparatively level.", falling: "The processed curve falls overall, so this take tends toward a falling ending." } },
      range: { purpose: "Measures the high-to-low span of the processed pitch curve to describe the amount of intonation movement.", next: "Record one natural version and one slightly more expressive version. Replay for clarity and authenticity, then choose a comfortable amount of movement.", result: { rich: "The pitch span is larger, so this take has more varied intonation.", medium: "The pitch span is in the app's middle band, so this take has moderate intonation movement.", narrow: "The pitch span is smaller, so this take has more concentrated intonation." } },
    },
    formants: {
      F1: { purpose: "F1 mainly relates to vowel openness and helps compare opening-related spectral changes in the same vowel or line.", next: "Use one vowel and record a slightly more-open and less-open version while keeping the word natural. Replay and compare F1; do not interpret one mixed-speech number alone.", low: "F1 is below the app reference band, so this take has a lower overall first-formant pattern.", inRange: "F1 is inside the app reference band, close to this app's comparison range.", high: "F1 is above the app reference band, so this take has a higher overall first-formant pattern." },
      F2: { purpose: "F2 mainly relates to vowel frontness/backness and helps compare timbre-position changes in the same vowel or line.", next: "Record the same vowels clearly and naturally twice, changing vowel frontness only slightly on the second take. Replay for F2 and intelligibility.", low: "F2 is below the app reference band, so this take has a lower overall second-formant pattern.", inRange: "F2 is inside the app reference band, close to this app's comparison range.", high: "F2 is above the app reference band, so this take has a higher overall second-formant pattern." },
      F3: { purpose: "F3 is a higher timbre formant, most useful for comparing brightness changes with matched vowels and recording conditions.", next: "Match the vowel, device, and distance, then compare F3 together with the brightness card and replay. Do not treat F3 alone as a target to raise or lower.", low: "F3 is below the app reference band, so this take has a lower overall third-formant pattern.", inRange: "F3 is inside the app reference band, close to this app's comparison range.", high: "F3 is above the app reference band, so this take has a higher overall third-formant pattern." },
    },
    insights: {
      balancedGrowth: ["Compares whether model, pitch, spectrum, and recording-quality evidence point in similar directions.", "The indicators are relatively aligned in this take without a strong contradiction.", "Replay for comfort and naturalness. If you like it, save the wording and microphone distance as a future baseline."],
      consistencyOpportunity: ["Checks whether evidence stays stable within the take rather than treating a brief fluctuation as the whole voice.", "Indicators vary more across this take, so wording or recording conditions may influence the score.", "Match distance and natural volume, record the same line once more, and first check whether the two results agree."],
      falsettoContrast: ["Compares whether pitch and spectral evidence describe a similar presentation.", "Pitch and spectral evidence point in different directions; vowels, wording, or microphone angle can cause this.", "Do not force them to match. Record the same line at the same distance and choose the take that better fits your goal."],
      insufficient: ["Checks whether there is enough voiced material for model and acoustic analysis.", "This take has too little usable data for meaningful detail trends.", "Speak continuously for 5–8 seconds at a natural volume in a quiet place, then analyze again."],
      pitchOpportunity: ["Compares the relative direction of pitch and spectral evidence in the composite result.", "Spectral evidence is stronger while the pitch direction differs in this take.", "Keep other conditions fixed and try one comfortably higher or lower version, then replay both against your goal."],
      resonanceOpportunity: ["Compares spectral/resonance proxies with pitch evidence in the composite result.", "Pitch evidence is stronger while the spectral direction differs; vowel content can also affect this.", "Use the same line and device, change vowel clarity once, then replay and see whether the spectral cards become more consistent."],
      strongIntegration: ["Checks whether model, pitch, and spectral evidence all point in a similar direction.", "Several signals align, making the composite tendency clearer in this take.", "Confirm that it feels natural, comfortable, and repeatable. If it fits your goal, save it rather than forcing a higher score."],
    },
  },
  ja: {
    labels: ["この数値", "今回の結果", "次に試すこと"],
    insufficient: "利用できる有声音フレームが少なく、安定した傾向をまだ示せません。",
    retry: "静かな場所で自然な音量のまま5〜8秒続けて話し、同じ端末と距離で比較してください。",
    compare: "同じ文・端末・距離でもう一度録音します。一度に一要素だけ変え、再生して目標に合う方を選びます。",
    model: { purpose: "複合モデルが今回の録音の女性的な声の提示傾向を推定し、テイク同士の比較に使います。", result: "割合は今回の録音に対するモデル傾向です。高いほど、モデルが学習した女性的特徴を多く検出したことを示しますが、性自認や健康の判定ではありません。", next: "まず再生して自然さと快適さを確認します。比較する場合は文と録音条件をそろえ、自分の目標に合うテイクを残してください。" },
    pitch: { purpose: "代表ピッチは検出された有声音フレームの中央値で、テイク全体のピッチ中心を比較します。", single: "今回1回の録音のピッチ中央値です。文、感情、話し方で変化します。", standard: "3文それぞれのピッチ中央値からさらに中央値を取り、1文だけの偶然の変動を減らしています。", next: "自然に聞こえるか再生します。ピッチを比べるなら、同じ文を無理のない範囲で少し高く／低く録音してください。" },
    metrics: {
      resonance: { purpose: "低・中・高域の相対エネルギーを比べ、厚い・集中した・明るいスペクトルバランスを記述します。体内の共鳴位置を測るものではありません。", result: { insufficient: "利用できるスペクトルフレームが少なく、3帯域の比率が安定していません。", balanced: "低・中・高域のエネルギーが比較的近く、今回の分布はより均等です。", headBright: "高域の割合が高く、今回の音色はより明るく軽く聞こえる可能性があります。", chestHeavy: "低域の割合が高く、今回の音色はより厚く温かく聞こえる可能性があります。", maskLead: "中域の割合が高く、今回の音色はより集中して明瞭に聞こえる可能性があります。" } },
      tilt: { purpose: "低域から高域へエネルギーが下がる速さを比べ、全体の音色を温かい・均衡・明るい傾向として記述します。", result: { insufficient: "安定した有声スペクトルが不足し、傾斜を推定できません。", warm: "低域が相対的に強く、今回は明確に温かく厚い傾向です。", gentleWarm: "低域がやや強く、今回は穏やかに温かい傾向です。", balanced: "低域と高域の差が中ほどで、今回は明暗のバランスが比較的均等です。", bright: "高域が相対的に活発で、今回はより明るい傾向です。" } },
      breathiness: { purpose: "調波と雑音のスペクトル関係から空気感を推定します。呼気量や発声器官の動きを直接測るものではありません。", result: { insufficient: "利用できる有声スペクトルが少なく、空気感の代理値が安定していません。", dense: "代理値が低く、今回は空気感が少ない密な質感です。", balanced: "代理値がアプリの中間帯にあり、今回は空気感が中程度です。", airy: "代理値が高く、今回は空気感がより多く聞こえる可能性があります。", style: "録音品質が利用可能な状態で代理値が高く、今回は空気感の強いスタイルです。", tooAiry: "代理値が非常に高く、強い空気感または雑音・録音条件の影響が考えられます。" } },
      brightness: { purpose: "F3、スペクトル傾斜、空気感代理値を組み合わせて音色の明るさを記述します。単独の女性らしさ得点ではありません。", result: { insufficient: "安定したF3データが不足し、明るさを分類できません。", balanced: "複合信号がアプリの中間帯にあり、今回は明るさが比較的均衡しています。", warm: "F3が相対的に低く、今回はより温かく丸い音色です。", sparkle: "F3が相対的に高く、今回は活発な明るさがあります。", sweet: "F3が高く他の代理信号が比較的安定し、今回は集中した明るい音色です。", sweetMasculine: "F3が高く他の代理信号が比較的安定し、今回は集中した明るい音色です。", sparkleMasculine: "高域信号が活発で、今回は明瞭で明るい音色です。", sharp: "F3と高域信号がともに強く、明るさが鋭く聞こえるか、録音条件の影響を受けた可能性があります。" } },
      vowel: { purpose: "利用可能フレームのうちアプリのF1/F2参考帯に入る割合を数え、同じ文の母音スペクトルの一貫性を比較します。", result: { insufficient: "利用できるF1/F2フレームが少なく、割合に代表性がありません。", strong: "参考帯に入るフレームが多く、今回はアプリ参考に合う母音スペクトルが多めです。", medium: "参考帯の内外にフレームが分かれ、今回は混合した分布です。", weak: "参考帯に入るフレームが少なめです。文に含まれる母音の種類でも変化します。" } },
      speech: { purpose: "1秒あたりの音節数を推定し、今回の話速を記述します。主に文と本人のリズムを反映します。", result: { insufficient: "連続発話が不足し、安定した話速を得られません。", tooSlow: "音節/秒がアプリの中間帯より低く、今回はゆっくりしたテンポです。", balanced: "音節/秒がアプリの中間帯にあり、今回は中程度のテンポです。", fast: "音節/秒がアプリの中間帯より高く、今回は速いテンポです。" } },
      liaison: { purpose: "有声区間どうしを結ぶ短い間の割合から、連続的／分節的な話し方を記述します。呼吸能力の検査ではありません。", result: { insufficient: "有声区間が少なく、短い間の割合が安定していません。", strong: "短い間でつながる区間が多く、今回はより連続した流れです。", medium: "短い間と長い間が混在し、今回は中程度の連続性です。", weak: "短い間でつながる区間が少なく、今回はより分節的です。句読点や文型でも変化します。" } },
      slope: { purpose: "文の前後のピッチを比べ、全体を上昇・平坦・下降の傾向として記述します。", result: { rising: "処理後の曲線が全体に上がり、今回は語尾が上昇する傾向です。", flat: "処理後の曲線の前後差が小さく、今回は比較的平坦です。", falling: "処理後の曲線が全体に下がり、今回は語尾が下降する傾向です。" } },
      range: { purpose: "処理後ピッチ曲線の高低幅を測り、イントネーション変化量を記述します。", result: { rich: "ピッチ幅が大きく、今回はイントネーション変化が豊かです。", medium: "ピッチ幅がアプリの中間帯にあり、今回は中程度の変化です。", narrow: "ピッチ幅が小さく、今回は変化がより集中しています。" } },
    },
    formants: {
      F1: { purpose: "F1は主に母音の開口度と関係し、同じ母音や文で開口に伴うスペクトル変化を比べます。", low: "F1がアプリ参考帯より低く、今回は第1フォルマントが全体に低めです。", inRange: "F1がアプリ参考帯内にあり、このアプリの比較範囲に近い結果です。", high: "F1がアプリ参考帯より高く、今回は第1フォルマントが全体に高めです。" },
      F2: { purpose: "F2は主に母音の前後性と関係し、同じ母音や文で音色位置の変化を比べます。", low: "F2がアプリ参考帯より低く、今回は第2フォルマントが全体に低めです。", inRange: "F2がアプリ参考帯内にあり、このアプリの比較範囲に近い結果です。", high: "F2がアプリ参考帯より高く、今回は第2フォルマントが全体に高めです。" },
      F3: { purpose: "F3は高次の音色フォルマントで、母音と録音条件をそろえた明るさ比較に向きます。", low: "F3がアプリ参考帯より低く、今回は第3フォルマントが全体に低めです。", inRange: "F3がアプリ参考帯内にあり、このアプリの比較範囲に近い結果です。", high: "F3がアプリ参考帯より高く、今回は第3フォルマントが全体に高めです。" },
    },
    insights: {
      balancedGrowth: ["モデル、ピッチ、スペクトル、録音品質が近い方向を示すか比較します。", "今回は各指標が比較的一致し、大きな矛盾はありません。", "自然さと快適さを再生で確認し、気に入れば文とマイク距離を今後の基準として残してください。"],
      consistencyOpportunity: ["短い変動を声全体と見なさないよう、録音内の信号安定性を確認します。", "今回は区間ごとの変化が大きく、文や録音条件が得点に影響した可能性があります。", "距離と自然な音量をそろえ、同じ文をもう一度録音して2回が近いか確認してください。"],
      falsettoContrast: ["ピッチとスペクトルが似た声の提示を示すか比較します。", "今回は両者の方向が異なり、母音、文、マイク角度の影響も考えられます。", "一致させようと無理をせず、同じ文と距離で再録音し、目標に合う方を選んでください。"],
      insufficient: ["モデルと音響分析に十分な有声音があるか確認します。", "今回は利用可能データが少なく、細かな傾向を解釈できません。", "静かな場所で自然な音量のまま5〜8秒続けて話し、再分析してください。"],
      pitchOpportunity: ["複合結果におけるピッチとスペクトル証拠の相対方向を比較します。", "今回はスペクトル信号が強く、ピッチ方向が相対的に異なります。", "他の条件を固定し、無理のない範囲で少し高い／低い版を一つ録音して目標と比べてください。"],
      resonanceOpportunity: ["スペクトル／共鳴代理信号とピッチ証拠が近い方向を示すか比較します。", "今回はピッチ信号が強く、スペクトル方向が相対的に異なります。母音の影響もあります。", "同じ文と端末で母音の明瞭さだけを一度変え、再生とスペクトルカードを比較してください。"],
      strongIntegration: ["モデル、ピッチ、スペクトルが同時に近い方向を示すか確認します。", "今回は複数の信号が一致し、複合傾向がより明確です。", "自然・快適・再現可能か確認し、目標に合えば保存してください。高得点のために無理をする必要はありません。"],
    },
  },
};

const JA_METRIC_NEXT = {
  resonance: "より明るく明瞭にしたい場合は同じ文の子音と母音を少し明確に、より温かくしたい場合は穏やかな話し方で録音し、自然な音量で再生比較します。",
  tilt: "明るさを目指すなら明瞭な発音、温かさなら穏やかな話し方を少しだけ試し、再生して明暗差を比べます。",
  breathiness: "同じ文を少し柔らかい版と明瞭な版で録音し、自然な音量を保ちます。息や圧迫を無理に増やさないでください。",
  brightness: "より明るくしたいなら発音を少し明瞭に、温かくしたいなら子音と話し方を穏やかにし、分類名ではなく再生音で選びます。",
  vowel: "同じ文を少しゆっくり、各母音を明瞭に言い、自然版と割合・聞こえ方を比べます。参考帯のために語音をゆがめないでください。",
  speech: "急いで聞こえるなら約1割遅く、間延びするなら約1割速く同じ文を録音し、明瞭さ・自然さ・楽な呼吸で選びます。",
  liaison: "自然版の後、意味のまとまりごとに不要な間を少し短くした版を録音し、一息で無理をせず流れを比較します。",
  slope: "同じ文を陳述・疑問・別の感情で録音し、意味に合う語尾を選びます。上昇を固定目標にしません。",
  range: "自然版と少し表情豊かな版を録音し、明瞭さと自分らしさを再生比較して快適な変化幅を選びます。",
};

const JA_FORMANT_NEXT = {
  F1: "同じ母音を、語音が自然な範囲で口を少し大きく／小さくした2版で録音し、F1と再生音を比べます。混合発話の単一値だけで結論を出さないでください。",
  F2: "同じ母音を明瞭かつ自然に2回録音し、2回目だけ母音の前後感を少し変えて、F2と聞き取りやすさを比べます。",
  F3: "母音・端末・距離をそろえ、F3を明るさカードと実際の再生音と一緒に比較します。F3だけを上げ下げする目標にしないでください。",
};

for (const [key, next] of Object.entries(JA_METRIC_NEXT)) GUIDANCE.ja.metrics[key].next = next;
for (const [key, next] of Object.entries(JA_FORMANT_NEXT)) GUIDANCE.ja.formants[key].next = next;

const EXTRA_GUIDANCE = {
  "zh-Hant": {
    components: {
      model: ["顯示模型從整段錄音辨識到的聲音呈現特徵，是綜合分數的一個來源。", "本次模型分項為 {{value}}%；套用 {{weight}}% 權重後參與總分。它不是性別或健康判定。", "先回聽自然度與舒適度；比較時固定句子與錄音條件，保留最符合自己目標的版本。"],
      resonance: ["彙整低、中、高頻段的相對能量，作為本次音色平衡的聲學證據。", "本次共鳴證據分項為 {{value}}%；套用 {{weight}}% 權重後參與總分。", "想更明亮可稍微說清楚子音與母音；想更溫暖可改用柔和語氣，維持自然音量後回聽比較。"],
      pitch: ["比較這次有聲片段的音高中心與應用模型，是綜合分數中的其中一項。", "本次音高分項為 {{value}}%；套用 {{weight}}% 權重後參與總分。單一分項不代表整體聲音。", "用同一句錄一個自然版與小幅升高或降低版，回聽哪個較自然、舒服且符合目標。"],
      intonation: ["彙整句中音高走向與變化幅度，描述這次語調的動態表現。", "本次語調分項為 {{value}}%；套用 {{weight}}% 權重後參與總分。", "把同一句用陳述、疑問或不同情緒各說一次，選擇最能傳達意思且不費力的版本。"],
    },
    voiceQuality: {
      jitter: ["Jitter 描述相鄰可用週期的頻率變動；在穩定持續母音且錄音品質足夠時較適合比較。", "本次為 {{value}}；{{status}}。日常語句、噪音與週期不足都會影響數字。", "若要比較，舒適地持續同一母音並固定裝置、距離與音量；不要為了壓低數字而僵硬發聲。"],
      shimmer: ["Shimmer 描述相鄰可用週期的振幅變動；較適合同條件的穩定持續母音比較。", "本次為 {{value}}；{{status}}。麥克風增益、距離與音量變化都會改變結果。", "用舒適音量持續同一母音並固定距離再比較；不要刻意擠壓或放大聲音追數字。"],
      hnr: ["HNR 比較週期性諧波與雜訊的相對強度，用來描述這次訊號的清晰／含噪程度。", "本次為 {{value}}；{{status}}。背景噪音、氣聲風格與麥克風都可能改變它。", "先在安靜處固定距離重錄；若仍要比較，再用同一句各錄自然版與稍清楚版，以回聽為準。"],
      cpp: ["CPP 描述語音中週期結構的突出程度，是聲音規律性的聲學證據之一。", "本次為 {{value}}；{{status}}。它會受句子、音量、噪音與錄音設備影響，不能診斷聲帶狀況。", "固定句子、音量、裝置與距離重錄；比較聲音是否清楚又舒服，不必把 CPP 當成越高越好的目標。"],
    },
  },
  "zh-Hans": {
    components: {
      model: ["显示模型从整段录音识别到的声音呈现特征，是综合分数的一个来源。", "本次模型分项为 {{value}}%；套用 {{weight}}% 权重后参与总分。它不是性别或健康判定。", "先回听自然度与舒适度；比较时固定句子与录音条件，保留最符合自己目标的版本。"],
      resonance: ["汇总低、中、高频段的相对能量，作为本次音色平衡的声学证据。", "本次共鸣证据分项为 {{value}}%；套用 {{weight}}% 权重后参与总分。", "想更明亮可稍微说清楚辅音与元音；想更温暖可改用柔和语气，维持自然音量后回听比较。"],
      pitch: ["比较这次有声片段的音高中心与应用模型，是综合分数中的其中一项。", "本次音高分项为 {{value}}%；套用 {{weight}}% 权重后参与总分。单一分项不代表整体声音。", "用同一句录一个自然版与小幅升高或降低版，回听哪个更自然、舒服且符合目标。"],
      intonation: ["汇总句中音高走向与变化幅度，描述这次语调的动态表现。", "本次语调分项为 {{value}}%；套用 {{weight}}% 权重后参与总分。", "把同一句用陈述、疑问或不同情绪各说一次，选择最能传达意思且不费力的版本。"],
    },
    voiceQuality: {
      jitter: ["Jitter 描述相邻可用周期的频率变化；在稳定持续元音且录音质量足够时更适合比较。", "本次为 {{value}}；{{status}}。日常语句、噪声与周期不足都会影响数字。", "若要比较，舒适地持续同一元音并固定设备、距离与音量；不要为了压低数字而僵硬发声。"],
      shimmer: ["Shimmer 描述相邻可用周期的振幅变化；更适合同条件的稳定持续元音比较。", "本次为 {{value}}；{{status}}。麦克风增益、距离与音量变化都会改变结果。", "用舒适音量持续同一元音并固定距离再比较；不要刻意挤压或放大声音追数字。"],
      hnr: ["HNR 比较周期性谐波与噪声的相对强度，用来描述这次信号的清晰／含噪程度。", "本次为 {{value}}；{{status}}。背景噪声、气声风格与麦克风都可能改变它。", "先在安静处固定距离重录；若仍要比较，再用同一句各录自然版与稍清楚版，以回听为准。"],
      cpp: ["CPP 描述语音中周期结构的突出程度，是声音规律性的声学证据之一。", "本次为 {{value}}；{{status}}。它会受句子、音量、噪声与设备影响，不能诊断声带状况。", "固定句子、音量、设备与距离重录；比较声音是否清楚又舒服，不必把 CPP 当成越高越好的目标。"],
    },
  },
  en: {
    components: {
      model: ["Shows presentation features the model recognized across the recording and supplies one part of the combined score.", "This take's model component is {{value}}%; it contributes after a {{weight}}% weighting. It is not a gender or health determination.", "Replay for naturalness and comfort. For comparisons, keep the line and recording conditions fixed and retain the take that fits your goal."],
      resonance: ["Summarizes relative low-, mid-, and high-band energy as acoustic evidence about this take's tonal balance.", "This take's resonance-evidence component is {{value}}%; it contributes after a {{weight}}% weighting.", "For more brightness, try slightly clearer consonants and vowels; for more warmth, try a gentler delivery. Keep natural volume and replay both."],
      pitch: ["Compares the pitch center of voiced frames with the app model as one part of the combined score.", "This take's pitch component is {{value}}%; it contributes after a {{weight}}% weighting. One component does not describe the whole voice.", "Record the same line naturally and with a small comfortable pitch change, then replay which version feels natural and fits your goal."],
      intonation: ["Combines pitch direction and movement across the line to describe this take's intonation dynamics.", "This take's intonation component is {{value}}%; it contributes after a {{weight}}% weighting.", "Say the same line as a statement, question, or with different emotion, then choose the version that communicates well without effort."],
    },
    voiceQuality: {
      jitter: ["Jitter describes frequency variation between usable adjacent periods; it is most comparable in a stable sustained vowel with adequate recording quality.", "This take is {{value}}; {{status}}. Connected speech, noise, and too few periods can change the number.", "For comparison, sustain the same vowel comfortably with matched device, distance, and volume; do not stiffen the voice to lower the number."],
      shimmer: ["Shimmer describes amplitude variation between usable adjacent periods and is most comparable across stable sustained vowels in matched conditions.", "This take is {{value}}; {{status}}. Microphone gain, distance, and volume changes can alter it.", "Sustain the same vowel at comfortable volume and matched distance; do not squeeze or amplify the voice to chase the number."],
      hnr: ["HNR compares periodic harmonic energy with noise and describes how clear or noise-influenced this recorded signal is.", "This take is {{value}}; {{status}}. Background noise, an airy style, and the microphone can all alter it.", "First re-record in a quiet place at matched distance; then compare a natural and slightly clearer version of the same line by replay."],
      cpp: ["CPP describes how prominent periodic organization is in speech and supplies one piece of acoustic evidence about regularity.", "This take is {{value}}; {{status}}. The line, volume, noise, and device affect it, and it cannot diagnose vocal-fold health.", "Match the line, volume, device, and distance; compare clarity and comfort rather than treating higher CPP as an automatic goal."],
    },
  },
  ja: {
    components: {
      model: ["録音全体からモデルが認識した声の提示特徴を示し、総合スコアの一部になります。", "今回のモデル項目は{{value}}%、{{weight}}%の重みで総合値に加わります。性別や健康の判定ではありません。", "自然さと快適さを再生確認します。比較時は文と録音条件をそろえ、自分の目標に合うテイクを残します。"],
      resonance: ["低・中・高域の相対エネルギーをまとめ、今回の音色バランスの音響証拠として使います。", "今回の共鳴証拠項目は{{value}}%、{{weight}}%の重みで総合値に加わります。", "明るさには子音と母音を少し明瞭に、温かさには穏やかな話し方を試し、自然な音量で再生比較します。"],
      pitch: ["有声音フレームのピッチ中心をアプリモデルと比較し、総合スコアの一部にします。", "今回のピッチ項目は{{value}}%、{{weight}}%の重みで総合値に加わります。一項目だけで声全体は決まりません。", "同じ文を自然版と無理のない小さな高低変化版で録音し、自然で快適かつ目標に合う方を再生で選びます。"],
      intonation: ["文中のピッチ方向と変化幅をまとめ、今回のイントネーション動態を記述します。", "今回のイントネーション項目は{{value}}%、{{weight}}%の重みで総合値に加わります。", "同じ文を陳述・疑問・別の感情で言い、意味が伝わり無理のない版を選びます。"],
    },
    voiceQuality: {
      jitter: ["Jitterは利用可能な隣接周期の周波数変動を示し、品質十分な安定持続母音で比較しやすい指標です。", "今回は{{value}}；{{status}}。連続発話、雑音、周期不足で数値は変わります。", "同じ母音を快適に持続し、端末・距離・音量をそろえて比較します。数値を下げるために声を固めないでください。"],
      shimmer: ["Shimmerは利用可能な隣接周期の振幅変動を示し、同条件の安定持続母音で比較しやすい指標です。", "今回は{{value}}；{{status}}。マイク利得、距離、音量変化で結果は変わります。", "快適な音量で同じ母音を持続し距離をそろえます。数値のために声を圧迫したり大きくしないでください。"],
      hnr: ["HNRは周期的な調波と雑音の相対強度を比べ、今回の信号の明瞭さ／雑音影響を記述します。", "今回は{{value}}；{{status}}。背景雑音、空気感のあるスタイル、マイクで変化します。", "まず静かな場所と同じ距離で再録音し、自然版と少し明瞭な版を再生して比べます。"],
      cpp: ["CPPは音声中の周期構造の目立ち方を示し、規則性に関する音響証拠の一つです。", "今回は{{value}}；{{status}}。文、音量、雑音、端末で変わり、声帯の健康を診断できません。", "文・音量・端末・距離をそろえ、CPPの高さではなく明瞭さと快適さを比較します。"],
    },
  },
};

function guidance(locale, purpose, result, next) {
  const labels = (GUIDANCE[locale] || GUIDANCE.en).labels;
  const separator = locale === "en" ? ": " : "：";
  return `${labels[0]}${separator}${purpose}\n${labels[1]}${separator}${result}\n${labels[2]}${separator}${next}`;
}

function guidedStates(locale, metric, next, labels = {}) {
  return Object.fromEntries(Object.entries(metric.result).map(([state, result]) => [state, {
    ...(labels[state] ? { label: labels[state] } : {}),
    hint: guidance(locale, metric.purpose, result, state === "insufficient" ? (GUIDANCE[locale] || GUIDANCE.en).retry : (metric.next || next)),
  }]));
}

export function buildAnalysisGuidance(locale, ui) {
  const g = GUIDANCE[locale] || GUIDANCE.en;
  const extra = EXTRA_GUIDANCE[locale] || EXTRA_GUIDANCE.en;
  const componentGuidance = Object.fromEntries(Object.entries(extra.components).map(([key, parts]) => [key, guidance(locale, ...parts)]));
  const voiceQualityGuidance = Object.fromEntries(Object.entries(extra.voiceQuality).map(([key, parts]) => [key, guidance(locale, ...parts)]));
  const formantGuidance = Object.fromEntries(Object.entries(g.formants).map(([key, metric]) => [key, {
    insufficient: guidance(locale, metric.purpose, g.insufficient, g.retry),
    low: guidance(locale, metric.purpose, metric.low, metric.next || g.compare),
    inRange: guidance(locale, metric.purpose, metric.inRange, metric.next || g.compare),
    high: guidance(locale, metric.purpose, metric.high, metric.next || g.compare),
  }]));
  const insight = Object.fromEntries(Object.entries(g.insights).map(([key, parts]) => [key, guidance(locale, ...parts)]));
  const compact = (purpose, result, next = g.compare) => guidance(locale, purpose, result, next);

  return {
    experiment: {
      advanced: {
        pitchMedianHint: guidance(locale, g.pitch.purpose, g.pitch.single, g.pitch.next),
        insight,
        components: { guidance: componentGuidance },
        voiceAgeV2: { metricGuidance: voiceQualityGuidance },
      },
      quick: {
        reveal: {
          pitchSingleHint: guidance(locale, g.pitch.purpose, g.pitch.single, g.pitch.next),
          pitchStandardHint: guidance(locale, g.pitch.purpose, g.pitch.standard, g.pitch.next),
        },
      },
    },
    analysis: {
      meter: { hint: guidance(locale, g.model.purpose, g.model.result, g.model.next) },
      resonanceBalance: guidedStates(locale, g.metrics.resonance, g.compare, ui.resonanceLabels),
      tilt: guidedStates(locale, g.metrics.tilt, g.compare),
      breathiness: guidedStates(locale, g.metrics.breathiness, g.compare),
      brightness: guidedStates(locale, g.metrics.brightness, g.compare),
      formant: {
        guidance: formantGuidance,
        moreSamplesHint: locale === "en"
          ? "The usable sample count is low, so treat this result as provisional."
          : locale === "ja"
            ? "利用可能なサンプルが少ないため、今回は暫定的な結果です。"
            : locale === "zh-Hans"
              ? "可用样本较少，因此本次结果仅作初步参考。"
              : "可用樣本較少，因此本次結果僅作初步參考。",
      },
      vowelFocus: guidedStates(locale, g.metrics.vowel, g.compare, ui.vowelLabels),
      speechRate: guidedStates(locale, g.metrics.speech, g.compare),
      liaison: guidedStates(locale, g.metrics.liaison, g.compare),
      intonation: {
        insufficient: {
          slopeHint: guidance(locale, g.metrics.slope.purpose, g.insufficient, g.retry),
          rangeHint: guidance(locale, g.metrics.range.purpose, g.insufficient, g.retry),
        },
        slope: guidedStates(locale, g.metrics.slope, g.compare),
        range: guidedStates(locale, g.metrics.range, g.compare),
      },
    },
    summary: {
      beginnerHighlights: {
        items: {
          pitch: { tip: compact(g.pitch.purpose, locale === "en" ? "This card shows the pitch band detected for this take." : locale === "ja" ? "このカードは今回検出されたピッチ帯を示します。" : locale === "zh-Hans" ? "这张卡显示本次检测到的音高带。" : "這張卡顯示本次偵測到的音高帶。", g.pitch.next) },
          resonance: { tip: compact(g.metrics.resonance.purpose, locale === "en" ? "The category above summarizes this take's low/mid/high-band balance." : locale === "ja" ? "上の分類は今回の低・中・高域バランスをまとめたものです。" : locale === "zh-Hans" ? "上方分类汇总了本次低、中、高频段的平衡。" : "上方分類彙整了本次低、中、高頻段的平衡。", g.metrics.resonance.next) },
          speech: { tip: compact(g.metrics.speech.purpose, locale === "en" ? "The category above summarizes the measured pace of this take." : locale === "ja" ? "上の分類は今回測定された話速をまとめたものです。" : locale === "zh-Hans" ? "上方分类汇总了本次测得的说话速度。" : "上方分類彙整了本次測得的說話速度。", g.metrics.speech.next) },
        },
      },
      focus: {
        items: {
          divergence: compact(locale === "en" ? "Compares pitch-band and model directions." : locale === "ja" ? "ピッチ帯とモデル方向を比較します。" : locale === "zh-Hans" ? "比较音高带与模型方向。" : "比較音高帶與模型方向。", locale === "en" ? "Pitch is {{band}} while the model trends {{trend}}, so the two cues differ in this take." : locale === "ja" ? "ピッチは{{band}}、モデルは{{trend}}傾向で、今回は2つの手掛かりが異なります。" : locale === "zh-Hans" ? "音高位于{{band}}，模型倾向{{trend}}，这次两种线索并不一致。" : "音高位於{{band}}，模型傾向{{trend}}，這次兩種線索並不一致。"),
          noisy: compact(locale === "en" ? "Checks whether background noise is strong enough to affect analysis." : locale === "ja" ? "背景雑音が分析に影響する強さか確認します。" : locale === "zh-Hans" ? "检查背景噪声是否足以影响分析。" : "檢查背景噪音是否足以影響分析。", locale === "en" ? "Noise is {{snrLabel}} ({{snrDisplay}}), so acoustic details may be less stable." : locale === "ja" ? "雑音は{{snrLabel}}（{{snrDisplay}}）で、音響細部が不安定になる可能性があります。" : locale === "zh-Hans" ? "噪声为{{snrLabel}}（{{snrDisplay}}），声学细节可能较不稳定。" : "噪音為{{snrLabel}}（{{snrDisplay}}），聲學細節可能較不穩定。", g.retry),
          pitchWide: compact(g.pitch.purpose, locale === "en" ? "The detected pitch spread is about {{spread}} Hz ({{stability}}), showing larger variation in this take." : locale === "ja" ? "検出ピッチ幅は約{{spread}} Hz（{{stability}}）で、今回は変化が大きめです。" : locale === "zh-Hans" ? "检测到的音高跨度约 {{spread}} Hz（{{stability}}），这次变化较大。" : "偵測到的音高跨度約 {{spread}} Hz（{{stability}}），這次變化較大。", g.pitch.next),
          pitchModerate: compact(g.pitch.purpose, locale === "en" ? "The detected pitch spread is about {{spread}} Hz ({{stability}}), showing moderate variation in this take." : locale === "ja" ? "検出ピッチ幅は約{{spread}} Hz（{{stability}}）で、今回は中程度の変化です。" : locale === "zh-Hans" ? "检测到的音高跨度约 {{spread}} Hz（{{stability}}），这次变化适中。" : "偵測到的音高跨度約 {{spread}} Hz（{{stability}}），這次變化適中。", g.pitch.next),
          breathinessAiry: compact(g.metrics.breathiness.purpose, locale === "en" ? "The proxy is {{label}}, indicating more airiness in this take." : locale === "ja" ? "代理値は「{{label}}」で、今回は空気感が多めです。" : locale === "zh-Hans" ? "代理值为“{{label}}”，这次空气感较多。" : "代理值為「{{label}}」，這次空氣感較多。", g.metrics.breathiness.next),
          breathinessDense: compact(g.metrics.breathiness.purpose, locale === "en" ? "The proxy is {{label}}, indicating less airiness in this take." : locale === "ja" ? "代理値は「{{label}}」で、今回は空気感が少なめです。" : locale === "zh-Hans" ? "代理值为“{{label}}”，这次空气感较少。" : "代理值為「{{label}}」，這次空氣感較少。", g.metrics.breathiness.next),
          vowelWeak: compact(g.metrics.vowel.purpose, locale === "en" ? "The result is {{label}}, meaning fewer usable frames fall inside the app reference band." : locale === "ja" ? "結果は「{{label}}」で、参考帯内の利用可能フレームが少なめです。" : locale === "zh-Hans" ? "结果为“{{label}}”，落在应用参考带内的可用帧较少。" : "結果為「{{label}}」，落在應用參考帶內的可用幀較少。", g.metrics.vowel.next),
          speechFast: compact(g.metrics.speech.purpose, locale === "en" ? "The measured pace is {{label}}, so this take is faster." : locale === "ja" ? "測定話速は「{{label}}」で、今回は速めです。" : locale === "zh-Hans" ? "测得语速为“{{label}}”，这次节奏较快。" : "測得語速為「{{label}}」，這次節奏較快。", g.metrics.speech.next),
          speechSlow: compact(g.metrics.speech.purpose, locale === "en" ? "The measured pace is {{label}}, so this take is slower." : locale === "ja" ? "測定話速は「{{label}}」で、今回は遅めです。" : locale === "zh-Hans" ? "测得语速为“{{label}}”，这次节奏较慢。" : "測得語速為「{{label}}」，這次節奏較慢。", g.metrics.speech.next),
          voicedLow: compact(locale === "en" ? "Checks how much usable voiced material supports the statistics." : locale === "ja" ? "統計を支える利用可能な有声音量を確認します。" : locale === "zh-Hans" ? "检查有多少可用有声素材支持统计。" : "檢查有多少可用有聲素材支持統計。", "{{label}}", g.retry),
          brightnessSharp: compact(g.metrics.brightness.purpose, locale === "en" ? "The category is {{label}}, meaning high-frequency evidence is strong in this take." : locale === "ja" ? "分類は「{{label}}」で、今回は高域の証拠が強めです。" : locale === "zh-Hans" ? "分类为“{{label}}”，这次高频信号较强。" : "分類為「{{label}}」，這次高頻訊號較強。", g.metrics.brightness.next),
        },
      },
    },
  };
}

export { GUIDANCE as ANALYSIS_GUIDANCE_TEXT };
