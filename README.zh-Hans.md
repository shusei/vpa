# Voice Presentation Analyzer (VPA)

> **Languages:** [English](./README.md) | [繁體中文](./README.zh-Hant.md) | [簡體中文](./README.zh-Hans.md) | [日本語](./README.ja.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![WebGPU Accelerated](https://img.shields.io/badge/WebGPU-Accelerated-blue.svg)](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)
[![Privacy First](https://img.shields.io/badge/Privacy-100%25%20Local-brightgreen.svg)](#)
[![GitHub Release](https://img.shields.io/github/v/release/shusei/vpa.svg?color=orange)](https://github.com/shusei/vpa/releases)

---

> 100% 浏览器端推理的声音呈现分析与练习工具，从录音或上传的语音片段中推估女性化 (Feminine) / 男性化 (Masculine) 呈现倾向，提供快速挑战测试、句库练习、实时声学监控、动态卡片分享与进阶统计摘要。

- 🌐 **Demo (GitHub Pages)**：https://shusei.github.io/vpa
- 📋 **更新日志 (Changelog)**：[CHANGELOG.md](./CHANGELOG.md)
- 🔬 **算法验证报告**：[ALGORITHM_VERIFICATION.md](./ALGORITHM_VERIFICATION.md)
- 🔒 **隐私防护**：音频与模型推理 100% 于本地浏览器完成，音档绝不上传任何服务器。
- 🌍 **多语系支持**：繁体中文 / 简体中文 / English / 日本语，自动记忆默认偏好。

---

## 目录

- [产品特色与双模式体验](#产品特色与双模式体验)
  - [1. 快速测试模式 (Quick Test Experience)](#1-快速测试模式-quick-test-experience)
  - [2. 进阶专业模式 (Professional Experience)](#2-进阶专业模式-professional-experience)
  - [3. 句库练习抽屉 (Practice Drawer)](#3-句库练习抽屉-practice-drawer)
  - [4. 社区分享与动态卡片 (Social Sharing)](#4-社区分享与动态卡片-social-sharing)
- [核心能力一览](#核心能力一览)
- [操作流程](#操作流程)
- [输出解读指南](#输出解读指南)
- [技术架构](#技术架构)
- [开发与测试](#开发与测试)
- [常见问题与疑难排解](#常见问题与疑难排解)
- [隐私、定位与免责声明](#隐私定位与免责声明)
- [版本信息与授权](#版本信息与授权)

---

## 产品特色与双模式体验

Voice Presentation Analyzer（VPA）支持双重界面体验，无论是想快速测量单句声线，或是进行专业声学指标分析，都能一键切换：

### 1. 快速测试模式 (Quick Test Experience)
- **每日一练 (Daily Test)**：精选经典测试句，几秒钟即可完成一次语音检测。
- **标准测试 (Standard Test)**：连续 3 句测试挑战，全面评估跨语句的声音稳定性与一致性。
- **直观评分卡与实时重播**：显示声音倾向百分比、估计声龄与声线原型；结果页上的播放按钮支持 **播放/暂停 (Play/Pause)** 切换与重放。

### 2. 进阶专业模式 (Professional Experience)
- **实时 Pitch Stream 走势**：50–450 Hz 声线实时追踪、瞬时音量与底噪监控。
- **Formant / Resonance 共鸣面板**：估计 F1–F3 共鸣峰、胸腔 / 前置 / 头腔共鸣比例与气声比例。
- **语调与进阶统计卡**：提供洋红语调曲线图、语速、连音比例、共鸣亮度评估与个性化建议简评。

### 3. 句库练习抽屉 (Practice Drawer)
- **Core-36 经典句库**：内置分类丰富的训练句型，支持单句快捷录音、回听、历程成绩纪录与自动比对。

### 4. 社区分享与动态卡片 (Social Sharing)
- **动态成果卡片 (PNG / Video)**：一键绘制包含声线倾向、音高 Hz 与评语的动态视觉卡片。
- **社区快捷传送**：自动整合短网址与图片视频，支持 X (Twitter)、Threads、LINE 免费短链接一键分享。

---

## 核心能力一览

### 隐私与安全

- **本地推理**：采用 [`@xenova/transformers`](https://github.com/xenova/transformers.js) 的 ONNX Runtime，模型自 Hugging Face Hub 下载后完全于本地执行，音档绝不离身。
- **离线支持**：首次下载后模型缓存于 IndexedDB，日后造访即可 100% 离线运作。
- **最小化预处理**：仅将输入混成单声道并重采样至 16 kHz，完整保留语音品质与音量细节。

### 互动与主题

- **多主题系统**：内置 30+ 派别与 Lux 系列主题，支持系统深浅色模式自动跟随与手动切换。
- **录音与文件上传二合一**：支持 MediaRecorder 实时录音，或直接上传 `mp3 / m4a / mp4 / mov / wav` 格式文件（可自动从视频抽取音轨），亦支持拖拽文件上传。
- **键盘快捷与无障碍 (ARIA)**：支持 <kbd>Space</kbd> 快捷录音、全界面 `aria-*` 属性标记与键盘友善操作。

---

## 操作流程

1. **开启网页**：造访 <https://shusei.github.io/vpa>。首次造访时请联网下载模型。
2. **选择测试体验**：
   - **快速测试**：点击“开始测试”，跟随画面提示朗读句子，完成后即可获得倾向评分、声龄与原型，并可随时点按播放按钮进行“播放 / 暂停”回听。
   - **进阶专业模式**：切换至专业模式，随时进行 5-10 秒自由发声录音或上传音频档，检视 Pitch 走势、共鸣面板与详细统计数据。
3. **句库练习**：点击录音键旁“句库练习”开启抽屉，挑选 Core-36 句子进行单句反复训练与成绩追踪。
4. **社区分享**：在测试结果页点击“分享结果”，生成专属视觉卡片或复制短链接至 X、Threads 或 LINE。

---

## 输出解读指南

- **倾向仪表**：显示模型推估的 feminine / masculine 百分比。40–60% 为灰色过渡带，建议多录几段观察声音趋势。
- **Pitch 音高卡**：包含平均音高 (Hz)、中位数与 5th / 95th 百分位数，标记常见声域范围 (50–600 Hz)。
- **Formant & Resonance 共鸣卡**：展示 F1–F3 中位数、胸腔 / 前置 / 头腔共鸣比例与气声比例。
- **语调曲线与进阶摘要**：洋红线描绘语调走势，灰色区段标示静音或低信心区，并标注连音比例与说话语速。

---

## 技术架构

```
使用者行为 → MediaRecorder / 文件上传 → 音频解码 (WebAudio / FFmpeg)
  ↘ WebAudio 实时分析 (Pitch / Formant / Noise)
     ↘ IndexedDB 缓存模型 → Transformers.js ONNX Runtime (WebGPU / WASM)
        ↘ 整段 / 串流分段 → Log-odds 聚合 → 倾向评估
           ↘ 统计汇总 (百分位、语速、共鸣、语调曲线)
              ↘ 快速测试 / 专业模式 UI → 社区分享卡片生成
```

- **推理引擎**：`@xenova/transformers` 搭配 `prithivMLmods/Common-Voice-Gender-Detection-ONNX` 模型。
- **分段策略**：自动根据装置硬件（WebGPU / WASM）与时长选择最佳串流窗口与 hop。

---

## 开发与测试

### 前置需求
- Node.js 18+
- npm 8+

### 执行测试
```bash
npm test      # 执行单元测试、语法检查、HTML/CSS 标记检查与社区分享验证
npm run test:e2e # 执行全套 Playwright 端到端浏览器测试
```

---

## 常见问题与疑难排解

| 问题 | 可能原因 | 建议作法 |
| ---- | -------- | -------- |
| 无法录音 | 浏览器未授予麦克风权限 | 请在浏览器网址列左侧允许麦克风权限，或使用文件上传功能。 |
| 快速测试重听 | 播放中再次点击按钮 | 播放按钮具备“播放/暂停”切换功能，播放中按下可直接暂停。 |
| 模型下载缓慢 | 首次造访时网络较慢 | 重整页面或等待下载完成，完成后模型会自动缓存于 IndexedDB 供离线使用。 |

---

## 隐私、定位与免责声明

- **非医疗诊断工具**：本工具指标系结合声学算法与机器学习模型，仅供自我声音探索、训练与语音呈现反馈，**不等同于性别认同、身份判定或任何医疗诊断／法律结论**。
- **发声安全提醒**：练习时请保持自然舒服的音量。若喉咙感觉沙哑、疲劳或不适，请立即停止休息并咨询专业语言治疗师或耳鼻喉科医师。
- **隐私承诺**：所有录音与声音分析 100% 于您的个人装置内完成，资料绝不上传任何服务器。

---

## 版本信息与授权

- **更新细节**：完整版本历史请参阅 [CHANGELOG.md](./CHANGELOG.md)。
- **项目授权**：MIT License。
- **模型授权**：Apache-2.0 License (`prithivMLmods/Common-Voice-Gender-Detection-ONNX`)。
- **句库资料**：Core-36 句库以 CC0-1.0 释出。

---

## 支援作者

[![Buy Me a Coffee](https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png)](https://buymeacoffee.com/shusei)
