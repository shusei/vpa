#!/usr/bin/env node
import fs from "fs";
import path from "path";
import process from "process";

import {
  PS_INTERVAL_MS,
  DEFAULT_PITCH_RANGE,
  clampPitchRange,
  createPitchPostState,
  resetPitchPostState,
  appendPitchSample,
  makeNoiseTracker,
  filterPitchForStats,
  makeStats,
  computeIntonationMetrics,
  CONFIDENCE_INCLUDE_THRESHOLD,
  CONFIDENCE_VOICED_THRESHOLD,
  fmt1,
  logPostProcessingDiagnostics,
} from "../assets/js/pitch-shared.js";

function readInput(argv){
  const arg = argv[0];
  if (arg && arg !== "-"){
    const resolved = path.resolve(process.cwd(), arg);
    return fs.readFileSync(resolved, "utf8");
  }
  const stat = fs.fstatSync(process.stdin.fd);
  if (!stat.isFIFO() && !stat.isFile()){
    throw new Error("請提供 JSON 檔案路徑或透過 STDIN 傳入分析結果。");
  }
  return fs.readFileSync(0, "utf8");
}

function findOfflineSamples(node, depth = 0, seen = new Set()){ 
  if (!node || typeof node !== "object" || seen.has(node)) return null;
  seen.add(node);
  if (Array.isArray(node.pitchRaw) && Array.isArray(node.pitchProcessed) && Array.isArray(node.db)){
    return node;
  }
  if (node.offlineSamples){
    const found = findOfflineSamples(node.offlineSamples, depth + 1, seen);
    if (found) return found;
  }
  for (const value of Object.values(node)){
    if (value && typeof value === "object"){
      const found = findOfflineSamples(value, depth + 1, seen);
      if (found) return found;
    }
  }
  return null;
}

function buildSpectral(breathiness, zcr, energy){
  const hasBreath = Number.isFinite(breathiness);
  const hasZcr = Number.isFinite(zcr);
  const hasEnergy = Array.isArray(energy);
  if (!hasBreath && !hasZcr && !hasEnergy) return null;
  const spectral = {};
  if (hasBreath) spectral.breathiness = breathiness;
  if (hasZcr) spectral.zcr = zcr;
  if (hasEnergy){
    spectral.energy = {
      low: Number.isFinite(energy[0]) ? energy[0] : NaN,
      mid: Number.isFinite(energy[1]) ? energy[1] : NaN,
      high: Number.isFinite(energy[2]) ? energy[2] : NaN,
    };
  }
  return spectral;
}

function main(){
  let rawText;
  try{
    rawText = readInput(process.argv.slice(2));
  }catch(err){
    console.error(err.message || err);
    process.exit(1);
  }
  if (!rawText || !rawText.trim()){
    console.error("未取得有效 JSON 內容。");
    process.exit(1);
  }

  let data;
  try{
    data = JSON.parse(rawText);
  }catch(err){
    console.error("解析 JSON 失敗：", err.message || err);
    process.exit(1);
  }

  const offline = findOfflineSamples(data);
  if (!offline){
    console.error("找不到離線分析樣本，請確認輸入格式包含 offlineSamples。");
    process.exit(1);
  }

  const frameSec = Number.isFinite(offline.frameSec) ? offline.frameSec : (PS_INTERVAL_MS / 1000);
  const dbSeries = Array.isArray(offline.db) ? offline.db : [];
  const rawPitch = Array.isArray(offline.pitchRaw) ? offline.pitchRaw : [];
  const breathiness = Array.isArray(offline.breathiness) ? offline.breathiness : [];
  const zcr = Array.isArray(offline.zcr) ? offline.zcr : [];
  const energy = Array.isArray(offline.energy) ? offline.energy : [];

  const total = Math.max(dbSeries.length, rawPitch.length, breathiness.length, zcr.length, energy.length);
  if (!total){
    console.error("離線樣本為空，無法驗證。");
    process.exit(1);
  }

  const rangeConfig = data?.pitch?.detectorRange || data?.pitch?.range;
  const pitchRange = clampPitchRange(rangeConfig || DEFAULT_PITCH_RANGE);

  const state = createPitchPostState();
  resetPitchPostState(state);
  const arrays = { raw: [], smooth: [], voiced: [], confidence: [] };
  const tracker = makeNoiseTracker();

  let lastVoiced = false;
  for (let i=0;i<total;i++){
    const db = dbSeries[i] ?? NaN;
    const gate = tracker.shouldDetect(db, lastVoiced);
    let candHz = null;
    if (gate.detect){
      const raw = rawPitch[i];
      if (Number.isFinite(raw)){
        candHz = raw;
      } else {
        tracker.capture(db);
      }
    } else {
      tracker.capture(db);
    }
    const spectral = buildSpectral(breathiness[i], zcr[i], energy[i]);
    const { voiced } = appendPitchSample(candHz, {
      db,
      ambientDb: gate.ambient ?? NaN,
      spectral,
    }, {
      state,
      arrays,
      range: pitchRange,
    });
    lastVoiced = Boolean(voiced);
  }

  const filtered = [];
  for (let i=0;i<arrays.smooth.length;i++){
    const hz = arrays.smooth[i];
    const conf = arrays.confidence[i] ?? 0;
    if (Number.isFinite(hz) && conf >= CONFIDENCE_INCLUDE_THRESHOLD){
      filtered.push(hz);
    }
  }
  const stable = filterPitchForStats(filtered);
  const statsSamples = stable.length ? stable : filtered;
  const stats = makeStats(statsSamples);
  const spread = Number.isFinite(stats.p95) && Number.isFinite(stats.p05)
    ? (stats.p95 - stats.p05)
    : NaN;

  const intonation = computeIntonationMetrics({
    processed: arrays.smooth,
    raw: arrays.raw,
    confidence: arrays.confidence,
    voiced: arrays.voiced,
  }, frameSec, {
    confidenceThreshold: CONFIDENCE_INCLUDE_THRESHOLD,
    voicedThreshold: CONFIDENCE_VOICED_THRESHOLD,
  });
  const dynamicRange = Number.isFinite(intonation?.range) ? intonation.range : NaN;

  console.info(`[音高後處理] High/Low（統計表對照）：${fmt1(stats.p95)} Hz / ${fmt1(stats.p05)} Hz`);
  logPostProcessingDiagnostics(state, {
    spread,
    intonationRange: dynamicRange,
  });
}

main();
