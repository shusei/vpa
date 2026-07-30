export function createPitchStrategyController(deps) {
  const {
    log,
    pitchStrategies,
    trimYinBuffers,
  } = deps;

  const PITCH_RUNTIME_BASE_BUDGET_MS = 26;
  const PITCH_RUNTIME_OVER_BUDGET_LIMIT = 3;
  const PITCH_RUNTIME_RECOVERY_MS = 1500;
  const PITCH_RUNTIME_OFFLINE_MULTIPLIER = 1.65;
  const PITCH_RUNTIME_MIN_BUDGET_MS = 18;
  const PITCH_RUNTIME_MAX_BUDGET_MS = 60;
  const PITCH_RETRY_MIN_INTERVAL_MS = 3000;
  const PITCH_RETRY_COOLDOWN_MS = 20000;
  const PITCH_RETRY_ERROR_COOLDOWN_MS = 45000;
  const PITCH_RETRY_ERROR_GUARD_MS = 6500;
  const PITCH_RETRY_ERROR_ATTEMPT_SPACING_MS = 8000;

  const pitchStrategyState = {
    activeKey: null,
    lockUntil: 0,
    lockReason: null,
    lockReasonDetail: null,
    overBudgetStreak: 0,
    lastSwitch: 0,
    lastOverBudget: 0,
    runtimeEwma: 0,
    lastEnableAttempt: 0,
    lockedAt: 0,
    lockDuration: 0,
    lockContext: null,
    autoRetryUntil: 0,
  };

  let trimPitchBuffersTimer = null;
  let pitchAutoRetryTimer = null;
  const pitchRetryTimers = new Set();

  function registerPitchRetryTimer(id) {
    if (id == null) return null;
    pitchRetryTimers.add(id);
    pitchAutoRetryTimer = id;
    return id;
  }

  function releasePitchRetryTimer(id) {
    if (id == null) return;
    if (pitchRetryTimers.has(id)) {
      pitchRetryTimers.delete(id);
    }
    if (pitchAutoRetryTimer === id) {
      pitchAutoRetryTimer = null;
    }
  }

  function cancelPitchAutoRetryTimers() {
    if (!pitchRetryTimers.size) return;
    const clearFn = typeof clearTimeout === "function" ? clearTimeout : null;
    for (const handle of pitchRetryTimers) {
      if (clearFn) clearFn(handle);
    }
    pitchRetryTimers.clear();
    pitchAutoRetryTimer = null;
  }

  function nowMs() {
    try {
      if (typeof performance !== "undefined" && performance?.now) {
        return performance.now();
      }
    } catch { }
    return Date.now();
  }

  function initializePitchStrategy() {
    try {
      const preferred = selectPreferredPitchStrategy("initial");
      switchPitchStrategy(preferred, "initial");
    } catch (err) {
      console.warn("[pitch] initialize failed", err);
      switchPitchStrategy(pitchStrategies.acf, "initial-fallback");
    }
  }

  function switchPitchStrategy(strategy, reason) {
    const next = strategy || pitchStrategies.acf;
    if (pitchStrategyState.activeKey === next.key) return;
    pitchStrategyState.activeKey = next.key;
    pitchStrategyState.overBudgetStreak = 0;
    pitchStrategyState.lastSwitch = nowMs();
    pitchStrategyState.lastOverBudget = 0;
    pitchStrategyState.runtimeEwma = 0;
    if (next.key === "yin") {
      cancelPitchBufferTrimTimer();
      cancelPitchAutoRetryTimers();
    }
    if (reason) {
      log(`[pitch] strategy -> ${next.label} (${next.key}) via ${reason}`);
    } else {
      log(`[pitch] strategy -> ${next.label} (${next.key})`);
    }
  }

  function getActivePitchStrategy() {
    return pitchStrategies[pitchStrategyState.activeKey] || pitchStrategies.acf;
  }

  function maybeEnableAdvancedPitch(context, { allowRetry = false, force = false } = {}) {
    const active = getActivePitchStrategy();
    if (!force && active.key !== "acf") return;

    const now = nowMs();
    if (!force) {
      if (pitchStrategyState.lockUntil && now < pitchStrategyState.lockUntil) {
        if (!allowRetry) return;
        const reason = pitchStrategyState.lockReason;
        const lockDuration = pitchStrategyState.lockDuration || PITCH_RETRY_COOLDOWN_MS;
        const elapsed = Math.max(0, now - (pitchStrategyState.lockedAt || 0));
        const ratio = lockDuration > 0 ? elapsed / lockDuration : 1;
        const offlineGrace = context === "offline" && reason === "runtime" && ratio >= 0.35;
        const runtimeGrace = reason === "runtime" && ratio >= 0.5;
        if (!(offlineGrace || runtimeGrace)) return;
      }
      const minInterval = allowRetry ? PITCH_RETRY_MIN_INTERVAL_MS / 2 : PITCH_RETRY_MIN_INTERVAL_MS;
      if (now - pitchStrategyState.lastEnableAttempt < minInterval) return;
    }

    pitchStrategyState.lastEnableAttempt = now;
    const preferred = selectPreferredPitchStrategy(context);
    if (preferred.key === active.key && !force) return;
    switchPitchStrategy(preferred, context ? `${context}-enable` : "enable");
    if (preferred.key === "yin") {
      pitchStrategyState.lockUntil = 0;
      pitchStrategyState.lockReason = null;
      pitchStrategyState.lockReasonDetail = null;
      pitchStrategyState.lockDuration = 0;
      pitchStrategyState.lockedAt = 0;
      pitchStrategyState.lockContext = null;
      pitchStrategyState.autoRetryUntil = 0;
      cancelPitchAutoRetryTimers();
    }
  }

  function degradePitchStrategy(reason, { cooldownMs, detail, context } = {}) {
    if (pitchStrategyState.activeKey === "acf") return;
    const now = nowMs();
    const requestedCooldown = Number.isFinite(cooldownMs)
      ? cooldownMs
      : (reason === "error" ? PITCH_RETRY_ERROR_COOLDOWN_MS : PITCH_RETRY_COOLDOWN_MS);

    clearPitchAutoRetry();

    let timeout;
    if (reason === "error") {
      const guard = Math.max(3500, Math.min(PITCH_RETRY_ERROR_GUARD_MS, requestedCooldown));
      timeout = guard;
      pitchStrategyState.autoRetryUntil = now + Math.max(requestedCooldown, guard + 4000);
    } else {
      timeout = Math.max(PITCH_RETRY_COOLDOWN_MS, requestedCooldown);
      pitchStrategyState.autoRetryUntil = 0;
    }

    pitchStrategyState.lockUntil = Math.max(pitchStrategyState.lockUntil, now + timeout);
    pitchStrategyState.lockReason = reason || "degraded";
    pitchStrategyState.lockReasonDetail = detail || null;
    pitchStrategyState.lockedAt = now;
    pitchStrategyState.lockDuration = timeout;
    pitchStrategyState.lockContext = context || null;
    schedulePitchBufferTrim();
    const logReason = detail ? `${reason || "degraded"}:${detail}` : (reason || "degraded");
    if (reason === "error") {
      schedulePitchAutoRetry("error");
    }
    switchPitchStrategy(pitchStrategies.acf, logReason);
  }

  function clearPitchAutoRetry({ resetWindow = true } = {}) {
    cancelPitchAutoRetryTimers();
    if (resetWindow) {
      pitchStrategyState.autoRetryUntil = 0;
    }
  }

  function schedulePitchAutoRetry(reason) {
    if (typeof setTimeout !== "function") return;
    if (!pitchStrategyState.autoRetryUntil) return;

    let pendingTimer = null;

    function scheduleNext(delay) {
      if (typeof setTimeout !== "function") return null;
      pendingTimer = registerPitchRetryTimer(setTimeout(attempt, delay));
      return pendingTimer;
    }

    function attempt() {
      if (pendingTimer != null) {
        releasePitchRetryTimer(pendingTimer);
        pendingTimer = null;
      }
      const now = nowMs();
      if (pitchStrategyState.lockReason !== reason) {
        clearPitchAutoRetry();
        return;
      }
      if (pitchStrategyState.lockUntil && now + 50 < pitchStrategyState.lockUntil) {
        const retryDelay = Math.max(
          800,
          Math.min(
            PITCH_RETRY_ERROR_ATTEMPT_SPACING_MS,
            pitchStrategyState.lockUntil - now + 200,
          ),
        );
        scheduleNext(retryDelay);
        return;
      }

      const contexts = pitchStrategyState.lockContext
        ? [pitchStrategyState.lockContext]
        : ["realtime", "offline"];
      for (const ctx of contexts) {
        maybeEnableAdvancedPitch(ctx, { allowRetry: true });
      }

      if (
        pitchStrategyState.activeKey === "acf" &&
        pitchStrategyState.lockReason === reason &&
        pitchStrategyState.autoRetryUntil &&
        now < pitchStrategyState.autoRetryUntil
      ) {
        const nextDelay = Math.max(
          1500,
          Math.min(
            PITCH_RETRY_ERROR_ATTEMPT_SPACING_MS,
            pitchStrategyState.autoRetryUntil - now,
          ),
        );
        scheduleNext(nextDelay);
      } else {
        clearPitchAutoRetry();
      }
    }

    const now = nowMs();
    const guardDelay = pitchStrategyState.lockUntil && pitchStrategyState.lockUntil > now
      ? pitchStrategyState.lockUntil - now + 200
      : 800;
    const initialDelay = Math.max(
      1200,
      Math.min(PITCH_RETRY_ERROR_ATTEMPT_SPACING_MS, guardDelay),
    );
    scheduleNext(initialDelay);
  }

  function selectPreferredPitchStrategy(context) {
    const override = getPitchModeOverride();
    if (override === "force-baseline") return pitchStrategies.acf;
    if (override === "force-advanced") return pitchStrategies.yin;

    const info = estimateDeviceTier();
    if (info.saveData) return pitchStrategies.acf;

    const contextHint = context === "offline" ? 1 : 0;
    const now = nowMs();
    const stillCooling = pitchStrategyState.lockUntil && now < pitchStrategyState.lockUntil;
    const lockDuration = pitchStrategyState.lockDuration || PITCH_RETRY_COOLDOWN_MS;
    const elapsed = Math.max(0, now - (pitchStrategyState.lockedAt || 0));
    const ratio = lockDuration > 0 ? elapsed / lockDuration : 1;
    const runtimePenalty =
      stillCooling &&
        pitchStrategyState.lockReason === "runtime" &&
        ratio < (context === "offline" ? 0.35 : 0.5)
        ? 1
        : 0;
    const allowMobileAdvanced = info.isMobile && info.score >= 7 && !info.lowPowerMode;
    const allowDesktopAdvanced = !info.isMobile && info.score >= 5;

    if ((allowMobileAdvanced || allowDesktopAdvanced) && runtimePenalty === 0) {
      return pitchStrategies.yin;
    }

    if ((allowMobileAdvanced || allowDesktopAdvanced) && runtimePenalty && contextHint) {
      return pitchStrategies.yin;
    }

    if (!info.isMobile && info.score >= 6 && runtimePenalty === 0) return pitchStrategies.yin;
    return pitchStrategies.acf;
  }

  function getPitchModeOverride() {
    try {
      if (typeof localStorage === "undefined") return null;
      const val = localStorage.getItem("vpa:pitchMode");
      if (val === "force-baseline" || val === "force-advanced") return val;
    } catch { }
    return null;
  }

  function estimateDeviceTier() {
    const nav = typeof navigator !== "undefined" ? navigator : {};
    const uaData = nav.userAgentData || null;
    const rawUa = nav.userAgent || "";
    const isMobile = uaData?.mobile ?? /Android|iP(hone|od)|Mobile/i.test(rawUa);
    const isTablet = /iPad|Tablet|Silk|PlayBook|Pixel C|Nexus 7/i.test(rawUa);
    const concurrency = Number.isFinite(nav.hardwareConcurrency) ? nav.hardwareConcurrency : 2;
    const deviceMemory = Number.isFinite(nav.deviceMemory) ? nav.deviceMemory : 0;
    const saveData = !!nav?.connection?.saveData;
    const lowPowerMode = !!nav?.connection?.effectiveType && /2g|slow-2g/.test(nav.connection.effectiveType);
    let score = concurrency;
    if (deviceMemory) { score += deviceMemory; }
    if (!isMobile || isTablet) { score += 2; }
    if (typeof nav.gpu !== "undefined") score += 1.5;
    if (saveData) score -= 2;
    if (lowPowerMode) score -= 1;

    const highEndApple = /iPhone1[2-9]|iPhone2[0-9]|iPad\sPro|AppleCoreMedia.*(M1|M2|M3)/i.test(rawUa);
    if (highEndApple) score += 2;
    const flagshipAndroid = /Pixel\s(7|8|9)|SM-G99|SM-S9|Snapdragon\s8/i.test(rawUa);
    if (flagshipAndroid) score += 2;

    return { isMobile: isMobile && !isTablet, isTablet, score, saveData, lowPowerMode };
  }

  function schedulePitchBufferTrim() {
    if (typeof setTimeout !== "function") return;
    cancelPitchBufferTrimTimer();
    const now = nowMs();
    const baseDelay = pitchStrategyState.lockUntil
      ? Math.max(0, pitchStrategyState.lockUntil - now)
      : 8000;
    const delay = Math.max(4000, Math.min(baseDelay, 15000));
    trimPitchBuffersTimer = setTimeout(() => {
      if (getActivePitchStrategy().key !== "acf") return;
      if (pitchStrategyState.lockUntil && nowMs() < pitchStrategyState.lockUntil - 5000) return;
      trimYinBuffers();
    }, delay);
  }

  function cancelPitchBufferTrimTimer() {
    if (trimPitchBuffersTimer) {
      clearTimeout(trimPitchBuffersTimer);
      trimPitchBuffersTimer = null;
    }
  }

  function trackPitchRuntime(elapsedMs, strategy, context, frameSamples) {
    if (!strategy || strategy.key === "acf") return;
    const now = nowMs();
    const frameScale = frameSamples ? Math.min(3, Math.max(0.7, frameSamples / 2048)) : 1;
    const contextMultiplier = context === "offline" ? PITCH_RUNTIME_OFFLINE_MULTIPLIER : 1;
    const dynamicBudget = PITCH_RUNTIME_BASE_BUDGET_MS * frameScale * contextMultiplier;
    pitchStrategyState.runtimeEwma = pitchStrategyState.runtimeEwma
      ? (pitchStrategyState.runtimeEwma * 0.7 + elapsedMs * 0.3)
      : elapsedMs;
    const adaptiveBudget = Math.min(
      PITCH_RUNTIME_MAX_BUDGET_MS,
      Math.max(
        PITCH_RUNTIME_MIN_BUDGET_MS,
        Math.max(dynamicBudget, pitchStrategyState.runtimeEwma * 1.6),
      ),
    );

    if (elapsedMs <= adaptiveBudget) {
      if (pitchStrategyState.overBudgetStreak && pitchStrategyState.lastOverBudget) {
        if (now - pitchStrategyState.lastOverBudget > PITCH_RUNTIME_RECOVERY_MS) {
          pitchStrategyState.overBudgetStreak = Math.max(0, pitchStrategyState.overBudgetStreak - 1);
          if (!pitchStrategyState.overBudgetStreak) pitchStrategyState.lastOverBudget = 0;
        }
      }
      return;
    }

    pitchStrategyState.overBudgetStreak += 1;
    pitchStrategyState.lastOverBudget = now;
    if (pitchStrategyState.overBudgetStreak >= PITCH_RUNTIME_OVER_BUDGET_LIMIT) {
      const cooldown = context === "offline"
        ? Math.max(8000, PITCH_RETRY_COOLDOWN_MS / 2)
        : PITCH_RETRY_COOLDOWN_MS;
      degradePitchStrategy("runtime", { cooldownMs: cooldown, detail: `${elapsedMs.toFixed(1)}ms`, context });
    }
  }

  function runPitchDetection(input, sr, { context = "realtime" } = {}) {
    const strategy = getActivePitchStrategy();
    const start = nowMs();
    try {
      const hz = strategy.detect(input, sr);
      const elapsed = nowMs() - start;
      trackPitchRuntime(elapsed, strategy, context, input?.length || 0);
      return hz;
    } catch (err) {
      console.error(`[pitch] ${strategy.key} failed`, err);
      degradePitchStrategy("error", { cooldownMs: PITCH_RETRY_ERROR_COOLDOWN_MS, context });
      if (strategy.key !== "acf") {
        try {
          return pitchStrategies.acf.detect(input, sr);
        } catch (fallbackErr) {
          console.error("[pitch] fallback failed", fallbackErr);
        }
      }
      return null;
    }
  }

  return {
    initializePitchStrategy,
    maybeEnableAdvancedPitch,
    runPitchDetection,
  };
}
