export async function bootstrapAppRuntime(deps) {
  const {
    env,
    getLocaleValue,
    initI18n,
    onLocaleChange,
    onLocaleUpdated,
    sharedDetectThreadCount,
    threadStorageKey,
  } = deps;

  const threadDecision = sharedDetectThreadCount({ storageKey: threadStorageKey });
  const pickedThreads = threadDecision.threads;

  if (!env.backends.onnx) {
    env.backends.onnx = {};
  }
  if (!env.backends.onnx.wasm) {
    env.backends.onnx.wasm = {};
  }
  env.backends.onnx.wasm.numThreads = pickedThreads;

  if (threadDecision.reason === "error" && threadDecision.error) {
    console.warn("WASM thread detection failed, falling back to single thread.", threadDecision.error);
  } else if (threadDecision.reason === "safari") {
    console.info("Safari detected – forcing ONNX Runtime WASM to single thread.");
  } else {
    console.info(`ONNX Runtime WASM threads: ${pickedThreads} (${threadDecision.reason}).`);
  }

  await initI18n();

  let analysisText = getLocaleValue("analysis");
  let summaryText = getLocaleValue("summary");

  onLocaleChange(() => {
    analysisText = getLocaleValue("analysis");
    summaryText = getLocaleValue("summary");
    if (typeof onLocaleUpdated === "function") onLocaleUpdated();
  });

  return {
    getAnalysisText: () => analysisText,
    getSummaryText: () => summaryText,
    pickedThreads,
    threadDecision,
  };
}
