export function mixChannelDataToMono(channels, outCh){
  if (!outCh || typeof outCh.length !== "number"){
    throw new TypeError("outCh must be a Float32Array");
  }
  const frameCount = outCh.length;
  outCh.fill(0);
  if (!Array.isArray(channels) || frameCount === 0) return 0;

  let validChannels = 0;
  for (const chData of channels){
    if (!(chData instanceof Float32Array) || chData.length !== frameCount) continue;
    let hasNaN = false;
    for (let i = 0; i < frameCount; i++){
      if (Number.isNaN(chData[i])){
        hasNaN = true;
        break;
      }
    }
    if (hasNaN) continue;
    for (let i = 0; i < frameCount; i++){
      outCh[i] += chData[i];
    }
    validChannels++;
  }

  if (validChannels > 0){
    const inv = 1 / validChannels;
    for (let i = 0; i < frameCount; i++){
      outCh[i] *= inv;
    }
  }
  return validChannels;
}
