export let dropZone, recordBtn, fileInput, uploadFab, statusEl, statusLabel, statusTimer, meter, femaleVal, maleVal, warmupCard;
export let pitchWrap, pitchCanvas, pitchNowEl, bandNowEl, volNowEl, formantWrap, f1NowEl, f2NowEl, f3NowEl, breathNowEl, resonanceNowEl, tiltNowEl, resBarChest, resBarMask, resBarHead, resValChest, resValMask, resValHead;
export let exportBtn, helpBtn, helpOverlay, localeBtn, localeMenu, onboardTip, onboardDismissBtn;

function init() {
    dropZone = document.getElementById("dropZone");
    recordBtn = document.getElementById("recordBtn");
    fileInput = document.getElementById("fileInput");
    uploadFab = document.getElementById("uploadFab");
    statusEl = document.getElementById("status");
    statusLabel = document.getElementById("statusLabel");
    statusTimer = document.getElementById("statusTimer");
    meter = document.getElementById("meter");
    femaleVal = document.getElementById("femaleVal");
    maleVal = document.getElementById("maleVal");
    warmupCard = document.getElementById("warmupCard");

    pitchWrap = document.getElementById("pitchWrap");
    pitchCanvas = document.getElementById("pitchCanvas");
    pitchNowEl = document.getElementById("pitchNow");
    bandNowEl = document.getElementById("bandNow");
    volNowEl = document.getElementById("volNow");
    formantWrap = document.getElementById("formantWrap");
    f1NowEl = document.getElementById("f1Now");
    f2NowEl = document.getElementById("f2Now");
    f3NowEl = document.getElementById("f3Now");
    breathNowEl = document.getElementById("breathNow");
    resonanceNowEl = document.getElementById("resonanceNow");
    tiltNowEl = document.getElementById("tiltNow");
    resBarChest = document.getElementById("resChest");
    resBarMask = document.getElementById("resMask");
    resBarHead = document.getElementById("resHead");
    resValChest = document.getElementById("resChestVal");
    resValMask = document.getElementById("resMaskVal");
    resValHead = document.getElementById("resHeadVal");

    exportBtn = document.getElementById("exportBtn");

    helpBtn = document.getElementById("helpBtn");
    helpOverlay = document.getElementById("helpOverlay");
    localeBtn = document.getElementById("localeBtn");
    localeMenu = document.getElementById("localeMenu");
    onboardTip = document.getElementById("onboardTip");
    onboardDismissBtn = document.getElementById("onboardDismiss");
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
} else {
    init();
}
