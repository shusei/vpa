# Voice Age 2.0

## Product Contract

Voice Age 2.0 estimates a broad voice-age impression. It does not estimate a
person's actual age and must not be used for identity, medical, legal, or
eligibility decisions.

The strict feminine/masculine score remains `advanced-beta-1`. Voice age has
its own version, `voice-age-impression-2.0.0-research`, so an age-only change
does not silently change challenge scores.

## Analysis Modes

Connected speech and sustained vowels use separate evidence:

| Evidence | Connected speech | Sustained vowel |
| --- | --- | --- |
| CPP | Used after quality gate | Used after quality gate |
| HNR | Used after quality gate | Used after quality gate |
| Jitter | Measured as a reference only | Used after cycle-stability gate |
| Shimmer | Measured as a reference only | Used after cycle-stability gate |
| Intonation | Used | Not used |

Jitter and Shimmer are cycle perturbation measures. Their conventional
interpretation assumes a stable sustained vowel, so values from everyday
speech are never included in the age calculation.

## Browser Measurement

- Audio is decoded to mono and analyzed locally.
- At most 12 seconds are measured. Long files use evenly spaced four-second
  windows so the feature does not add unbounded work.
- HNR uses normalized short-term autocorrelation.
- CPP uses a power-cepstral peak relative to a fitted background trend.
- Jitter uses consecutive accepted period differences.
- Shimmer uses consecutive accepted cycle-amplitude differences.
- Only aggregate values are retained. The extension does not retain or upload
  waveform samples.

These browser implementations follow the metric definitions but are not
claimed to be numerically identical to Praat. Algorithm versioning and
fixtures are therefore mandatory.

## Refusal Rules

Voice age is withheld while the strict presentation result remains available
when any required condition fails:

- connected speech shorter than 3.5 seconds;
- sustained vowel shorter than 2.5 seconds;
- low level, clipping, insufficient voiced coverage, or weak periodicity;
- missing reliable CPP or HNR;
- missing stable Jitter or Shimmer for sustained-vowel analysis;
- combined age confidence below 0.5.

Until a speaker-disjoint human calibration set is accepted, the research
preview caps age confidence below `high`.

## Calibration Protocol

The repository manifest defines the acceptance target. A release-ready
calibration set must:

1. Cover each declared age band with the minimum number of distinct speakers.
2. Keep connected speech and sustained vowels as separate samples and models.
3. Split train, calibration, and test sets by speaker, never by clip.
4. Balance or report language, voice presentation, microphone class, and noise.
5. Include clean, noisy, breathy, falsetto, and clipped quality conditions.
6. Report refusal rate, macro-F1 by age band, adjacent-band tolerance, and
   calibration error.
7. Keep one immutable fixture manifest per released age-scoring version.

Common Voice age labels are self-reported decade categories. They may support
research calibration, but they do not by themselves prove accuracy for
Taiwanese Mandarin, Japanese, transformed voices, or this product's recording
prompts.

## Analytics

GA4 may receive flow name, readiness, sample type, confidence label, scoring
version, and strict score in ten-point bands. It must not receive audio,
precise age ranges, pitch, Jitter, Shimmer, HNR, CPP, or other precise acoustic
measurements.

## References

- [Praat Voice 2: Jitter](https://praat.org/manual/Voice_2__Jitter.html)
- [Praat Voice 3: Shimmer](https://praat.org/manual/Voice_3__Shimmer.html)
- [Praat Harmonicity](https://praat.org/manual/Harmonicity.html)
- [Praat PowerCepstrogram: Get CPPS](https://praat.org/manual/PowerCepstrogram__Get_CPPS___.html)
- [Mozilla Common Voice datasets](https://commonvoice.mozilla.org/en/datasets)
- [Effects of Aging on Vocal Fundamental Frequency and Voice Quality](https://pubmed.ncbi.nlm.nih.gov/23328404/)
- [Vocal Function in Older Singers and Nonsingers](https://pubmed.ncbi.nlm.nih.gov/21889299/)
