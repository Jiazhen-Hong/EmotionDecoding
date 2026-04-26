# Decoding Video-Elicited Emotions from EEG

Submission for the **Alljoined Machine Learning Researcher take-home challenge** (April 2026).

**Author:** Jiazhen Hong
([Website](https://jiazhen-hong.github.io/jiazhenhong/) · [LinkedIn](http://www.linkedin.com/in/jiazhen-hong66) · jh1590@rutgers.edu)

## Task

6-class emotion classification (anger, disgust, fear, happy, neutral, sadness)
from 31-channel EEG, with test trials drawn from a held-out session unseen at training.

- 336 training trials over 14 sessions
- 24 test trials from 1 held-out session
- Sampling rate: 200 Hz, notch-filtered at 60 Hz
- Trials are inf-padded to a uniform tensor

## Result

| | Mean LOSO accuracy | Std |
|---|---|---|
| **Submitted (Row 9, hard balanced assignment)** | **0.652** | 0.135 |
| Chance | 0.167 | – |

Pipeline: spectral features (band power + differential entropy + Hjorth) →
per-session robust z-score → CORAL alignment → log-probability ensemble of a
spectral LR head and a Riemannian tangent-space head → Hungarian assignment
under the balanced-class prior.

## Files

```
Emotion_Detection/
├── raw_files_from_alljoined/  (local-only, ignore due to NDA)
   ├── Take-home assignment instructions.docx
   ├── data.npz
   └── dtrial_info_train.csv
├── notebook.ipynb              # full pipeline (run top-to-bottom to reproduce)
├── submission.csv              # delivered predictions (24 labels)
├── ShortReport.pdf             # 5-page report
├── Presentation.pdf            # slide 
├── Results/                    # ablation CSV, figures
├── Signal_Quality_Check/       # raw + cleaned diagnostic figures
   ├── cleaned
   ├── ERP_TF_visulization (raw)
   └── raw
└── README.md
```

## Reproduce

Run `notebook.ipynb` top-to-bottom. Outputs land in `Results/` and `Signal_Quality_Check/`.
