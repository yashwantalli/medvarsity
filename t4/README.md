# Laparoscopic Surgery Datasets — GitHub Repository Survey

A catalog of 12 GitHub repositories relevant to laparoscopic surgery video/image datasets, compiled to scope out data availability, storage requirements, and extractable training examples for computer-vision work on surgical footage.

## Summary

| | Count | Storage | Examples |
|---|---|---|---|
| **Category A** — full video datasets | 9 repos | ~520–700+ GB combined | ~950,000+ annotated frames |
| **Category B** — extracted-image-only datasets | 3 repos | <1 GB each | ~1,200 labeled images |
| **Total** | **12 repos** | — | **~950,000+ usable examples** |

- Largest single dataset: **Cholec80** (85.2 GB post-extraction, 166 GB to download)
- Most videos in one dataset: **MultiBypass140** (140 videos, 2 surgical centers)
- **CAMMA-public** (University of Strasbourg) contributes 6 of the 12 repos and is the dominant source
- Most datasets center on **laparoscopic cholecystectomy** (gallbladder removal), the most common minimally-invasive procedure
- GitHub's 100 MB file limit means repos ship code/annotations only, with videos hosted externally via download links

## Category A — Video Datasets

| # | Repository | Videos | Frames/Examples | Storage | Focus |
|---|---|---|---|---|---|
| 1 | [CAMMA-public/TF-Cholec80](https://github.com/CAMMA-public/TF-Cholec80) | 80 | 370,168 frames | 85.2 GB (166 GB download) | Phase (7) + tool (7) labels |
| 2 | [CAMMA-public/cholect45](https://github.com/CAMMA-public/cholect45) | 45 | 90,489 frames, 127,385 triplets | ~50 GB | Instrument-verb-target action triplets |
| 3 | [CAMMA-public/MultiBypass140](https://github.com/CAMMA-public/MultiBypass140) | 140 | Tens of thousands | ~150–200 GB | Gastric bypass, multi-center (Strasbourg + Bern) |
| 4 | [CAMMA-public/Endoscapes](https://github.com/CAMMA-public/Endoscapes) | 201 | 58,813 frames (+11,090 CVS, 1,933 bbox, 493 seg) | ~100+ GB | Scene segmentation + Critical View of Safety |
| 5 | [CAMMA-public/cholectrack20](https://github.com/CAMMA-public/cholectrack20) | 20 (~14 hrs) | 35,000 frames, 65,000+ tool labels | ~30–50 GB | Multi-tool tracking |
| 6 | [franciszchen/VLSurg-dataset](https://github.com/franciszchen/VLSurg-dataset) | 60 | Thousands | ~50–100 GB | Phases, skill scores, descriptions |
| 7 | [CAMMA-public/ConvLSTM-Surgical-Tool-Tracker](https://github.com/CAMMA-public/ConvLSTM-Surgical-Tool-Tracker) | 80 (Cholec80) | 370,168 frames | 85.2 GB | Tool trajectory tracking |
| 8 | [AI-Medical-Robotics/Surgical-Tool-Localization](https://github.com/AI-Medical-Robotics/Surgical-Tool-Localization) | 24,000 clips + 15 (m2cai16) | 23,000 frames, 2,532 bbox | 109 GB | Tool localization, 7 tool classes |
| 9 | [ruaridhg/blender_laparoscopy](https://github.com/ruaridhg/blender_laparoscopy) | Synthetic | Unlimited | ~6.2 GB | Blender-generated synthetic data |

## Category B — Extracted-Image Datasets

| # | Repository | Source | Examples | Storage | Focus |
|---|---|---|---|---|---|
| 10 | [nancy280/Smoke-Detection-in-Laproscopic-Surgery](https://github.com/nancy280/Smoke-Detection-in-Laproscopic-Surgery) | 10 hysterectomy videos | 600 images (300 hazy / 300 clear) | <1 GB | Smoke/haze detection |
| 11 | [AakarshMishra/Smoke-Detection-in-Laproscopic-Surgery-main](https://github.com/AakarshMishra/Smoke-Detection-in-Laproscopic-Surgery-main) | Same as #10 (fork) | 600 images | <1 GB | Smoke/haze detection |
| 12 | [cchandel-dev/Laproscopic-Surgery-Work](https://github.com/cchandel-dev/Laproscopic-Surgery-Work) | User-supplied (private) | Depends on input | Small (code only) | Image annotation, tool detection, temporal ordering |

## Repository Details

### Video-processing / smoke-detection pipeline (repos #10–11)
- **Source:** 10 robot-assisted laparoscopic hysterectomy videos from the EPSRC Centre for Interventional and Surgical Sciences
- **Extraction:** 1 FPS, manually curated to 300 hazy + 300 clear frames
- **Models:** CNN, Random Forest, XGBoost, SVM (scikit-learn)
- **Stack:** Python, Jupyter, OpenCV

### cchandel-dev/Laproscopic-Surgery-Work (repo #12)
Three sub-projects in one repo:
1. **Private Image Annotator** — local-only annotation tool (no image upload off-device); run via `pip install -r requirements.txt` then `python "Image Annotator.py"`
2. **Surgical Tool Detection** — object detection flagging when a tool enters frame
3. **Temporal Ordering** — Siamese network predicting which of two frames occurred first

### TF-Cholec80 (repo #1) — requirements detail
- **Storage:** 166 GB free space needed to download; 85.2 GB after extraction/archive removal
- **Software:** Python 3, TensorFlow ≥1.4 or TF2, tqdm, Matplotlib (demo notebooks only)
- **Dev config used upstream:** Ubuntu 20.04, CUDA 10.1, NVIDIA GTX 1080 Ti

### blender_laparoscopy (repo #9) — synthetic data detail
- **Assets:** `blender.zip` (~61 MB, editable `.blend` liver files), `liver.zip` (~6.09 GB training PNGs), `examples.blend` (~15.67 MB, `.avi` colonoscopy/laparoscopy demos)
- **Prep:** each rendered frame needs a paired segmentation mask for polyp detection
- **Software:** Blender 3.4.1 (newer versions may break compatibility with the Blender Randomiser add-on), Python, Pandas, PyTorch (UNet)
- **Hardware:** min. dual-core CPU / 4 GB RAM; recommended quad-core / 16 GB RAM / discrete GPU; no-GPU laptops can use Google Colab free tier (T4 GPU, 15 GB VRAM, 12-hour session cap)

## Common Software Requirements Across Repos

- Python 3
- PyTorch or TensorFlow (1.x or 2.x depending on repo age)
- OpenCV
- CUDA-capable GPU recommended for training

## Sources

- `handbook` — raw research notes with per-repo links and requirement details, plus the original scoping question ("how many repos, what sizes, how many extractable examples")
- `laproscopic_data.pdf` — the compiled, formatted summary report this README is based on
