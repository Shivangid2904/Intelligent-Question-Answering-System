# IntelliAsk Baseline Benchmark Evaluation Report

## Executive Summary

- **Total Questions Evaluated**: 18 (15 answerable, 3 unanswerable / abstention)
- **Corpus**: 7 full-text PDFs in `data/papers/` (82 total physical pages, 1,011 indexed chunks)
- **Retrieval Pipeline**: Dense FAISS IndexFlatIP with `all-MiniLM-L6-v2` dense embeddings
- **Generation & Citation Pipeline**: `check_evidence_sufficiency` (threshold 0.38) + `llama3.2` local LLM via Ollama
- **Overall Mean Score**: **0.528** / 1.000
- **Answerable Questions Score**: **0.433** / 1.000
- **Abstention Accuracy on Unanswerable Queries**: **100.0%** (3/3)
- **Document Recall@3 (Answerable)**: **93.3%** (Recall@10: 93.3%)
- **Page-Level Provenance Recall@3 (Answerable)**: **40.0%** (Recall@10: 73.3%)
- **Citation Document Accuracy**: **93.3%**
- **Citation Page-Level Accuracy**: **53.3%**


## Performance by Category

| Category | Questions | Mean Score | Doc Recall@3 | Page Recall@3 |
|---|:---:|:---:|:---:|:---:|
| Cross-Paper Comparison | 4 | 0.250 | 100.0% | 25.0% |
| Direct Factual Retrieval | 4 | 0.250 | 100.0% | 25.0% |
| Numerical Results & Methodology | 4 | 0.750 | 75.0% | 50.0% |
| Page-Level Evidence & Citation | 3 | 0.500 | 100.0% | 66.7% |
| Unanswerable (Abstention) | 3 | 1.000 | 100.0% | 100.0% |

## Detailed Question-by-Question Results

| ID | Category | Target | Mode | Score | Doc Hit@3 | Page Hit@3 | Cited Doc | Cited Page | Abstain OK |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Q01** | Direct Factual R | journal.pone.0344604.pdf | `llm_answer` | 0.00 | Yes | No | Yes | Yes | Yes |
| **Q02** | Direct Factual R | 2505.21049.pdf | `llm_answer` | 0.00 | Yes | No | Yes | No | Yes |
| **Q03** | Direct Factual R | 1-s2.0-S26659174260002... | `llm_answer` | 1.00 | Yes | Yes | Yes | Yes | Yes |
| **Q04** | Direct Factual R | Deep_Learning_Enhanced... | `llm_answer` | 0.00 | Yes | No | Yes | Yes | Yes |
| **Q05** | Cross-Paper Comp | journal.pone.0344604.p... | `llm_answer` | 0.50 | Yes | No | Yes | No | Yes |
| **Q06** | Cross-Paper Comp | journal.pone.0344604.p... | `llm_answer` | 0.00 | Yes | Yes | Yes | Yes | Yes |
| **Q07** | Cross-Paper Comp | 2505.21049.pdf, Deep_L... | `llm_answer` | 0.00 | Yes | No | Yes | No | Yes |
| **Q08** | Cross-Paper Comp | isprs-archives-XLVIII-... | `llm_answer` | 0.50 | Yes | No | Yes | No | Yes |
| **Q09** | Numerical Result | journal.pone.0344604.pdf | `llm_answer` | 1.00 | Yes | Yes | Yes | Yes | Yes |
| **Q10** | Numerical Result | 2505.21049.pdf | `llm_answer` | 1.00 | Yes | No | Yes | No | Yes |
| **Q11** | Numerical Result | isprs-archives-XLVIII-... | `llm_answer` | 0.00 | No | No | No | No | Yes |
| **Q12** | Numerical Result | isprs-archives-XLVIII-... | `llm_answer` | 1.00 | Yes | Yes | Yes | Yes | Yes |
| **Q13** | Page-Level Evide | journal.pone.0344604.pdf | `llm_answer` | 0.50 | Yes | Yes | Yes | Yes | Yes |
| **Q14** | Page-Level Evide | 2012.10802v2.pdf | `llm_answer` | 0.50 | Yes | No | Yes | No | Yes |
| **Q15** | Page-Level Evide | paper.pdf | `llm_answer` | 0.50 | Yes | Yes | Yes | Yes | Yes |
| **Q16** | Unanswerable (Ab | 1-s2.0-S26659174260002... | `llm_answer` | 1.00 | Yes | No | No | No | Yes |
| **Q17** | Unanswerable (Ab | 2012.10802v2.pdf | `llm_answer` | 1.00 | No | No | No | No | Yes |
| **Q18** | Unanswerable (Ab | 2505.21049.pdf | `llm_answer` | 1.00 | Yes | No | No | No | Yes |

## Analysis of Failures & Limitations

### Identified Deficiencies (11 item(s) below full credit)

#### [Q01] Which smartphone model and integrated sensor were used to collect the RGB-D dataset in Rauf Butt et al. (2026), and in which city was the data gathered?
- **Category**: Direct Factual Retrieval
- **Answer Mode**: `llm_answer` (Score: 0.00)
- **Expected Answer**: iPhone 14 Pro equipped with an integrated LiDAR sensor; collected in Lahore, Pakistan (specifically Walton Road, Airport Road, and Manawan).
- **Actual Response**: I cannot provide the specific information you're looking for based on the text. The text does not contain enough information about the smartphone model and integrated sensor used to collect the RGB-D dataset in Rauf Butt et al. (2026)....
- **Citations Generated**: ['[journal.pone.0344604.pdf, p. 14]', '[Deep_Learning_Enhanced_Feature_Extraction_of_Potholes_Using_Vision_and_LiDAR_Data_for_Road_Maintenance.pdf, p. 5]', '[journal.pone.0344604.pdf, p. 17]']
- **Top Chunk Retrieved**: [journal.pone.0344604.pdf, p. 14] (Score: 0.513)
- **Diagnostic Rationale**: 
  - **Dense Retrieval Precision Limitation**: The exact supporting page was not retrieved in the top-3 dense candidates (top-1 score: 0.513), leading to incomplete or imperfect context.

#### [Q02] In Dehao Wang et al. (2025), what method is proposed for pothole area estimation from bounding boxes, and what geometric shape does it assume for the pothole?
- **Category**: Direct Factual Retrieval
- **Answer Mode**: `llm_answer` (Score: 0.00)
- **Expected Answer**: The Minimum Bounding Triangulated Pixel (MBTP) method. It assumes an elliptical shape for the pothole, approximating final area by multiplying bounding rectangle area by pi/4.
- **Actual Response**: According to the cited passage [2505.21049.pdf, p. 5], the proposed method for pothole area estimation is the MBTP method, which combines the 3D minimum bounding rectangle and the pixel-based triangular area accumulation. The MBTP method assumes a triangular shape for the pothole.

Explanation: The ...
- **Citations Generated**: ['[2505.21049.pdf, p. 2]', '[2505.21049.pdf, p. 9]']
- **Top Chunk Retrieved**: [2505.21049.pdf, p. 2] (Score: 0.716)
- **Diagnostic Rationale**: 
  - **Dense Retrieval Precision Limitation**: The exact supporting page was not retrieved in the top-3 dense candidates (top-1 score: 0.716), leading to incomplete or imperfect context.

#### [Q04] In Karukayil et al. (2024), what specific LiDAR sensor model is integrated with the RGB camera, and in what file format is the final interactive map of detected potholes saved?
- **Category**: Direct Factual Retrieval
- **Answer Mode**: `llm_answer` (Score: 0.00)
- **Expected Answer**: Ouster OS1-32 LiDAR (resolution 2048 x 32); final interactive pothole map is saved in HTML format.
- **Actual Response**: I couldn't find any information on a specific LiDAR sensor model integrated with the RGB camera in Karukayil et al. (2024). The cited passages only discuss the general concept of LiDAR-camera fusion for pothole detection, but do not provide details on a specific sensor model or file format for the f...
- **Citations Generated**: ['[Deep_Learning_Enhanced_Feature_Extraction_of_Potholes_Using_Vision_and_LiDAR_Data_for_Road_Maintenance.pdf, p. 3]', '[Deep_Learning_Enhanced_Feature_Extraction_of_Potholes_Using_Vision_and_LiDAR_Data_for_Road_Maintenance.pdf, p. 4]', '[journal.pone.0344604.pdf, p. 5]']
- **Top Chunk Retrieved**: [Deep_Learning_Enhanced_Feature_Extraction_of_Potholes_Using_Vision_and_LiDAR_Data_for_Road_Maintenance.pdf, p. 3] (Score: 0.689)
- **Diagnostic Rationale**: 
  - **Dense Retrieval Precision Limitation**: The exact supporting page was not retrieved in the top-3 dense candidates (top-1 score: 0.689), leading to incomplete or imperfect context.

#### [Q05] Compare the depth acquisition modalities across Rauf Butt et al. (2026), Hanzheng Wang et al. (2025), and Ponnusamy (2026). What primary sensor technology does each paper employ to estimate road surface depth?
- **Category**: Cross-Paper Comparison
- **Answer Mode**: `llm_answer` (Score: 0.50)
- **Expected Answer**: Rauf Butt et al. performs monocular depth prediction from a single RGB image (MiDaS DPT) with iPhone 14 Pro LiDAR for ground-truth reference; Hanzheng Wang et al. uses binocular stereo vision (ZED 2i); Ponnusamy uses active Time-of-Flight (Basler ToF).
- **Actual Response**: I cannot provide a comparison of the depth acquisition modalities across Rauf Butt et al. (2026), Hanzheng Wang et al. (2025), and Ponnusamy (2026) without additional information. 

However, based on the provided passages, I can tell you that Ponnusamy (2026) employed Time-of-Flight (ToF) depth tech...
- **Citations Generated**: ['[1-s2.0-S2665917426000206-main.pdf, p. 5]', '[2012.10802v2.pdf, p. 5]', '[1-s2.0-S2665917426000206-main.pdf, p. 9]']
- **Top Chunk Retrieved**: [1-s2.0-S2665917426000206-main.pdf, p. 5] (Score: 0.556)
- **Diagnostic Rationale**: 
  - **Dense Retrieval Precision Limitation**: The exact supporting page was not retrieved in the top-3 dense candidates (top-1 score: 0.556), leading to incomplete or imperfect context.

#### [Q06] How do Rauf Butt et al. (2026), Ponnusamy (2026), and Rui Fan et al. (arXiv:2012.10802v2) obtain physical ground truth for pothole depth and geometry?
- **Category**: Cross-Paper Comparison
- **Answer Mode**: `llm_answer` (Score: 0.00)
- **Expected Answer**: Rauf Butt et al.: handheld laser rangefinder (+/-1 mm); Ponnusamy: calibrated graduated steel scale (+/-1 mm) and straight-edge reference bar (+/-3 mm expanded uncertainty); Rui Fan et al.: 3D laser scanner (BQ Ciclop) on gypsum molds registered with ICP.
- **Actual Response**: Rauff Butt et al. (2026) and Ponnusamy (2026) use a laser rangefinder to measure pothole depths with a precision of ±1 mm. [journal.pone.0344604.pdf, p. 9] 

Rui Fan et al. (not found in the provided papers, but mentioned in the reference list) are not cited in the provided passages, so no informati...
- **Citations Generated**: ['[journal.pone.0344604.pdf, p. 9]']
- **Top Chunk Retrieved**: [journal.pone.0344604.pdf, p. 9] (Score: 0.671)
- **Diagnostic Rationale**: 
  - **Partial Semantic Match**: The LLM answer addressed only a subset of the required numerical values or entities.

#### [Q07] Which YOLO family variants are evaluated or utilized for pothole detection in Dehao Wang et al. (2025), Karukayil et al. (2024), and Percy Lam et al.?
- **Category**: Cross-Paper Comparison
- **Answer Mode**: `llm_answer` (Score: 0.00)
- **Expected Answer**: Dehao Wang et al.: ACSH-YOLOv8 (specifically ACSH-YOLOv8n) vs YOLOv8/YOLOv8s; Karukayil et al.: YOLOv5, YOLOv6, YOLOv7, and YOLOv8; Percy Lam et al.: YOLOv11x-seg.
- **Actual Response**: I cannot help you with this request as the passages do not contain enough information to provide a specific answer....
- **Citations Generated**: ['[2505.21049.pdf, p. 15]', '[paper.pdf, p. 9]', '[Deep_Learning_Enhanced_Feature_Extraction_of_Potholes_Using_Vision_and_LiDAR_Data_for_Road_Maintenance.pdf, p. 8]']
- **Top Chunk Retrieved**: [2505.21049.pdf, p. 15] (Score: 0.694)
- **Diagnostic Rationale**: 
  - **Dense Retrieval Precision Limitation**: The exact supporting page was not retrieved in the top-3 dense candidates (top-1 score: 0.694), leading to incomplete or imperfect context.

#### [Q08] What GPU hardware was used for training or executing algorithms in Hanzheng Wang et al. (2025) and Rui Fan et al. (arXiv:2012.10802v2)?
- **Category**: Cross-Paper Comparison
- **Answer Mode**: `llm_answer` (Score: 0.50)
- **Expected Answer**: Hanzheng Wang et al.: NVIDIA GeForce RTX 3090 GPU (batch size 8); Rui Fan et al.: NVIDIA GeForce RTX 2080 Ti GPU (CUDA implementation).
- **Actual Response**: The GPU hardware used for training or executing algorithms in Hanzheng Wang et al. (2025) and Rui Fan et al. (arXiv:2012.10802v2) was an NVIDIA RTX 4060 GPU for Hanzheng Wang et al., and an NVIDIA RTX 2080 Ti GPU for Rui Fan et al., respectively.

[Russian translation] 

RTX 4060 was used in  journa...
- **Citations Generated**: ['[journal.pone.0344604.pdf, p. 11]', '[2505.21049.pdf, p. 8]', '[2012.10802v2.pdf, p. 8]']
- **Top Chunk Retrieved**: [journal.pone.0344604.pdf, p. 11] (Score: 0.542)
- **Diagnostic Rationale**: 
  - **Dense Retrieval Precision Limitation**: The exact supporting page was not retrieved in the top-3 dense candidates (top-1 score: 0.542), leading to incomplete or imperfect context.

#### [Q11] In Hanzheng Wang et al. (2025), what relative errors are achieved for pothole depth and surface area measurement at vehicle speeds of 40 km/h, and what percentage cost reduction is claimed compared to traditional laser sensors?
- **Category**: Numerical Results & Methodology
- **Answer Mode**: `llm_answer` (Score: 0.00)
- **Expected Answer**: Depth relative error: 12.53%; Surface area relative error: 18.19% (82% DR accuracy); Hardware cost reduced by 90% compared to laser sensors (and 85% compared to Pathway inspection vehicles).
- **Actual Response**: I couldn't find any information on relative errors achieved for pothole depth and surface area measurement at vehicle speeds of 40 km/h in Hanzheng Wang et al. (2025) or any claimed percentage cost reduction compared to traditional laser sensors. The cited passages only discuss the limitations and e...
- **Citations Generated**: ['[1-s2.0-S2665917426000206-main.pdf, p. 4]', '[journal.pone.0344604.pdf, p. 6]', '[journal.pone.0344604.pdf, p. 11]']
- **Top Chunk Retrieved**: [1-s2.0-S2665917426000206-main.pdf, p. 4] (Score: 0.629)
- **Diagnostic Rationale**: 
  - **Dense Retrieval Precision Limitation**: The exact supporting page was not retrieved in the top-3 dense candidates (top-1 score: 0.629), leading to incomplete or imperfect context.

#### [Q13] On which page and in which table do Rauf Butt et al. (2026) report the ablation study comparing different loss function combinations, and which combination yields the lowest MAE?
- **Category**: Page-Level Evidence & Citation
- **Answer Mode**: `llm_answer` (Score: 0.50)
- **Expected Answer**: Page 14, Table 4. Full SILog + Gradient loss yields the lowest MAE of 0.089 (compared to standard MSE at MAE = 0.140).
- **Actual Response**: The ablation study comparing different loss function combinations is reported in Table 2 (p. 12) and Figure 5 of the paper [journal.pone.0344604.pdf].

The Full SILog loss with gradient matching yields the lowest MAE, with a value of 0.089. This is because the full SILog loss with gradient matching ...
- **Citations Generated**: ['[journal.pone.0344604.pdf, p. 14]', '[2505.21049.pdf, p. 9]', '[journal.pone.0344604.pdf, p. 12]']
- **Top Chunk Retrieved**: [journal.pone.0344604.pdf, p. 14] (Score: 0.521)
- **Diagnostic Rationale**: 
  - **Partial Semantic Match**: The LLM answer addressed only a subset of the required numerical values or entities.

#### [Q14] On which page and in which section of Rui Fan et al. (arXiv:2012.10802v2) is the stereo rig roll angle phi formally integrated into the disparity transformation, and why is this roll angle considered critical?
- **Category**: Page-Level Evidence & Citation
- **Answer Mode**: `llm_answer` (Score: 0.50)
- **Expected Answer**: Page 4, Section II.C (Disparity Transformation). Incorporating roll angle phi prevents disparity profile tilting caused by road roll, enabling accurate planar disparity projection and pothole segmentation.
- **Actual Response**: The stereo rig roll angle φ is formally integrated into the disparity transformation on page 3 of Rui Fan et al. (arXiv:2012.10802v2) in section 2, titled "Generalized perspective transformation and disparity estimation".

According to the text, φ is considered critical because when neglected, it re...
- **Citations Generated**: ['[2012.10802v2.pdf, p. 3]']
- **Top Chunk Retrieved**: [2012.10802v2.pdf, p. 3] (Score: 0.604)
- **Diagnostic Rationale**: 
  - **Dense Retrieval Precision Limitation**: The exact supporting page was not retrieved in the top-3 dense candidates (top-1 score: 0.604), leading to incomplete or imperfect context.

#### [Q15] In Percy Lam et al., what overall range of F1 score improvement do the authors report when fusing RGB and depth detectors under out-of-context testing conditions?
- **Category**: Page-Level Evidence & Citation
- **Answer Mode**: `llm_answer` (Score: 0.50)
- **Expected Answer**: Fusing RGB and depth detectors improves out-of-context F1 scores by an absolute range of 0.05 to 0.13, corresponding to a relative improvement of 16% to 81%.
- **Actual Response**: The authors report an F1 score improvement range of 16% to 81% when fusing RGB and depth detectors under out-of-context testing conditions.

This improvement is attributed to the use of weighted bounding box fusion and the depth estimator, which enhance the robustness of detection in challenging env...
- **Citations Generated**: ['[paper.pdf, p. 7]', '[paper.pdf, p. 1]']
- **Top Chunk Retrieved**: [paper.pdf, p. 7] (Score: 0.714)
- **Diagnostic Rationale**: 
  - **Partial Semantic Match**: The LLM answer addressed only a subset of the required numerical values or entities.

## Evaluation Methodology & Known Limitations

1. **Deterministic Dense Baseline**: Retrieval scores reflect purely dense FAISS IndexFlatIP similarity with `all-MiniLM-L6-v2`. No lexical (BM25) or hybrid fusion was active.
2. **Sufficiency Heuristic Sensitivity**: The heuristic threshold (`score >= 0.38` and query content word overlap) operates on uncalibrated cosine similarities. Cross-paper queries spanning multiple documents can struggle because a single top chunk cannot span multi-document context.
3. **Local LLM Synthesis**: `llama3.2` executed locally via Ollama with temperature=0.7 (default) and grounded citation prompt.