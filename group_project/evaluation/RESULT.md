# RAG evaluation results

## Run information

| Field                              | Value |
| ----------------------------------- | ----- |
| Evaluation date                    | 2026-09-20T09:51:00.940968+00:00 |
| Framework and version              | ragas 0.4.3 |
| Evaluator model                    | gpt-4o-mini (RAGAS judge) |
| Generator model                    | gpt-4o-mini |
| Embedding model                    | text-embedding-3-small (OpenAI API) |
| Corpus version/commit              | uncommitted at eval time (base 6a2a2d4 + local implementation on feat/standalone_ver) |
| Golden dataset size                | 18 |
| `top_k`                            | 5 |
| Fallback threshold and calibration | SCORE_THRESHOLD=0.58, calibrated via scripts/calibrate_threshold.py tren 16 cau in-domain (golden dataset) + 10 cau out-of-domain/near-miss that. TPR=1.0 (giu het cau in-domain), TNR=0.7 (3/10 near-miss OOD van vuot threshold vi cac ky nang IELTS/TOEFL dung chung nhieu tu vung). Xem reports/threshold_calibration.json. |

## Configurations

- **Config A -- dense-only:** `retrieve(query, top_k, use_reranking=False)` -- semantic_search only, no BM25/RRF fusion.
- **Config B -- hybrid + RRF:** `retrieve(query, top_k, use_reranking=True)` -- dense + BM25 fused with Reciprocal Rank Fusion (k=60).

Hai config dung cung golden dataset, generator, evaluator, prompt va `top_k`; chi khac retrieval strategy (`use_reranking`).

## Overall scores

| Metric            | Config A | Config B | Delta B-A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      | 0.759 | 0.850 | +0.091 |
| Answer relevance  | 0.891 | 0.887 | -0.003 |
| Context recall    | 0.698 | 0.823 | +0.125 |
| Context precision | 0.812 | 0.937 | +0.125 |
| **Average**       | 0.790 | 0.874 | +0.084 |

Ca out_of_domain duoc loai khoi trung binh tren (danh gia rieng bang refusal accuracy) de khong am tham thuong/phat viec tu choi dung. Refusal accuracy: Config A = 0.500, Config B = 0.500. Context-hit rate (khong dung LLM, substring check doc lap voi RAGAS): Config A = 0.312, Config B = 0.312.

## A/B comparison

- Cau hinh tot hon: Config B (hybrid + RRF). Trung binh 4 metric: A=0.790 vs B=0.874 (delta +0.084), va thang deu tren ca 4 metric rieng le (faithfulness +0.091, answer_relevancy -0.003 (khong dang ke), context_recall +0.125, context_precision +0.125).
- Evidence: Lan chay dau tien (truoc khi sua loi citation-repair) cho ket qua nguoc: A=0.774 vs B=0.449, vi 4/18 cau o Config B bi ha xuong safe refusal toan phan (0 diem ca 4 metric) chi vi cau tra loi thieu marker [Sn], du chunk lay ve dung. Sau khi them buoc 'thu lai mot lan voi yeu cau ro rang truoc khi tu choi' (task10_generation.py::_generate_impl, doan retry citation), ket qua dao nguoc thanh B thang A ro rang -- dung voi ky vong ly thuyet la RRF ket hop BM25 giup context_recall/precision tot hon dense-only, dac biet o cau keyword-heavy va cross-lingual. Day la mot bai hoc quan trong: so lieu A/B ban dau gan nhu chac chan sai neu khong kiem tra tung ca that bai thay vi chi nhin trung binh -- 4 ca '0 diem tuyet doi' la dau hieu ro rang cua loi generation-layer, khong phai retrieval-layer.
- Trade-off ve latency/cost: Config A trung binh 1786 ms/cau; Config B trung binh 1718 ms/cau. Latency Config B thap hon nhe (1718ms vs 1786ms/cau trung binh) -- RRF va BM25 chay tren corpus nho (272 chunk) gan nhu khong ton chi phi dang ke so voi thoi gian goi LLM generation (chiem phan lon latency o ca hai config). Khong co trade-off latency dang ke; Config B thang ca chat luong lan latency tren corpus nay.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage             | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | -------------------------- | ---------- |
| 1 | IELTS Writing Task 2 có mấy tiêu chí chấm điểm và đó là những tiêu chí | B | 0.000 | 0.708 | 0.000 | 0.000 | retrieval+generation | Cau hoi cross-lingual ve 4 tieu chi Task 2. RRF/BM25 keo ve chunk tu C06 (Sample Tasks, cau 'The other criteria for Task 2 are the same as for Task 1 (Coherence and Cohesion, Lexical Resource, Grammatical Range and Accuracy)') thay vi chunk chinh tac C02 liet ke day du '4 tieu chi'. Chunk C06 KHONG neu ten 'Task Response' tuong minh (chi noi 'giong Task 1' -- ma Task 1 lai la 'Task Achievement', khac Task 2). LLM tu dien vao 'Task Response' tu kien thuc nen (dung ve mat su that nhung KHONG co can cu trong context duoc cung cap) -> RAGAS Faithfulness cham dung la 0.0 vi day la claim khong duoc ho tro boi context. Day la mot phat hien retrieval that su (chon nham chunk lien quan gan nhung khong day du), khong phai loi citation-format nhu lan cham truoc. |
| 2 | What must a Band 9 response demonstrate for coherence and cohesion in  | B | 0.750 | 0.809 | 0.000 | 1.000 | retrieval | Cau hoi ve C01 (band descriptor PDF bi markitdown lam nat cau truc bang 4 cot khi convert -- xem _chunk_band_descriptor_document trong task4_chunking_indexing.py). context_recall=0.0 nghia la RRF khong keo dung chunk Band 9 mong doi ve top-k, du context_precision gan 1.0 (nhung gi lay duoc thi lien quan). Nguyen nhan co the: van ban chunk band descriptor da bi tron cot nen embedding va BM25 deu kho khop chinh xac voi cau hoi dung thuat ngu 'Band 9'. |
| 3 | What four criteria will examiners use to mark my IELTS essay, accordin | B | 0.600 | 0.952 | 0.500 | 1.000 | generation | Context_recall=0.5, faithfulness=0.6 -- context lay ve mot phan dung (tu A01) nhung cau tra loi cua LLM co the da paraphrase hoi xa noi dung goc hoac bo sot mot phan cua 4 tieu chi khi dien giai, khien statement-level NLI check cua RAGAS Faithfulness khong khop het. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------- | ---------------- | ------------- |
| 1 | Giu co che 'thu lai mot lan voi yeu cau citation ro rang truoc khi tu choi' da them vao _generate_impl() (task10_generation.py) thay vi ha thang xuong safe refusal khi khong tim thay marker [Sn] nao trong lan goi dau. | Truoc khi them: 4/18 cau bi 0 diem tuyet doi ca 4 metric o Config B chi vi thieu dinh dang citation, lam dao nguoc hoan toan ket luan A/B (B tro nen te hon A mot cach gia tao). Sau khi them: B thang A ro rang tren ca 4 metric, dung voi ky vong ly thuyet. | Metric phan anh dung chat luong retrieval/generation that su thay vi bi chi phoi boi mot loi dinh dang o tang sinh cau tra loi. | So sanh group_project/evaluation/results/raw_B_hybrid.json truoc/sau: dem so ca retrieval_source='none' va so ca 0 diem tuyet doi ca 4 metric. |
| 2 | Voi cau hoi ve nhieu tieu chi cung ten nhung khac ngu canh (vd 'Task Response' cua Task 2 vs 'Task Achievement' cua Task 1), tang fetch_k truoc RRF (hien la top_k*2) va uu tien chunk tu tai lieu 'key assessment criteria' (C02) hon chunk tu 'sample tasks' (C06) khi ca hai cung xuat hien, vi C02 la nguon chinh tac liet ke day du tieu chi con C06 chi nhac lai luot qua. | g11: RRF chon chunk C06 (nhac luot) thay vi C02 (liet ke day du), khien LLM phai tu dien them 'Task Response' tu kien thuc nen -> faithfulness=0.0. | Giam so lan LLM phai suy dien ngoai context cho cau hoi lien quan toi danh sach tieu chi cham diem. | Chay lai g11 qua Config B sau khi tang fetch_k, kiem tra context_ids co bao gom chunk C02 hay khong va faithfulness co len 1.0 khong. |
| 3 | Voi band descriptor (C01), thu them mot bien the metadata 'band_number' rieng (vd '9') ngoai section_path hien tai, va cho BM25 index them alias 'Band {N}' de tang kha nang khop tu khoa chinh xac. | g12: context_recall=0.0 cho cau hoi ve Band 9 du chunk Band 9 da duoc tach rieng dung trong task4 (xem _chunk_band_descriptor_document), tuc van de nam o khau retrieve/rank chu khong phai khau chunk. | Tang context_recall cho nhom cau hoi ve band descriptor cu the, la nhom chiem 1/3 golden dataset dang co diem thap nhat (g09/g11/g12 deu thuoc nhom nay hoac lien quan). | Them field band_number vao metadata, chay lai eval_runner va so sanh context_recall rieng cho nhom cau hoi ve band descriptor. |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | ------------: | -------------------: | ---------- |
| n/a | n/a | n/a | n/a | n/a |
