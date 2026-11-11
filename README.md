# Azraq Calibrated Integrated Petrophysical + Fluid Model v3.1 — K-5 Extension

This release extends the calibrated Azraq v3 Streamlit workflow from K-1/K-2/K-3 to **K-1/K-2/K-3/K-5**.

## K-5 additions
- Bundled merged K-5 `PEX-HRLA-HNGS-DSI-PPC-SP` LAS.
- Native LAS 2.0 ingestion in the Streamlit app; no extra `lasio` dependency is required.
- K-5 E-log formation framework: Nayil, Amal, Baraka composite, Aradeiba, Bentiu, Abu Gabra, Abu Gabra SQ-1 and Abu Gabra SQ-2.
- K-5 conventional-core calibration: SQ-1 core at 1607.5–1625.5 m (88.11% recovery) and SQ-2 core at 2233–2239.5 m (95% recovery).
- K-5 MDT fluid calibration at 2253.5, 2394.0 and 2516.5 m. These available SQ-2 samples are formation-water dominated with trace/rare oil and no measured API gravity.
- K-5 included in field summary, well interpretation, correlation, ML/QC, sensitivity and downloads.
- K-5 mud-gas module supports `Azraq K-5_Shows Summary.csv` automatically when placed in `data/`, or through the **Shows Summary uploader**.

## K-5 Shows Summary
The app expects the same columns/structure used for the existing Azraq Shows Summary files: show interval, TG, C1, C2, C3, iC4, nC4, iC5, nC5, direct fluorescence and cut fluorescence. Haworth/Pixler ratios and confidence scores are calculated automatically.

If the dedicated K-5 Shows Summary CSV is not bundled, upload it from the sidebar. The rest of the K-5 log/petrophysical and calibration workflow remains available.

## Evidence hierarchy
DST/PVT/fluid samples > calibrated MDT/FMT pressure > calibrated petrophysics > mud-gas ratios > raw log-response screen.

## Important K-5 interpretation note
The K-5 final report documents very heavy/thick/waxy oil in the upper section and oil shows in Bentiu/Abu Gabra. However, the available deep SQ-2 MDT samples are water dominated. The model therefore keeps **mud-gas/log screening** separate from **observed fluid samples** and does not infer one uniform oil type across K-5.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
