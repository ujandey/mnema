# Interactive demos

[Project overview](../README.md) · [Setup](../docs/GETTING_STARTED.md#set-up-the-interactive-demos)

The demos illustrate model interaction and historical telemetry. They are separate from the reproducible research evaluation. Use the repository's package/demo environment with OpenCV and Streamlit installed; the minimal research dependency file does not include them.

## Webcam demonstrator

From the repository root, on a machine with a graphical desktop and webcam:

```bash
uv run python main.py --demo
```

The program opens camera index 0, mirrors the frame, and extracts the central square. It converts that crop to a 28 x 28 grayscale intensity vector. The inset displays frame differences, but **the model receives intensity**, not the displayed difference image or event-camera measurements.

The demo creates a fresh four-class model with a configured 32 KiB FastStore occupancy budget. The same row-accounting issue applies; this is not a total memory bound. Training state is held in memory and is lost when the program exits.

### Controls

Give the OpenCV window keyboard focus before pressing a key.

| Key | Action |
| --- | --- |
| `1`, `2`, `3`, `4` | Train the current frame as one example of the selected object slot |
| `t` | Toggle continuous native inference; initially off |
| `s` | Run sleep consolidation and reset sleep pressure |
| `q` | Quit and release the camera |

Place an object inside the green central crop, provide several labeled views using its slot key, and press `t` to observe predictions. Repeat for other slots. The displayed training counts count key-triggered examples; assigning a slot does not establish category-level one-shot learning.

### Interpreting telemetry

The prediction and confidence describe native inference, which can change adaptive state. The fast/slow ratio is the arbitration weight, and sleep pressure comes from the controller. Confidence is a vote-margin measure, not calibrated certainty.

The current HUD hardcodes `Dense MAC: 0 (Pure Sparse)` even though training and consolidation perform counted dense operations. Its energy display is a cumulative partial projection, not a device measurement or reliably isolated per-frame cost. Use the [instrument API](../docs/API.md#energyinstrument) and research reports for audited interpretation rather than copying HUD labels into benchmark claims.

## Historical-results dashboard

```bash
uv run streamlit run demo/dashboard.py
```

Open the local URL printed by Streamlit. The dashboard reads [results/benchmark_results.json](../results/benchmark_results.json); it neither streams webcam telemetry nor reads `results/research_benchmark/` automatically.

The checked-in file can be used without retraining. To deliberately replace that historical dataset, run `python main.py --benchmark` from an appropriate environment; this overwrites the historical result JSON.

The UI has legacy labels such as `64 KB (Flat)` and an energy-advantage card that do not reflect the research accounting. Its memory label is not an actual total-memory measurement, and its energy comparisons are historical projections. Read the [artifact guide](../results/README.md) for the applicable caveats.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Cannot access webcam | Camera index 0 exists, camera permissions are enabled, and another app is not holding the device |
| Window opens but keys do nothing | Focus the OpenCV window rather than the terminal or browser |
| OpenCV display fails on a server | Use a local graphical session; the demo expects a windowing environment |
| Missing `cv2` or `streamlit` | Use the package/demo dependencies, not only the research requirements |
| Dashboard reports missing results | Run from the root and verify `results/benchmark_results.json` exists |
| Predictions change without labeled training | Native inference still updates adaptive state; see [architecture](../docs/ARCHITECTURE.md#inference-is-stateful) |

For dependency and working-directory issues, see [troubleshooting](../docs/TROUBLESHOOTING.md).
