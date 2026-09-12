# Glossary

[Documentation index](README.md)

| Term | Meaning in this repository |
| --- | --- |
| ACC / final accuracy | Mean accuracy across the five task test sets at the final checkpoint; a task-macro average, reported as a percentage |
| Adaptive state | Persistent model arrays that can change, including weights, synaptic memory variables, traces, and homeostatic state |
| ALIF | Adaptive leaky integrate-and-fire neuron dynamics, used in the shallow output cortex |
| Array payload | Bytes occupied by NumPy array data; excludes Python object headers and process overhead |
| Auxiliary allocated | Full persistent allocation for replay, Fisher/snapshot state, or FastStore arrays, including unused capacity |
| Auxiliary content | Currently occupied auxiliary array payload; an alternative view of allocated capacity, not an extra amount to add |
| Benna-Fusi variables | Four coupled synaptic state planes; the first is the visible weight plane |
| Checkpoint | Model state after training a task. Research evaluation restores its inference-mutable state after each image. It is not a persisted training checkpoint. |
| Class-IL | Class-incremental learning: predict among all ten classes without being told the current task |
| Config ID | Digest of the experiment configuration, including source hashes, data hashes, environment, and protocol settings |
| DER++ | Dark Experience Replay with logit matching and replay-label cross-entropy; the research implementation has a 300-item reservoir |
| EWC | Elastic Weight Consolidation; this implementation stores per-task empirical diagonal Fishers and parameter snapshots |
| FastStore | Dense arrays of sparse-addressed class associations with salience-based eviction |
| Fixed scaffold | Persistent arrays that do not adapt, such as separator connectivity and synaptic constants |
| FM / forgetting | Mean old-task drop from the best post-learning, pre-final checkpoint to the final checkpoint, reported in percentage points |
| KiB | 1,024 bytes; 64 KiB equals 65,536 bytes |
| k-WTA | k winners take all: retain the indices of the k strongest separator responses |
| MAC | Multiply-accumulate counter category; some multiply/divide-class training operations are also charged to this category |
| Native inference | The model's own prediction path, which changes adaptive state even with `is_training=False` |
| pp | Percentage points: the difference between two percentages |
| Projected energy | Counter totals multiplied by technology-card coefficients; not physical energy measured on a device |
| Replay / reservoir | Storage and later training on earlier examples; reservoir sampling keeps a fixed-capacity sample of the incoming stream |
| Retention matrix `R[t, j]` | Accuracy on task `j` after training task `t`; the research matrix measures all 25 cells |
| SD | Sample standard deviation across seeds (`ddof=1`); undefined for a single seed |
| Seed-method pair | One independent model initialization and five-task training/evaluation run; the unit of persisted work |
| SynOp | Sparse synaptic-operation counter, incremented for active input-to-cortex connections |
| TTFS | Time to first spike; the encoder creates a raster, then the current separator collapses it to input activity |

For exact metric formulas, see the [research protocol](RESEARCH_BENCHMARK.md#metrics). For behavior behind the terms, see the [architecture](ARCHITECTURE.md).
