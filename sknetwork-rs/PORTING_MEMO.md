# Rust Porting Memo (Sknetwork-rs)

Use this checklist before asking for a test run.

## Executive Dashboard

- [x] **Done**: Deep parity audit completed (Waves A-D), align-now pass implemented, export policy standardized, not-fitted contract standardized across hierarchy/regression/linkpred, second audit (module-by-module) completed, topology perf hardening for `cliques`/`core`, embedding KD-tree/per-iteration allocation hardening, `load_netset` remote fetch path implemented, Paris edge-case parity hardening tests added, and docs standardization completed.
- [x] **Done**: Audit 4 memo-first scan completed (no-code): P1/P2 backlog re-ranked, consistency drift reviewed, remediation batches prepared.
- [ ] **Pending**: implement Audit 4 remediation batches and re-run full validation gates.
- [x] **Discuss**: Lanczos / ARPACK — **defer ARPACK binding**; pure-Rust `q₀` path is quality-green; see [`docs/ARPACK_DECISION.md`](docs/ARPACK_DECISION.md).

## Current Action Backlog

### Audit 4 Backlog (Memo-First, Re-ranked)

#### P0 align-now blockers

- None currently.

#### P1 align-now (parity and contract)

1. `src/gnn/neighbor_sampler.rs::UniformNeighborSampler::sample`:
   - Aligned: now performs stochastic uniform row-wise neighbor sampling, supports optional deterministic seed (`random_state`) for reproducibility, and binarizes sampled edges to Python-like semantics.

#### Recently confirmed aligned in Audit 4 scan

- No new parity regressions found in previously remediated P1 files from audits 2/3.
- Global consistency choices remain coherent with current code:
  - curated export policy remains in place (`classification::knn` alias is explicit, no wildcard drift),
  - unsupported options continue to fail with typed errors in audited modules,
  - constructor string normalization is still consistently applied where introduced.

#### P1/P2 global package consistency (new third-audit dimension)

1. Error-contract consistency:
   - Aligned (current sweep scope): expanded contract matrix now covers representative estimators across classification/clustering/embedding/hierarchy/gnn/ranking/regression, and no additional drift surfaced in this pass.
2. Backend-policy consistency:
   - Aligned (current sweep scope): silent unsupported-option fallbacks were removed for `clustering/propagation_clustering`, `classification/propagation`, `embedding/louvain_embedding`, modularity-dependent `clustering/louvain`/`clustering/leiden`, `embedding/spectral` decomposition selection, and `embedding/spring` position initialization.
3. Export-surface consistency:
   - Aligned: wildcard alias drift removed in `classification::knn`; `data::test_graphs` is now test-only.
4. Parameter/default consistency:
   - Aligned (current sweep scope): constructor option semantics normalized across `ranking/closeness`, `embedding/spectral`, `clustering/propagation_clustering`, `embedding/louvain_embedding`, `embedding/spring`, `hierarchy/paris`, `classification/propagation`, and `linalg/eig_solver`.

#### P2 staged perf/idiomatic improvements

1. `src/embedding/{spring,force_atlas}.rs`:
   - KD-tree construction still occurs every iteration when radius approximation is active.
   - Decision: Stage as P2 benchmark-backed optimization.
2. `src/classification/nn.rs`:
   - Default no-embedding path still densifies sparse adjacency into `Vec<Vec<f64>>`.
   - Decision: Stage as P2 sparse-first refactor (parity-safe if outputs are preserved).
3. `src/data/parse.rs::from_graphml`:
   - Parser remains manual-string based and potentially brittle across wider GraphML variants.
   - Decision: Stage as P2 robustness hardening with targeted corpus tests.

### P3 cleanup/doc follow-ups

1. Keep interim explicit gaps synchronized with actual behavior (notably eig/svd semantics as they evolve).
2. Add parity-test checklist per module before final remediation closes.
3. Add a large synthetic CSV regression test for `src/data/parse.rs::from_csv` streaming path (deferred; ensure no reintroduction of full row buffering).
4. Extend compact contract test matrix coverage from representative estimators to all public estimators (`fit`/`predict`/`transform` before/after fit, invalid params, unsupported options).
5. Embedding micro-optimizations to consider later: contiguous position storage and/or incremental spatial index updates to reduce per-iteration KD-tree rebuild cost further.
6. Prepare a reproducible benchmark campaign for solver backends (`arpack-ng`, `svdlibrs`, current in-house iterative solvers): compare accuracy (residuals/subspace error), convergence robustness, runtime, and build/deployment complexity on representative sparse graphs.

## Open Decisions

- None currently.

## Explicit Accepted Divergences (current, temporary)

- `linalg/eig_solver`: still uses a simplified block-iteration proxy; pending migration to `symmetric_eigsh` (IRLM, ARPACK-equivalent).
- `linalg/svd_solver`: **2026-06-10** — `LanczosSVD` follows the SciPy ``svds(solver='arpack')`` path (Gram ``eigsh`` + QR + thin Rayleigh–Ritz SVD). Matrix-free IRLM in ``symmetric_eigsh``: Simon/Parlett selective reorthogonalization (converged-Ritz SO + partial reorth), true relative residuals, Ritz locking, sorted implicit shifts. Defaults: ``choose_ncv(k)=max(2k+1,20)``, ``maxiter=10n``, ``tol=0→1e-10``, ``v0~N(0,1)`` via ``random_state``. **2026-06-10** — `RandomizedSVD` (Halko / ``sklearn.utils.extmath.randomized_svd``): range finding + economic QR + thin SVD; defaults ``n_oversamples=10``, ``n_iter`` auto (4/7), ``transpose`` auto, ``flip_sign``; matrix-free on ``SVDInput``.
- `embedding/svd`: **2026-06-10** — ``GSVD``/``SVD``/``PCA`` use ``fit_partial_svd`` (``LanczosSVD`` default, ``RandomizedSVD`` via ``SvdSolverKind::Halko``). Replaced prior block power-iteration placeholder in ``GSVD::fit``.
- `linkpred/nn` (**benchmark export**): neighbor **columns** may differ when top-k cosine similarities **tie** (NumPy `argpartition` vs Rust `select_nth_unstable_by`). Export parity uses **`linkpred_row_weights`** (per-row sorted weights), not exact `(row, col, weight)` triples. See **Link prediction** section below.
- `hierarchy/{paris,louvain_iteration}` (**benchmark export**): dendrogram **merge matrices differ** in merge order and heights even when trees are similar; export uses **`dasgupta_score`** quality comparison (Python metric on each side's tree), not row-wise dendrogram parity. Rust Paris may emit **NaN merge heights** on large graphs (JSON `null`); Python `dasgupta_score` treats those as NaN. See **Hierarchy** section below.
- `gnn/gnn_classifier` (**benchmark export**): predicted **label vectors differ** even when **`gnn_accuracy`** passes (loose atol); export compares accuracy vs ground truth, not `int_vector_exact`. See **GNN** section below.

## Topology: `count_cliques` on directed graphs (2026-06-11)

Cross-runtime checks in `benchmarking/run_topology_export.py` (Python `sknetwork.topology` vs `benchmark_ipc` `topology_run`) now **pass on Tier S+M** including **polblogs** clique rows (`2026-06-11` export, **33/33** pass).

**Root cause (fixed):** Python builds the listing DAG with ``get_dag(adjacency, order=np.argsort(core))`` where ``order[i]`` is the *i*-th entry of ``argsort(core)`` (not per-node rank). Rust initially used rank-based order and stable sort, which diverged on tied core values.

**Fix:**

- `src/topology/cliques.rs` — DAG order matches Python's ``argsort`` layout; unstable sort when computing order locally.
- Export — Rust IPC `count_cliques` accepts optional **`core_order`** (Python `np.argsort(get_core_decomposition(graph))`) so NumPy quicksort tie-breaking matches on digraphs.

**Results snapshot (2026-06-11):** polblogs `cliques_k2` / `cliques_k3` **Pass** (9903 / 30357); citeseer and karate clique rows **Pass**. See `benchmarking/topology/COMPARISON.md`.

**Library note:** standalone Rust `count_cliques` without `core_order` may still differ slightly from Python on graphs with many tied core numbers; pass `core_order` for bit-exact parity.

## Path: Python `breadth_first_search` vs Rust BFS visit order (convention gap to tighten)

**Python (`sknetwork.path.breadth_first_search`):** implementation is *not* a classical queue walk. It computes hop distances (`get_distances`), applies **`numpy.argsort(distances)`**, drops indices for unreachable nodes (`distances < 0`), and returns the remaining order. For nodes at the **same distance**, order is therefore whatever **NumPy’s `argsort` tie-breaking** produces for the installed NumPy version (not the same as sorting by `(distance, node_index)` and not the same as neighbor-discovery order along outgoing CSR edges).

**Rust (`src/path/search.rs::breadth_first_search`):** standard **FIFO queue** traversal: when a node is dequeued, neighbors are visited in **CSR row iteration order**, so the visit sequence is determined by the sparse storage order of the adjacency.

**Benchmark / IPC note:** cross-runtime parity checks that compared a stable `(distance, node_id)` sort to Python’s output **failed on real digraphs** (e.g. polblogs): almost all positions in the returned index vector disagreed with NumPy’s order. Replicating Python exactly would require matching **NumPy’s sorting algorithm and tie rules** in Rust (or calling into NumPy), which is brittle across versions.

**Distances vs visit order:** the **hop-count vector** from BFS (`get_distances` / multi-source Dijkstra-style layering on unweighted graphs) is **not** ambiguous for a fixed adjacency and source set: each node’s distance is the minimum number of hops from any source. Python and Rust agree on those values (see `path_run` op `get_distances` in exports). What differs is only the **ordering** of nodes in `breadth_first_search`, which Python derives from **`numpy.argsort`** on that same vector. So parity on **lengths** is already the right check; omitting `breadth_first_search` from IPC avoids comparing **permutations** that carry no extra distance information.

**Interim policy:** `benchmarking/run_path_export.py` and `benchmark_ipc` **`path_run`** benchmark **`get_distances`** and **`get_shortest_path`** only; **`breadth_first_search` is omitted** from IPC until the contract is clarified.

**Tighten next (pick one direction):** (1) **Document** Python behavior as explicitly NumPy-`argsort`-dependent and keep Rust as queue-BFS, with no parity claim between the two for visit order; (2) **Change Python** to a specified order (e.g. stable sort by `(distance, node)` or true queue BFS) and align Rust tests + docs; (3) **Add a dedicated parity surface** (e.g. Rust bench-only helper that mirrors `argsort` via a pinned NumPy reference or a version-tested port) if visit-order parity is required for exports.

## Clustering: partition labels on M-tier graphs (benchmark gap)

Cross-runtime checks in `benchmarking/run_clustering_export.py` compare **Louvain** and **Leiden** via **modularity** (`parity.metric_type` = `modularity_score`, `validation_mode` = `output_quality`): each side’s partition is scored with `get_modularity` (`weights='degree'`, run `resolution`). **PropagationClustering** and **KCenters** still use **canonical partition labels** (`canonical_partition_labels`).

- **Tier S (`karate`):** Louvain/Leiden modularity scores **match** within tolerance; propagation / k-centers label parity **pass**.
- **Tier M (`polblogs`, `citeseer`):** Louvain/Leiden **modularity scores** are typically within **~1e-4** of Python on the same graph (see export JSON `python_quality` / `rust_quality`); **partitions** still differ (propagation / k-centers label parity fails). Treat remaining Q gaps as near-optima / scoring drift, not order-of-magnitude quality regressions.

**Tighten next:** align modularity objective / local moves so M-tier Q scores match, or adopt a looser quality tolerance if near-optima differ; keep label parity only where the algorithm contract requires identical partitions.

## Link prediction: `NNLinker` benchmark (2026-06-10)

Cross-runtime checks in `benchmarking/run_linkpred_export.py` (Python `sknetwork.linkpred.NNLinker` vs `benchmark_ipc` **`linkpred_run`**, algorithm **`nn`**) are **green on Tier S+M** under the export parity rule below.

**What is benchmarked today**

- Estimator: **`NNLinker`** with **identity embedding** (adjacency rows L2-normalized as features; Python `embedding_method=None`).
- Parameter sets: `nn_default` (`n_neighbors=10`, `threshold=0.0`) and `nn_tuned` (`n_neighbors=5`, `threshold=0.5`).
- Datasets: Tier **S** `karate`, Tier **M** `polblogs`, `citeseer` (square adjacency via `load_netset`).
- Timing: graph load untimed; timed Rust IPC sets **`return_links=false`** (shape/`nnz` only) so edge JSON does not dominate — same pattern as embedding `return_embedding=false`. One **untimed** call with `return_links=true` drives parity.

**Export parity rule: `linkpred_row_weights`**

Do **not** use `sorted_csr_edges` or `stored_csr_edges` for link-prediction outputs:

1. **Explicit zero weights:** Python stores below-threshold top-k slots as **zero entries** in CSR (`links.nnz` counts them; `scipy.sparse.csr_matrix.nonzero()` does **not**). A strict edge-list compare that skips zeros will report a false `nnz` mismatch even when algorithms agree.
2. **Top-k tie-breaking:** For a fixed source row, Python and Rust pick the same **multiset of cosine similarities** (same count, same sorted weights within atol/rtol), but **neighbor column indices** can differ when similarities tie (e.g. karate row 4: weight `≈0.289` to node **7** in Python vs node **0** in Rust). NumPy `argpartition` and Rust `select_nth_unstable_by` do not share tie policy.

**Interim policy:** parity passes when, for every source row, predicted **degree** matches and **sorted similarity weights** match (`atol=rtol=1e-5`). Column identity is **not** asserted.

**Results snapshot (2026-06-10):** all six export rows **Pass**; Rust IPC timings much faster on M-tier (Python ~0.4–1.0 s vs Rust ~10–33 ms per fit on polblogs/citeseer — identity-embedding sparse dot products).

**Not in export scope yet**

- **`Spectral` embedding** (`embedding_method=Spectral(...)` / Rust `EmbeddingMethod::Spectral`) — wired in `src/bench/linkpred.rs` but no export row; add when spectral embedding parity on link outputs is understood.
- **Partial source set** (`index` / seed mask) and **bipartite** biadjacency — not swept (catalog uses square netsets, all nodes predicted).
- **Exact CSR edge parity** including columns on ties — deferred unless Rust adopts NumPy-compatible tie-breaking or Python documents a stable `(−sim, col)` sort.

**Contract gaps to remember (not triggered by current params)**

- `check_n_neighbors`: Python caps to **`n_seeds - 1`** when `n_neighbors >= n_seeds`; Rust uses **`min(n_neighbors, n_seeds)`** (`src/utils/check.rs`). Current export uses `n_neighbors ∈ {5, 10}` on graphs with hundreds–thousands of nodes, so both sides keep the requested k.
- Revisit if benchmarking `n_neighbors` near graph order or `n_neighbors=None` (all nodes).

**Tighten next (optional):** (1) deterministic shared tie-break `(-weight, col)` on both sides and switch export to `stored_csr_edges`; (2) align `check_n_neighbors` with Python; (3) add `nn_spectral` export row; (4) document bipartite column indexing in parity if biadjacency graphs are added to the catalog.

## Hierarchy: Paris / LouvainIteration benchmark (2026-06-10)

Cross-runtime checks in `benchmarking/run_hierarchy_export.py` (Python `sknetwork.hierarchy` vs `benchmark_ipc` **`hierarchy_run`**) compare **Dasgupta quality** of each side's dendrogram, not exact merge matrices.

**What is benchmarked today**

- Estimators: **Paris** (`weights='degree'`, `reorder=True`) and **LouvainIteration** (`depth=3`, Louvain tolerances aligned with clustering export, `shuffle_nodes=False`).
- Datasets: Tier **S** `karate`, Tier **M** `polblogs`, `citeseer`.
- Timing: graph load untimed; timed Rust IPC sets **`return_dendrogram=false`** (`n_merges` only). One untimed call with `return_dendrogram=true` drives validation.

**Export validation rule: `dasgupta_score`**

- Python and Rust **dendrogram matrices differ** (merge order, heights) even on karate — do **not** use `dendrogram_l1` parity for Paris.
- Pass when |score_py − score_rs| ≤ **1e-2** or relative difference ≤ **2.5e-2** (`parity_dasgupta_score` defaults), scoring each tree with Python `dasgupta_score` (`weights='degree'` unless noted).
- Rust Paris on large graphs can produce **NaN merge heights** (serde JSON `null`); metrics map `null` → NaN for scoring.

**Results snapshot (2026-06-10):** all six export rows **Pass** under the quality rule above; Rust IPC timings are much faster on M-tier (Paris ~0.3–1 ms vs Python ~0.8–3 s on polblogs/citeseer).

**Not in export scope yet**

- **LouvainHierarchy** (bottom-up full Louvain tree).
- **Bipartite** / partial `index` fits.
- Exact dendrogram matrix parity (deferred unless Paris merge schedule is aligned).

## GNN: GNNClassifier benchmark (2026-06-10, Tier S + M)

Cross-runtime checks in `benchmarking/run_gnn_export.py` (Python `sknetwork.gnn.GNNClassifier` vs `benchmark_ipc` **`gnn_run`**) compare **classification accuracy** vs ground truth, not exact label vectors.

**What is benchmarked**

- Estimator: **GNNClassifier**; features = dense adjacency (`n_nodes × n_nodes`).
- **Tier S:** **karate** — semi-supervised labels on nodes **`0`**, **`1`**, **`33`**.
- **Tier M:** **polblogs**, **citeseer** — **two** labeled seeds per ground-truth class; `dims = [n, 8, n_classes]`.
- Parameter sets: `gcn_conv_default` (GCN / Conv, Adam, **10 epochs**) and `sage_small` (SAGE, GD, **2 epochs on S / 20 on M**).
- Timing: graph load untimed; timed Rust IPC sets **`return_labels=false`**. Features and training labels are passed in each `gnn_run` params payload (not cached in IPC state). M-tier JSON payloads are large (`n_nodes²` floats per IPC call).

**Export validation rule: `gnn_accuracy`**

- Score each side's predictions with Python `get_accuracy_score` against full ground-truth labels.
- Pass when |acc_py − acc_rs| ≤ **0.15** or relative difference ≤ **0.20**.
- Do **not** use `int_vector_exact` on predicted labels until weight init, normalization, and backprop are fully aligned.

**Results snapshot (2026-06-11):** all **6** export rows **Pass** after full backprop + M-tier tuning (`benchmarking/results/runs/2026-06-11/gnn/`).

| Tier | Dataset | Benchmark | Parity | Python acc | Rust acc |
|------|---------|-----------|--------|------------|----------|
| S | karate | `gcn_conv_default` | Pass | ≈0.94 | ≈0.85 |
| S | karate | `sage_small` | Pass | ≈0.97 | ≈0.94 |
| M | citeseer | `gcn_conv_default` | Pass | ≈0.25 | ≈0.17 |
| M | citeseer | `sage_small` | Pass | ≈0.20 | ≈0.20 |
| M | polblogs | `gcn_conv_default` | Pass | ≈0.64 | ≈0.68 |
| M | polblogs | `sage_small` | Pass | ≈0.75 | ≈0.61 |

**Rust training (2026-06-11):** `GNNClassifier::fit` now mirrors Python `BaseGNN.backward` — masked pre-softmax CE gradient, ReLU backprop through hidden layers, per-layer Adam/GD (`MultiLayerAdam` / `MultiLayerGD` in `src/gnn/optimizer.rs`). Layer forward stores pre-activation embeddings via `Convolution::forward_with_embedding`.

### Residual accuracy gaps (export passes, models still differ)

All six rows **pass** `gnn_accuracy` (atol **0.15**), but Python and Rust **accuracies and label vectors are not identical**. Typical deltas from the **2026-06-11** run:

| Tier | Dataset | Benchmark | Python acc | Rust acc | Δ (py − rs) |
|------|---------|-----------|------------|----------|-------------|
| S | karate | GCN | ≈0.94 | ≈0.85 | +0.09 |
| S | karate | SAGE | ≈0.97 | ≈0.94 | +0.03 |
| M | polblogs | GCN | ≈0.64 | ≈0.68 | −0.04 |
| M | polblogs | SAGE | ≈0.75 | ≈0.61 | **+0.15** (at tolerance edge) |
| M | citeseer | GCN | ≈0.25 | ≈0.17 | +0.08 |
| M | citeseer | SAGE | ≈0.20 | ≈0.20 | ≈0 |

**Label agreement** (fraction of nodes with the same predicted class) can be much lower than accuracy agreement — e.g. citeseer SAGE ~20% accuracy with only ~15–25% label overlap between sides. The metric accepts divergent predictions when absolute accuracies are both low.

**Likely causes (priority order for future tightening):**

1. **Loose parity contract** — `gnn_accuracy` allows up to **15 percentage points** (or 20% relative) difference; polblogs SAGE sits at the boundary. Tighter atol or a secondary **label-agreement** check would surface remaining drift.
2. **Weight-init RNG** — Python He init uses `np.random.randn` (global NumPy RNG); Rust uses Box–Muller + `StdRng`. Same `random_state` does **not** guarantee identical initial weights → different decision boundaries after training.
3. **SAGE sampling RNG** — `UniformNeighborSampler` draws once per fit; Python uses `np.random.choice` after `np.random.seed`; Rust uses `StdRng`. Subgraph samples can differ if RNG streams diverge (especially on large directed graphs like **polblogs**).
4. **Floating-point / matmul** — Python SciPy sparse×dense vs Rust manual CSR loops; accumulation order drifts over 10–20 epochs.
5. **Semi-supervised fragility** — M-tier uses only **two seeds per class**; models memorize seeds quickly and generalization is noisy — small training differences swing test accuracy.
6. **Low absolute accuracy (citeseer)** — ~20% on 6 classes is near random; passes are weak (both sides are poor classifiers, not aligned ones).
7. **Convergence rate** — Rust GD on large hidden layers (e.g. polblogs **1490×8**) may need more than 20 SAGE epochs to match Python; karate saturates at 2 epochs.
8. **Not yet ported** — early stopping / validation splits; fused softmax–CE numerics (Python clips at 1e-10 in loss); per-epoch SAGE resampling (neither side resamples today).

**Follow-ups (deferred):**

- Align NumPy and Rust RNG for **He init + SAGE sampling** (same weight matrix and sampled adjacency from one seed).
- Report **label agreement** alongside accuracy in export JSON / `COMPARISON.md`.
- Consider tier-specific SAGE epoch counts or shared convergence criterion.
- Tighten atol only after RNG alignment (otherwise false failures).

**Not in export scope yet**

- Tier **L**.
- Early stopping, validation splits, partial-label masking differences.
- Exact label parity or embedding/probability L1 checks.
- Binary/sparse feature IPC to avoid `n_nodes²` JSON on large graphs.

## Data: Tier S `karate` netset (fixed)

**Tier S** `karate` now uses a canonical **34-node** Zachary club bundle (`adjacency.npz` + `labels.npy`) under `benchmarking/datasets/bundles/karate/`, installed into `SCIKIT_NETWORK_DATA/netset/karate/` by `benchmarking/lib/seed_netset.py`. Export scripts call `ensure_karate_netset()` at startup; stale `adjacency.tsv` stubs are removed on seed.

- **Python:** `load_netset('karate')` and `load_labeled_square_graph` (classification).
- **Rust:** `benchmark_ipc` `load_netset` reads the same NPZ adjacency.

Do **not** add a standalone `adjacency.tsv` for karate — Python ignores it; Rust would prefer TSV over NPZ.

**Re-run status:** Tier S **embedding** and **clustering** were refreshed on **2026-06-10** (34-node karate). Older **2026-05-12** S-tier artifacts should be ignored for those modules.

## Data I/O: native NPZ/NPY via `npyz`

**Naming (Rust):** fast on-disk reload can use **CSR text** via ``save`` / ``load`` / ``save_csr_bundle`` / ``load_csr_bundle`` (``adjacency.tsv``, …). Do **not** confuse with Python ``save_to_numpy_bundle`` (real ``.npz``). Old Rust names ``save_to_numpy_bundle`` / ``load_from_numpy_bundle`` were removed as misleading.

``load_npz_matrix`` / ``load_npy_labels`` in ``src/data/load.rs`` use the **`npyz`** crate (native SciPy CSR ``.npz`` and ``.npy`` readers). This replaced the earlier ``python3 -c …`` subprocess bridge (~**0.5–0.8 s** per NetSet load on karate).

**Load tuning (Rust):**

- ``LoadOptions::adjacency_only`` — skip labels/names sidecars (algorithm IPC uses this).
- ``LoadOptions::materialize_csr`` — after NPZ load, write ``adjacency.tsv`` beside the bundle; subsequent loads read TSV first, then still merge ``.npy`` sidecars.
- ``load_dataset_folder`` — hybrid TSV matrix + NPZ/NPY sidecars.
- ``data_io`` benchmarks export **full** loads, **``matrix_only``** rows (adjacency-only, no edge JSON), and **``load_csr_folder``** (TSV fast path).

## Optimization Ideas (not Python divergences)

- `embedding/spring` and `embedding/force_atlas`: borrowed KD-tree build and neighbor-buffer reuse are implemented; remaining optimization is reducing full KD-tree rebuild cost per iteration.
- `embedding/spring` and `embedding/force_atlas`: high-dimensional radius-neighborhood fallback still uses brute-force scans; evaluate stronger spatial indexing only if profiling justifies it.
- Layout-engine internals: consider contiguous position storage to improve cache locality and reduce allocation overhead in hot loops.
- `kcenters`: current PageRank-based path is parity-aligned and avoids all-pairs materialization; only profile-driven micro-optimizations remain.
- `hierarchy/paris`: core edge-case parity suite is now covered; remaining work is optional large-scale numerical regression benchmarking against Python outputs.

## Audit 4 Snapshot (Memo-First, No-Code)

- Scope executed: strict no-code audit in four waves (P1 closure, P2 triage, consistency drift check, remediation blueprint).
- Outcome:
  - The previously remaining P1 parity gap (`gnn/neighbor_sampler` stochastic+binarized behavior) is now aligned.
  - No newly introduced P0/P1 regressions found in previously aligned modules after implementation validation.
  - P2 candidates re-ranked by expected gain vs implementation risk.
- Decision policy applied for all reviewed items: `Align now`, `Explicit gap`, or `Accept divergence`.

### Audit 4 Remediation Batches (implementation phase)

1. **Batch 1 (P1 parity/contracts, highest priority)**
   - `src/gnn/neighbor_sampler.rs::UniformNeighborSampler::sample`:
     - implemented stochastic uniform row-wise sampling,
     - implemented binarized sampled-edge semantics,
     - implemented deterministic-seed mode and parity-oriented tests.

2. **Batch 2 (P2 quick wins, low-risk)**
   - `src/classification/nn.rs`:
     - Aligned (quick win): default `embedding_method=None` path now avoids full sparse-to-dense materialization and performs sparse row-distance KNN directly.
     - inference contracts and output semantics preserved (test suite green).
   - `src/embedding/{spring,force_atlas}.rs`:
     - Aligned (quick win): avoid KD-tree construction on small graphs via node-count gating (`n >= 128`) while preserving existing radius-query behavior.
     - precompute radius-squared constant in ForceAtlas hot loop to reduce repeated per-neighbor arithmetic.

3. **Batch 3 (P2/P3 benchmark- and corpus-driven)**
   - `src/data/parse.rs::from_graphml`:
     - Aligned (robustness hardening): parser now accepts both single- and double-quoted XML attributes and supports self-closing `<edge .../>` tags.
     - Added regression coverage for mixed GraphML variants (single quotes + self-closing edge + weighted data edge).
   - `src/linalg/{eig_solver,svd_solver}.rs` benchmark campaign:
     - Deferred by decision: benchmarking will be handled in a separate project spanning the entire Rust codebase (not inside this repository track).

### Validation Gates Per Batch

- Contract/API: extend `src/contract_matrix.rs` where applicable (pre-fit/post-fit/invalid-option checks).
- Correctness: targeted unit tests for each changed module and parity-focused edge cases.
- Global checks: `cargo test -q` and `cargo doc -q --no-deps`.
- Current status:
  - `cargo test -q`: green after Batch 1 and Batch 2 quick-win implementations.

## Second Audit Snapshot

- Scope executed: full module-by-module reread in waves (A/B/C/D), parity + idiomatic + performance.
- Outcome: no new P0 blockers; multiple new P1/P2 align-now findings identified and logged.
- Status: active backlog reopened with concrete parity/API/perf tasks below.
- Verification basis: full suite currently green (`196 passed`), findings are audit-driven risk items rather than failing-test regressions.

## Second Audit Findings (module-by-module)

### Wave A (data, linalg, clustering, hierarchy, gnn)

- `[P1][data] src/data/parse.rs::from_edge_array`
  - Negative ids are cast to `usize` in non-reindex paths.
  - Decision: Align now.
- `[P1][data] src/data/parse.rs::from_graphml`
  - Manual XML parsing remains fragile vs valid GraphML variants.
  - Decision: Align now.
- `[P1][data] src/data/load.rs::load_netset`
  - Remote retrieval path was missing (local-cache behavior only).
  - Decision: Aligned (implemented remote download + safe extraction + npz/npy parsing, with explicit pickle unsupported error).
- `[P1][linalg] src/linalg/eig_solver.rs::LanczosEig::fit`
  - `which` semantics do not fully match smallest-eigen behavior expectations.
  - Decision: Align now.
- `[P1][linalg] src/linalg/ppr_solver.rs` solver operator consistency
  - Backend operator semantics may diverge across backends.
  - Decision: Align now.
- `[P1][clustering] src/clustering/kcenters.rs::KCenters::fit`
  - Re-checked against Python: current behavior (`directed` triggers undirectization) matches Python contract.
  - Decision: No action (audit false positive).
- `[P1][clustering] src/clustering/louvain.rs::run_louvain`
  - Directed modularity path may be over-symmetrized.
  - Decision: Align now.
- `[P1][hierarchy] src/hierarchy/postprocess.rs::check_dendrogram`
  - Validation is minimal; malformed dendrograms can pass too far downstream.
  - Decision: Align now.
- `[P1][hierarchy] src/hierarchy/paris.rs::Paris::fit`
  - Edge-case numerical behavior needed hardening coverage (disconnected components, isolates/null-weights, tie cases, bipartite split consistency).
  - Decision: Aligned (implemented targeted edge-case parity test suite and validated stability).
- `[P1][gnn] src/gnn/base.rs` inference not-fitted gating
  - Predict/transform can run without explicit fitted-state guard.
  - Decision: Align now.
- `[P2][gnn] src/gnn/gnn_classifier.rs::fit`
  - Training failures are not propagated as `Result`.
  - Decision: Align now.

### Wave B (path, topology, ranking, embedding)

- `[P1][topology] src/topology/cycles.rs::break_cycles`
  - Root validity checks weaker than Python expectations.
  - Decision: Align now.
- `[P2][topology] src/topology/cliques.rs::count_from_candidates`
  - Recursive candidate intersections allocated per branch and per depth.
  - Decision: Aligned (implemented reusable listing-state recursion; removed allocation-heavy per-branch intersections).
- `[P2][topology] src/topology/core.rs::get_core_decomposition`
  - Naive repeated min-degree scans were O(n^2) in node count.
  - Decision: Aligned (implemented bucket/peeling decomposition; removed global min scans).
- `[P1][ranking] src/ranking/katz.rs::Katz::fit`
  - Weighted-vs-bool adjacency semantics may diverge from Python behavior.
  - Decision: Align now.
- `[P1][ranking] src/ranking/betweenness.rs::Betweenness::fit`
  - Disconnected-graph handling was previously more permissive than Python.
  - Decision: Aligned (implemented explicit disconnected-graph error).
- `[P1][embedding] src/embedding/svd.rs::GSVD::predict`
  - Projection pipeline omits parts of Python weighting/scaling contract.
  - Decision: Align now.
- `[P1][embedding] src/embedding/svd.rs::PCA::fit`
  - PCA centering semantics differ from Python SparseLR-centered path.
  - Decision: Align now.
- `[P1][embedding] src/embedding/spring.rs::Spring::fit`
  - Init/force dynamics differ materially from Python behavior.
  - Decision: Align now.
- `[P1][embedding] src/embedding/force_atlas.rs::ForceAtlas::fit`
  - Dynamics/tolerance/update-order profile previously diverged from Python.
  - Decision: Aligned (implemented Python-style update order, node swing-based speed, speed cap by resultant norm, size-based tolerance policy, and random-normal initialization).
- `[P2][embedding] src/embedding/{spring,force_atlas}.rs`
  - Per-node/per-iteration temporary neighbor allocations and repeated degree scans increased layout overhead.
  - Decision: Aligned (implemented neighbor-buffer reuse for radius lookups, removed full-range neighbor `Vec` allocations, and precomputed degree terms in ForceAtlas).

### Wave C (utils, classification, regression, linkpred, visualization)

- `[P2][utils] src/utils/neighbors.rs::get_neighbors(transpose=true)`
  - Predecessor lookup scanned all rows and did per-row membership checks.
  - Decision: Aligned (implemented transposed-CSR lookup path and optional precomputed transpose helper for repeated queries).
- `[P1][utils] src/utils/check.rs::check_n_neighbors`
  - Off-by-one behavior can collapse to zero neighbors in small candidate cases.
  - Decision: Aligned for Rust contract (returns `min(requested, n_seeds)` with `0` only when no seeds). **Python still uses `n_seeds - 1` when `n_neighbors >= n_seeds`** — see **Link prediction** benchmark section if export params approach graph order.
- `[P1][classification] src/classification/nn.rs`
  - Default path still densifies adjacency; sparse-first parity not complete.
  - Decision: Explicit gap or align-now (preferred align-now).
- `[P2][classification] src/classification/metrics.rs::get_confusion_matrix`
  - Matrix shape used global max labels; invalid samples could inflate dimensions.
  - Decision: Aligned (implemented valid-samples-only label-domain inference and regression test).
- `[P1][classification] src/classification/nn.rs::fit_core`
  - `k=0` corner case can force default class label.
  - Decision: Aligned (implemented: enforce `k>=1` when seeds exist).
- `[P1][regression] src/regression/diffusion.rs::Diffusion::new`
  - `damping_factor` validation is missing.
  - Decision: Aligned (implemented).
- `[P1][linkpred] src/linkpred/nn.rs::top_k_indices`
  - `select_nth_unstable_by` partition index should be `k-1`.
  - Decision: Aligned (implemented).
- `[P1][linkpred] src/linkpred/nn.rs::fit_core`
  - Bipartite neighbor count is under-selected via `check_n_neighbors` self-exclusion logic.
  - Decision: Aligned (implemented).
- `[P3][visualization] src/visualization/graphs.rs` marker-id sanitization
  - SVG marker ids derived from raw colors can be stricter/sanitized.
  - Decision: Aligned (implemented deterministic sanitized marker ids).

### Wave D (cross-cutting API surface)

- `[P1][api-surface] src/classification/mod.rs::nn`
  - Python naming uses `knn`; Rust path `nn` may benefit from alias for path-level parity.
  - Decision: Aligned (implemented alias module `classification::knn` forwarding to `nn`).
- `[P1][api-surface] src/ranking/mod.rs`
  - Python has `ranking.base`; Rust lacks an explicit comparable base module surface.
  - Decision: Aligned (implemented compatibility module `ranking::base`).
- `[P2][api-surface] src/data/mod.rs::test_graphs`
  - Test fixture namespace remains publicly surfaced.
  - Decision: Align now or explicit documented divergence.

## Frequent Fixes

- Rustdoc consistency standard:
  - See `docs/rustdoc_style.md` for canonical conventions.
  - Use contract-first rustdoc (`//!` modules, `///` items, `# Errors` for fallible APIs, and concise examples).

- Keep `TriMat` imports test-local:
  - Prefer `use sprs::CsMat;` at module scope.
  - Add `use sprs::TriMat;` only inside `#[cfg(test)] mod tests`.
  - If tests fail with `use of undeclared type TriMat`, check this first.
- Avoid `assert_eq!(Result<OkType, ErrType>, Err(...))` when `OkType` is not `PartialEq`:
  - Use `assert!(matches!(... , Err(...)))`.
- Remove unused imports immediately after refactors:
  - Move imports to the narrowest scope.

## Third Audit Acceleration Notes

- Third audit scope = same pillars as audits 1/2 (**Python parity**, **idiomatic Rust**, **performance opportunities**) **plus global package consistency** across modules.
- Prefer parity checks that validate **invariants** (shape, monotonicity, residual, stability) over exact row ordering when symmetric/tie-equivalent outputs are valid.
- For edge-case hardening, prioritize a compact suite first (disconnected, isolates/null-weights, tie cases, bipartite split consistency), then scale to larger regression corpora.
- When backend semantics differ (`which`, solver choices, unsupported modes), fail explicitly with typed errors instead of silent approximations.
- Keep optimization changes behavior-preserving by default; separate performance patches from numerical-semantics patches in review.
- For external-data paths (`load_netset`), always pair remote retrieval with path-traversal checks and deterministic local-cache fallback.

### Global Package Consistency Checklist (Third Audit)

- API naming parity:
  - Module and symbol naming should be consistent package-wide and aligned with Python-facing terminology where intended (`knn`, `ranking::base`, etc.).
- Error-contract consistency:
  - Public `fit`/`predict`/`transform` families should use consistent typed errors (`NotFitted`, invalid-parameter errors, unsupported-backend errors) without silent fallbacks.
- Parameter/default consistency:
  - Defaults and option semantics should be coherent across modules (normalization flags, tolerance/iteration defaults, directed/bipartite handling).
- Export-surface consistency:
  - Keep curated exports stable and avoid wildcard drift; ensure each module exposes only intended public entry points.
- Backend-policy consistency:
  - Unsupported backends/options must fail explicitly everywhere; no mixed behavior where some modules silently approximate while others error.
- Data I/O consistency:
  - Local/remote loading, parsing strictness, and path-safety rules should follow one shared policy across data utilities.
- Test-policy consistency:
  - Every module should include parity-relevant edge cases plus one contract test for error behavior (especially `NotFitted` and invalid input).
- Documentation/memo sync:
  - Every accepted divergence must appear in one place in the memo with current status; remove stale divergence notes immediately after alignment.

## API Coherence Rules

- Do not keep unused API knobs.
  - If implementation has one backend, remove backend-selection params from signatures.
  - Keep constructors/signatures aligned with actual behavior.
- Do not silently route unsupported backend names to a different backend.
- If behavior is intentionally simplified for parity-first progress, document it in code comments and in the memo.

## Pre-flight Before Handing Back

1. Check edited files for scope-local imports.
2. Check test modules for missing builder imports (`TriMat`).
3. Run lints/diagnostics on touched files.
4. Confirm no stale parameters remain from previous API versions.

## Current recurring compile messages to recognize fast

- `E0433 use of undeclared type TriMat`:
  - Add `use sprs::TriMat;` in the test module.
- `E0369 binary operation == cannot be applied to Result<...>`:
  - Replace `assert_eq!` with `matches!`, or derive `PartialEq` only if appropriate.

## Final Full-Review Guardrail

- Before final sign-off, run a dedicated silent behavior change audit across all ported modules.
- Explicitly check for:
  - fallback behavior that changes semantics without raising an error,
  - accepted parameters ignored or remapped to another backend,
  - defaults diverging from Python behavior,
  - shape/typing coercions that newly succeed/fail vs Python,
  - warning-vs-error differences in edge paths.
- For each divergence:
  - align to Python, or
  - raise explicit `NotImplemented` / typed error and document it.

## Sparse-backend hard rule

- Backend implementations must not densify sparse/operator inputs.
- Dense conversion is allowed only in explicit dense utility/testing paths, never in algorithm backends.
- If an algorithm cannot yet be implemented without densification, return explicit `NotImplemented` / error and add a memo note.
- During review, watch for suspicious patterns: `to_dense`, sparse-to-dense reconstruction loops, full `Array2` materialization of operator inputs.

## Dependency minimization rule

- Prefer stdlib / existing crate capabilities before adding a new dependency.
- Any new dependency must have clear parity or correctness justification, not convenience only.
- For parser/IO features, prefer lightweight local implementations first to keep offline builds workable.
- If a new dependency is unavoidable, document:
  - why it is needed,
  - why existing dependencies are insufficient,
  - fallback behavior when network install is unavailable.

## Benchmarking hard rules (protocol consistency)

- Graph sourcing consistency:
  - Python and Rust benchmark paths must each use their own package-native data loaders to materialize the same dataset.
  - Avoid custom cross-language graph conversion bridges unless explicitly justified and documented.
- Timing scope consistency:
  - algorithm benchmarks (for example ranking) must exclude dataset loading/materialization time.
  - Data-loading benchmark is separate and should not be mixed into algorithm timing results.

## Audit Protocol (Deep, Memo-First)

- Scope per module:
  - parity with Python behavior/API,
  - idiomatic Rust quality,
  - performance opportunities (while enforcing sparse-backend rule).
- Findings are logged before code changes and tagged:
  - `P0`: correctness regression / blocker,
  - `P1`: user-visible parity mismatch,
  - `P2`: performance/idiomatic opportunity,
  - `P3`: cleanup/doc-only.
- Every finding must include a decision:
  - `Align now`,
  - `Explicit gap (NotImplemented/Error)`,
  - `Accept divergence (documented)`.
- Execution mode:
  - batched waves, module-by-module findings.
- No implementation action before memo findings are complete and backlog is prioritized.

### Finding template

- `[Px][module] file::symbol`
  - Parity:
  - Idiomatic Rust:
  - Performance:
  - Decision:
  - Notes/Test ideas:
