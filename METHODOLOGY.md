# Methodology — How Many Measurements a Country Needs for a Reliable IQB Score

Companion to [`03-sample_size_reliability.ipynb`](03-sample_size_reliability.ipynb). The notebook
shows the work; this document states the method, the assumptions it rests on, what was verified
and how, and where the numbers should not be trusted. It should be readable without running
anything.

**Scope.** One question: given that a country's IQB score is computed from a finite sample of
M-Lab measurements, how many measurements are needed before the reported score can be trusted to
match the score of the underlying distribution? This is a question about sampling error in the
score, not about whether IQB measures the right thing.

**Interpretation of the result.** The requirement reported below is the primary, model-based
criterion in this project: the smallest sample size for which
`P(|estimated IQB - target IQB| <= 0.05) >= 0.95`. The target is the country's 24-month pooled
distribution, evaluated at p50. These are not guarantees for a single current month. They are iid
lower bounds because measurements are clustered by ASN, user, device, and time of day. An
exploratory diagnostic reports a different quantity, expected weighted threshold-flip error; its
approximately 100-sample crossover must not be described as 95% confidence.

---

## 1. The result

To land within **±0.05** of the true score **95%** of the time, evaluating IQB at **p50**:

| what you publish | median country | 90th percentile | sufficient today |
|---|---|---|---|
| p50 alone | ~560 | ~17,000 | 185 / 245 |
| p25 / p50 / p75 | ~2,400 | ~86,000 | 138 / 245 |
| all nine levels | ~5,400 | ~114,000 | 111 / 245 |

Counts are **download-side measurements per month** (§4). There is no single threshold that works
for every country: the requirement spans nearly seven orders of magnitude — from 2 measurements
to 16.6 million — and for nine countries no sample size suffices at all.

These figures should therefore be read as conditional planning estimates, not as a universal
minimum sample count. A current-month requirement adjusted for clustering is not available from
the country-month percentile aggregates alone.

---

## 2. Why the question is answerable at all

IQB is a weighted average of use-case scores, each a weighted average of network-requirement
scores, each of which is a **binary** pass/fail against a fixed threshold. Flattening the three
layers of weights collapses the whole configuration to ten *(metric, threshold)* tests — 24
use-case × requirement combinations, but only 10 distinct tests, because use cases share
thresholds:

```
IQB = Σ_k w_k · 1{metric_k passes threshold_k},    Σ_k w_k = 1
```

| test | weight | | test | weight |
|---|---|---|---|---|
| `loss < 0.01` | 0.2021 | | `upload > 25` | 0.1285 |
| `latency < 100` | 0.1644 | | `loss < 0.005` | 0.0809 |
| `download > 10` | 0.1416 | | `upload > 10` | 0.0495 |
| `download > 25` | 0.1285 | | `latency < 10` | 0.0490 |
| | | | `latency < 50` | 0.0417 |
| | | | `upload > 5` | 0.0139 |

Two consequences follow, and everything else in the method is built on them.

**The score cannot move a little, only jump.** An estimated IQB differs from the true one by exactly the
total weight of the tests that came out on the wrong side. So "how many measurements do I need?"
becomes "how many do I need to get each test on the correct side of its threshold?" — a question
about ten proportions, not about four continuous distributions.

**The metric's own dispersion is irrelevant.** A country with wildly variable download speeds is
no harder to score than a stable one. Only the *margin* — how far the pass rate sits from the
level's cut-off — matters. This is why the answer does not depend on modelling the shape of any
throughput distribution.

The flattening is verified rather than assumed: see §8.

---

## 3. The estimator and what is random about it

Evaluating IQB at percentile level *q* asks whether the *q*-quantile of each metric passes each
threshold. Rewriting that in terms of the pass rate π (the fraction of a country's measurements
that beat the threshold), the *q*-quantile passes test *k* exactly when

```
q > 1 − π_k
```

so the level converts into a **cut-off on a count**: p50 asks whether the pass count clears 0.50n,
p75 whether it clears 0.25n, p99 whether it clears 0.01n. Each of the nine levels asks a different
question of the *same* measurements, and a country is hard to score at a level only if one of its
ten pass rates happens to land near that level's cut-off.

### The approximation that builds intuition

Each measurement passes independently with probability π, so the pass count is Binomial(n, π) with
standard deviation √(nπ(1−π)) — the **spread**. The gap to the cut-off is |π − (1−q)|·n = δn.
Demanding the gap beat the spread by *z* standard deviations gives the ordinary sample size for a
proportion:

```
n  >  π(1−π) z² / δ²
```

The gap grows like *n* and the spread like √*n*, so more measurements always help — but every
factor of 10 in measurements improves margin resolution by only about 3×.

### The exact form actually used

The normal approximation is fine at n = 20,000 and poor at n = 20, and much of the panel is small.
The computation therefore uses the exact order-statistic result. For *n* i.i.d. measurements the
sample *q*-quantile is the *r*-th order statistic with `r = ⌈qn⌉`, and its position on the true CDF
is Beta distributed:

```
U = F(Q̂_q) ~ Beta(r, n − r + 1)
```

Test *k* looks like a pass exactly when `U > c_k`, writing `c_k = 1 − π_k`. Note this is exact for
**discrete** F as well, which matters because packet loss has a large point mass at zero:
`P(Q̂_q ≤ t) = P(Bin(n, F(t)) ≥ r) = P(Beta(r, n−r+1) ≤ F(t))` holds regardless of atoms in F. This
was confirmed by simulation (§8).

### Combining ten tests

Within a metric the thresholds are nested — passing 25 Mbps implies passing 10 — so all thresholds
on a metric are compared against a **single** draw of U, and each metric contributes one of at most
four values. Across four metrics the estimate lives on a fixed lattice of 3 × 4 × 4 × 3 = **144
points** spanning 0.000 to 1.000, with a smallest adjacent gap of 0.0004. Every probability on that
lattice is a difference of Beta survival functions, so the entire sampling distribution of the
score is available in closed form for any *n* — no simulation in the production path.

Metrics are treated as independent across the four blocks. §9 bounds what that assumption can cost.

---

## 4. Inputs

24 months of M-Lab measurements (2024-01 … 2025-12), pre-aggregated to one row per country per
month: 5,720 country-months across 245 countries. Downloaded by [`00-setup.ipynb`](00-setup.ipynb)
into `data/v1/`.

### Two sample counts, not one

The `downloads_by_country` table carries `download_*`, `latency_*` and `loss_*`; the
`uploads_by_country` table carries only `upload_*`. So **seven of the ten tests — both packet-loss
tests among them — rest on the download-side count**, and three rest on the upload-side count.

The two are not interchangeable: the upload-side count is 0.82× the download-side at the median
(P10 0.65×, P90 1.00×). All requirements in this analysis are therefore quoted in **download-side
measurements**, with the upload tests evaluated at each country's own upload/download ratio. Taking
`min(down, up)` as a single count — the obvious shortcut — understates the n behind latency and
loss by about 23% at the median, which matters because a loss test is the binding constraint for
roughly half the panel.

The two tables are joined on `country_code`; the inner join drops 25 of 5,745 country-months
(0.4%) where a country appears in one direction only.

### Recovering the pass rate

π is the one input the method needs, and the parquet does not report it. It is reconstructed from
the nine percentile knots by locating the threshold between them.

Values are first mapped into a **quality space** where higher is always better and passing always
means `y > t`: log for download and upload (throughput spans decades), negative log for latency,
and negative-linear for loss (which has a point mass at zero, so a log would distort it). This
makes one rule cover all four metrics. The direction convention — that the percentile index is a
*quality* rank, so `latency_p1` is the worst latency in a country and `latency_p99` the best — is
checked against every country-month rather than assumed (§8).

Between adjacent distinct knots the definition of a quantile brackets F(t) in
`[q_max(left), q_min(right))`, and the reconstruction interpolates linearly within that bracket.
Repeated values (loss = 0 across p75…p99) are treated as a point mass, so the CDF jumps there.

A threshold falling **outside** the reported p1–p99 range yields only a bound: if a threshold is
better than a country's p99, all we learn is that at most 1% of measurements pass. Such values are
pushed just past the edge (to a pass rate of 0.005 or 0.995) rather than set exactly at it — returning
exactly 0.01 would place the cut point precisely at the p99 reporting level and manufacture a
permanent 50/50 result there. This affects **12.0%** of country-month-threshold pairs, concentrated in
`latency < 10` (60.7% out of range) and `latency < 50` (26.1%). §10 records what it costs.

---

## 5. The reliability criterion

Two statistics, answering different questions.

**MAD of the sampling distribution** describes typical behaviour. It is the right robust width for
a lattice-valued estimate: most of the time the estimate lands exactly on the true score and
occasionally one test flips and it jumps by up to 0.2, so a standard deviation is dominated by
those rare jumps while MAD ignores them. `MAD = 0` reads cleanly as *more than half the time you
get exactly the right score*. Its limit is that it saturates at 0 and stops distinguishing a
country right 51% of the time from one right 99.9% of the time.

**A tail criterion sets the requirement:**

```
P(|ÎQB − IQB| ≤ τ) ≥ 0.95,     τ = 0.05
```

τ = 0.05 is strict on purpose — smaller than 6 of the 10 test weights, so in practice it means "no
consequential test flipped". Strictly it bounds the *net* error, so it also forgives two flips that
cancel. Requirements at τ = 0.02 and τ = 0.10 are computed alongside and exported.

**Finding the requirement.** The requirement is the smallest *n* on a log grid (170 points,
1 → 2×10⁸) at which the criterion holds **there and at every larger n**. The trailing condition is
not defensive padding: the sample quantile is order statistic `r = ⌈qn⌉`, so `r/n` jumps around *q*
and reliability can genuinely worsen when one more measurement is added. Taking the first crossing
alone would report a sawtooth artefact. Where no grid point qualifies, the requirement is infinite
— a real answer, not a failure, and it propagates as infinity through every downstream column.

Grid resolution is about 12% per step, so a reported requirement of "563" is a grid point; the
trailing digits carry no precision.

---

## 6. Estimating the margin without circularity

The margin has to come from the same percentile data whose reliability is in question, and a
country with three measurements gives a useless estimate of its own pass rates. Pooling all 24
months fixes this, and does so **exactly**.

[`02-cdf.ipynb`](02-cdf.ipynb) establishes that monthly distributions must be combined by averaging
**CDFs**, since the mean of two p50s is not the p50 of the pooled data. A pass rate is that same CDF
evaluated at a *fixed* x — the threshold — so

```
π_pooled = 1 − F_merged(t) = Σ_i w_i (1 − F_i(t)) = Σ_i w_i π_i
```

is an identity rather than an approximation. What cannot be averaged is a *quantile*, a position on
the vertical axis that moves when the distribution changes shape; a pass rate sits on the horizontal
axis at a pinned x, and those add linearly. Weights w_i are sample counts. This applies notebook
02's rule where it is exact and skips its inversion step entirely, because nothing here needs
percentiles back out.

### What this makes the numbers mean

Requirements are the sample size at which the reported score matches the score of the country's
**24-month pooled** distribution. That is the right target for capacity planning and the wrong one
for auditing a single month.

The gap is not academic. A country's own pass rate moves between months with a standard deviation
around 0.037 (measured on `download > 10` across countries with ≥5,000 measurements a month), while
the margin that the median requirement of 563 measurements resolves is about 0.041. **The target
moves by roughly as much as the sample size is being asked to resolve.** Recomputed month by
month, the requirement rises from a median of 563 to 706, and at the 90th percentile from 16,737
to 29,460.

Both are reported. Treat the pooled figures as a minimum.

---

## 7. What comes out

Running the notebook writes `analysis_outputs/03_sample_size_requirements.csv` — 245 rows, 31
columns. Note that `.gitignore` excludes `*.csv`, so the file is regenerated rather than tracked.

| column | meaning |
|---|---|
| `n_down_month_median`, `up_ratio` | what the country collects now, and its upload/download ratio |
| `n_req_0.02`, `n_req_0.05`, `n_req_0.1` | requirement at p50 for each tolerance |
| `n_req_upload_side` | the same requirement expressed in upload-side measurements |
| `n_req_p1` … `n_req_p99` | requirement at each of the nine levels |
| `n_req_curve`, `curve_reachable` | requirement to publish all nine; `False` when some level is unreachable at any n |
| `levels_supported`, `binding_level` | which levels the country's current volume supports, and which one binds |
| `margin`, `binding_test` | the smallest margin among heavy tests, and which test it belongs to |
| `bias_at_current_n` | signed bias of the published score at current volume (§10) |
| `margin_unresolved` | `True` where nine knots do not determine the margin (§10) |
| `verdict_p50`, `months_to_pool` | the summary judgement and how many months of pooling would close the gap |

`verdict_p50` applies the first matching rule:

```
n_month      ≥ n_req   →  monthly data is sufficient          (185 countries)
n_month × 12 ≥ n_req   →  sufficient if months are pooled      (25)
n_req > 100,000        →  sits on a threshold: publish a range            (9)
otherwise              →  short of data: pool or suppress      (26)
```

The 12-month line is a policy choice, not a statistical one — `months_to_pool` is the underlying
continuous number if you want the line elsewhere. The two middle categories differ in kind: the
"pooled" group needs a median of 3 months (worst case 7) and is a reporting-cadence problem, while
the "short" group needs a median of 47 months and is a collection problem. They need almost the
same amount of data (2,741 vs 2,317 median); what separates them is that the short group collects
37 measurements a month against the other's 1,283.

---

## 8. Verification

Every step that could be checked against something independent was.

| what | how | result |
|---|---|---|
| Config flattening to ten tests | 400 country-months × 9 levels scored both ways | agrees with `IQBCalculator` to 1.11e-16 |
| Closed-form sampling distribution | Monte Carlo (2,000 reps) through the production `IQBCalculator`, 5 countries × 2 sample sizes | mean, SD and P(within τ) agree within simulation error |
| Beta form under a point mass | simulation against a distribution with a 60% atom | exact to 4 decimal places at n = 50, 101, 200, 501 |
| Percentile index is a quality rank | monotonicity of quality across p1…p99 | 0 violations in 5,720 country-months, all four metrics |
| IQB(q) non-decreasing in q | implied by the reduction in §2 | 0 violations in 5,720 country-months |
| "One binding test sets the requirement" | whole-score requirement ÷ worst single-test requirement | median 1.00, max 2.00 — no union-bound inflation |
| `binding_test` names the real driver | compared against the test with the largest single-test requirement | agrees for 240/245; the 5 misses are ties |
| Sensitivity to the out-of-range bound | recomputed at 0.0001, 0.001, 0.005, 0.0095 | p50 median unchanged; all-nine median 5,113–7,178 |

Two limits on what this establishes. The Monte Carlo draws from the *same* reconstructed CDF the
closed form uses, so it validates the Beta algebra, the 144-point lattice and the config flattening
— **not** the reconstruction, which §10 identifies as the weakest link. And no within-month
replicate exists in a monthly pre-aggregate, so the reconstruction cannot be validated from this
data at all. That is a second argument for emitting the pass rates directly (§11).

### Does the model match reality?

The 24-month panel gives an independent read. If the model is right, a country's month-to-month
movement should be *at least* as large as the predicted sampling noise, the excess being real
change. It is: predicted noise is smaller than observed for 229 of 238 countries, and the nine
exceptions are all at small *n*, where the 24-month pooled mixture is wider than any single month
and over-predicts the spread.

Sampling error falls below 5% of observed variance in the **100–1,000** measurements-per-month bin
and is negligible above it. So past roughly 10³ measurements a month, more data does not make a
country's reported score steadier — the residual movement is real. Decomposing that movement using
the per-ASN tables, about 77% of it (median) is change *within* networks and about 7% is shifts in
which networks got measured, so treating it as signal rather than noise is defensible for the
typical country — though for 25 of 117 large countries the composition shift exceeds 30%.

---

## 9. Assumptions and what they cost

**Independence across metrics.** Recomputed under the opposite extreme — all four quantiles
perfectly comonotone, sharing a single CDF position — the median requirement moves from 563 to 533
and the P90 ratio is 1.00. Immaterial, and for a structural reason: one binding test sets the
requirement, so dependence has nothing to act on.

**Independence *between measurements*** is the assumption that does real work, and it is false.
M-Lab tests are self-selected and clustered by user, network and time of day. In the latest month
the median country's measurements sit in about **4.3 effective ASNs** (1/HHI), its largest ASN
supplies roughly **37%** of them, and pass rates differ across a country's ASNs with an
n-weighted standard deviation near **0.09**. Every figure here is therefore a **lower bound** on
what is really needed. The panel cannot quantify the design effect directly, because that would
require a within-month replicate the pre-aggregate does not contain.

**Tolerance.** At τ = 0.02 the median requirement is 885; at τ = 0.10, 380. The answer is not
sensitive to τ in a way that would change any recommendation.

**Rank convention.** `r = ⌈qn⌉`. Other conventions differ only at very small *n*.

---

## 10. Where the numbers should not be trusted

**Per-country figures are much less certain than aggregates.** The median and quartile requirements are
stable across four reconstruction conventions (bracket/top/bottom tie handling, log vs linear
interpolation), but individual countries are not. For **22 of 245**, the two conventions disagree
about the test that *sets* that country's requirement by more than that test's own margin; for 12
of those they land on opposite sides of the cut-off. Hungary is the clean example: both conventions
agree that about half its measurements pass `loss < 0.005`, and disagree on whether that is 49.98%
or 49.02%. At p50 that is the difference between a 50/50 result and a certain one, and more
measurement cannot resolve it — the ambiguity is in the nine-knot summary, not in the sample.

Treat those 22 requirements as **unresolved rather than large**. (An earlier version of this
analysis reported 62 countries here, by comparing the largest disagreement across all ten tests
against the *binding* test's margin. That counts countries whose disagreement sits on a test with
margin to spare; the matched per-test comparison is the one above.)

**Some countries cannot be fixed by any sample size.** If a pass rate for a heavy threshold sits
exactly at the reporting level, the test is unpredictable at every *n* and the requirement diverges.
Ten countries are permanently on a threshold in the pooled distribution — one of them, Canada,
already collects enough to clear even that requirement — and 138 sit on a threshold in at least
one month, which is what produces the wide confidence bands in notebook 02.
This is a property of threshold scoring, not of data collection.

**The estimate is biased, not merely noisy.** A country scoring near 1.0 cannot score any higher —
every test that can flip can only flip downward — so its expected reported score sits below its
true one, and near 0 the reverse holds. The estimate is pulled toward the middle of the lattice,
and since 0 is bad and 1 is good, **an under-measured country is published as worse than it is**.
At n = 50 the mean bias among countries scoring above 0.8 is −0.054. At current volumes, 24 of 245
countries carry a bias larger than the ±0.05 tolerance itself, the worst by −0.33, and the average
across all 245 is −0.0025 rather than zero. This does not shrink when the tolerance is widened and
does not average out under ranking or regional averaging, so it needs correcting or flagging rather
than a wider error bar.

**Thresholds outside p1–p99** are recorded with a margin of 0.495, which understates how safe those
tests really are. The all-nine-levels figure is the one most exposed to the bound chosen at the
edges: its median moves between about 5,100 and 7,200 across plausible choices and its 90th
percentile between 114,000 and 201,000. The p50 figure does not move.

---

## 11. What follows

1. **Gate per (country, percentile level), not on a global n.** Every requirement is computable in
   closed form from data already on hand, so a fixed cutoff is strictly worse: it suppresses
   well-separated small countries and passes large ones that sit on a threshold. A country that cannot support a
   trustworthy p99 usually still has a solid p50 — the median country supports 8 of the 9 levels
   today, and p50 is the binding level for only 79 of 245.

2. **Emit the ten pass rates from the aggregation pipeline.** Because IQB at level *q* is
   `Σ_k w_k · 1{q > 1 − π_k}`, the ten pass rates are a **sufficient statistic for the entire
   IQB-versus-percentile curve** — they determine the score at every level exactly. They cost one
   `COUNTIF`/`COUNT` per threshold in the existing aggregation. Ten extra columns would remove the
   reconstruction ambiguity for those 22 countries, fix the 12% of thresholds that fall outside the
   reported range, and make the sampling model checkable for the first time. This is the cheapest
   fix available and the only one that removes an error no sample size can.

3. **Correct or flag the small-sample bias.** A tolerance band describes spread around the wrong
   centre. Either publish the bias-corrected estimate — the sampling distribution is in closed
   form, so subtracting the bias costs nothing — or suppress the point estimate and publish the
   band.

4. **Publish a band, not a point, for countries sitting on a threshold.** Notebook 02's CI machinery already
   produces it.

5. **Do not collect measurements beyond ~10³/month for stability's sake.** Above that, sampling error
   accounts for well under 1% of the month-to-month movement actually observed.

**A structural note, outside the question but visible from it.** All of the difficulty here comes
from IQB being a sum of hard indicator functions. A score that ramped smoothly across each
threshold instead of stepping would make the estimate continuous in the data, remove the divergent
requirement entirely, remove the bias above, and shrink the typical requirement by roughly the
ratio of the step to the ramp width. That is a scoring-design decision rather than a measurement
one, but the sample-size problem is largely created by how the score is designed.

---

## Reproducing

Run `00-setup.ipynb` once to fetch the data into `data/v1/`, then run
`03-sample_size_reliability.ipynb` top to bottom. Both use the **`pulse-iqb`** kernel, which is
already the notebooks' recorded kernelspec — select it if your Jupyter front end does not pick it
up automatically.

Requires the `iqb` package (for `IQBCalculator` and `IQB_CONFIG`) plus numpy, pandas, scipy,
pyarrow, pycountry and matplotlib. Running the notebooks headless needs `nbclient`, which is not
currently installed in the `pulse-iqb` environment.

The notebook is deterministic apart from the §6c Monte Carlo, which is seeded, and writes
`analysis_outputs/03_sample_size_requirements.csv` on its final cell.
