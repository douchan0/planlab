---
title: "When an Economic Report Arrives Late"
subtitle: "Feedback, production constraints and shortages in an input–output model"
kicker: "PlanLab · Methods paper"
authors: "The PlanLab team"
affiliation: "Reproducible computational experiments in coordination"
version: "1.1"
status: "Working paper · audit revision"
date: "November 2026"
keywords: "input–output; information delay; feedback; material balance; rationing"
abstract: "A plan can balance on paper while running short of inputs, labour or capacity in execution. We explain feasibility with two-product examples, derive the exact stability threshold of a signed scalar feedback rule, and compare six coordination rules on the same EU input–output network. The central linear programme has the steepest estimated welfare decline at the reference settings. The difference from priority rationing falls short of the multiplicity-adjusted criterion. Combining target revision with material balancing can depress output when labour is tight. Sectoral shocks amplify relative to matched no-shock runs. Forward linkage and household incidence answer separate questions about that response. The simulated administered price rule adjusts only upwards in response to shortage, so its amplitude statistics cannot locate the signed scalar threshold. A sample maximum at an interior dual-track share also falls short of evidence of an advantage over both endpoints."
---

## What a delayed report can tell a coordinator {#sec:intro}

Imagine an equipment producer making its schedule from last quarter's orders. Electricity supply falls this quarter, but the producer learns that only after filing a production target. When the shortage finally appears in the report, the producer can raise its target. Whether that increases deliveries depends on what was scarce and who gets the available electricity.

There are two questions here. First, how much can a coordinator safely adjust a single tool if it sees a signed error only after a delay? This has an exact mathematical answer. Second, how do particular coordination rules behave when goods are inputs to each other's production, resources are limited, and reports are delayed? That is an experiment inside a model. Connecting the answers requires checking that the simulated rule has the feedback structure assumed in the mathematics.

Input–output analysis describes production as a network of recipes. The economic calculation debate asks what coordinators know and what they do with that knowledge. The shortage and rationing literature studies who receives goods when supply cannot fill every claim. PlanLab places stylised coordination rules in a shared production network and uses a difference equation to explain one way delay can matter. The names refer to algorithms, which capture selected features of the institutions that inspired them. [[@leontief1941]] [[@miller_blair2009]] [[@hayek1945]] [[@lange1936]] [[@kornai1992]] [[@dreze1975]] [[@levin_may1976]] [[@elaydi2005]]

## From two goods to twenty sectors {#sec:model}

### A column of the matrix is a production recipe

Suppose a unit of equipment requires half a unit of electricity, and a unit of electricity requires half a unit of equipment. Let $x$ be the vector of gross output, with electricity first. The input matrix is

$$
A=\begin{pmatrix}0&1/2\\1/2&0\end{pmatrix},
\qquad x=\begin{pmatrix}x_1\\x_2\end{pmatrix}.
$$ {#eq:example-a}

The second column is the equipment producer's recipe. Multiply each column by the corresponding producer's output and add the columns. The result, $Ax$, tells us how much of each good production uses up. At $x=(1,1)^\top$, production uses half a unit of each good. The other half is available for delivery or storage.

The main experiment uses Eurostat's annual 2020 domestic input–output table for the EU-27. The source mapping covers 65 categories including a zero-output category. We sum transaction values over the 20 groups on both axes before dividing by the using group's gross output. The source measures annual values at current basic prices. The model uses those numbers as a fixed-price benchmark and calls one simulation step a quarter. Absolute simulated flows therefore have the scale of the annual source, while the quarterly timing is a model convention. Labour coefficients, capital ratios, behavioural elasticities and utility weights also involve modelling choices. [[@eurostat_naio]] [[@eurostat_manual]]

### What happens in a period

Let $S^0$ and $S^1$ be opening and closing stocks. Let $C$, $G$ and $J$ be deliveries to households, government and investment, and let $D$ be goods recorded as spoilage. The economy is closed in these experiments. For each good, sources equal uses:

$$
S^0+x=Ax+C+G+J+S^1+D.
$$ {#eq:balance}

If a warehouse holds 8 units at the start, its capacity is 10, and nothing enters or leaves, 8 remain at the end. The opening stock is already part of the remaining supply. Once production and deliveries have been accounted for, remaining supply $r$ and storage capacity $B$ give

$$
S^1_i=\min(r_i,B_i),\qquad D_i=r_i-S^1_i.
$$ {#eq:storage}

Shocks first change current capacity or demand. A coordination rule then files output targets and prioritised claims using its information. The execution engine sees available resources. Let $\bar x$ be the smaller of target and capacity for each product. Let $a_i$ be labour per unit of product $i$, $L$ the labour supply and $R$ opening stock reserved for higher-priority claims. The engine chooses the feasible vector with the largest sum of gross output:

$$
\max_x\ \sum_i x_i
\quad\text{subject to}\quad
0\le x\le\bar x,\quad
a^\top x\le L,\quad
Ax-x\le S^0-R.
$$ {#eq:lp}

The last constraint requires output and remaining stock to cover intermediate use. If the target itself satisfies the constraints, the engine can use it directly. Goods are allocated by priority, with proportional shares within a tier. If a higher tier receives more than the reserve and leaves too little for production, the engine lowers the output bounds and reconciles again. Iteration counts describe this numerical process. Investment replaces depreciated capital, unfilled household demand can carry forward, and the mechanism sets the next period's price.

### Information and the six rules

In period $t$, the newest realised shortage report comes from

$$
\tau=\max(0,t-1-\ell),\qquad k=\ell+1,
$$ {#eq:timing}

where $\ell$ is the configured reporting lag and period zero denotes the initial state. Even at zero reporting lag, an outcome from this period has yet to be realised. Historical demand, capacity, inventory, backlog, prices and labour may be measured with multiplicative error. Realised shortage is delayed and is copied without another measurement error. Current government and investment demands are common inputs. Execution uses actual resources.

| Rule | Main decision |
| --- | --- |
| Central linear programme | Optimise weighted final delivery using observed capacity and demand |
| Material balance | Revise targets and balance them against reported resources |
| Priority rationing | Give selected final claims priority in the delivery ladder |
| Tâtonnement | Raise administered prices in response to reported household shortage |
| Market | Form orders and quotes using the model's market rules |
| Dual track | Split claims between a plan track and a market track |

The central linear programme optimises a linear delivery objective against the reported inputs. The engine applies actual resource constraints afterwards. The reported welfare score is a different objective, applied to household deliveries. We use the programme as one comparator. A welfare upper bound would require a separate benchmark with the same welfare objective and feasible set. The value of fresher information for a rule can be studied by holding that rule fixed and varying the reports it receives.

### Output, shortage and welfare answer different questions

Gross output includes goods used as inputs. The household shortage rate is unmet household demand divided by total household demand:

$$
s=\frac{\sum_i(d_i-C_i)}{\sum_i d_i}.
$$ {#eq:shortage}

It is zero when demand is zero. Receiving 90 out of 100 and receiving 72 out of 80 both give a shortage rate of 10%. Prices, income and carry-over affect the denominator, so consumption and the shortage rate should be read together.

The period utility score is

$$
U_t=\sum_i w_i\log(C_{i,t}+c_{0,i}).
$$ {#eq:utility}

A small constant tied to base final demand makes the score finite at zero delivery. The derivative $w_i/(C_i+c_{0,i})$ is larger at lower delivery. An extra unit in a severely undersupplied category can therefore change the score more than an extra unit where delivery is already high. H2 and H5 use post-burn-in mean utility points. The files also store $N\sum_t\beta^{t-t_0}U_t/\sum_t\beta^{t-t_0}$, where $N$ is the number of usable periods and $t_0$ is the first. This is a normalised discounted mean times $N$. Changing the monetary unit adds a common constant to the scores under the same technology, weights and window. Differences in points cancel that constant, whereas ratios to the level of the score depend on it.

## Why more room in one sector can reduce output in another {#sec:feasible}

Take two sectors with no intermediate inputs, total labour 2, and unit labour needs 2 and 1. If sector two can produce at most 0.5, the sum-maximising solution is $(0.75,0.5)$. Raise its bound to 2 and the unique maximiser becomes $(0,2)$. Total output rises from 1.25 to 2 while sector one's output falls. Enlarging a feasible set can improve the optimum value while changing the composition in either direction.

Return to the electricity and equipment example in [[REF:eq:example-a]], with both capacities equal to 1 and no stocks. Full output gives net delivery $(0.5,0.5)$. Producing $(1,0.5)$ gives $(0.75,0)$. Reversing the mix gives $(0,0.75)$. Both are feasible, but taking the larger entry from each gives a total of 1.5. The two sectors can produce at most 2 units in total, and half of that is used up in production, so a net total of 1.5 is impossible.

If $y$ combines all final uses, the feasible net deliveries before priorities are imposed are

$$
\mathcal Y=\{y\ge0:\ \text{some }x\text{ satisfies }0\le x\le\bar x,\ a^\top x\le L,\ y\le S^0+(I-A)x\}.
$$ {#eq:deliveries}

The negative entries in $I-A$ prevent us from multiplying a componentwise inequality through it and preserving the order. The set above can be explored with a linear programme. For fixed production and fixed net supply, priorities can distribute the same goods differently. Priorities that affect intermediates, stocks or investment can also change future supply.

This distinction matters for stationary plans. The full simulator would have to return to the same targets, stocks, capital, backlog and relevant history. The calm calibration checks one baseline. General existence, uniqueness and convergence require an argument about the full state transition. Maximisation of total output gives no componentwise monotonicity guarantee for that argument.

## The exact delay threshold for a signed scalar rule {#sec:theory}

Let $u_t$ measure how far a tool is from its equilibrium value. Suppose the signed error near that value is approximately $-b u_t$, with $b>0$. This is the first-order or tangent-line approximation. An adjustment coefficient $\eta$ and the slope $b$ give the effective gain $g=\eta b$. If the error arrives $k$ periods late, the rule is

$$
u_{t+1}=u_t-g u_{t-k}.
$$ {#eq:delay}

At $k=0$ this becomes multiplication by $1-g$ each period. With $g=0.5$, a deviation halves. With $g=1.5$, it changes sign and still halves in magnitude. With $g=2.2$, its magnitude grows by a factor of 1.2 each period. The stability condition is $|1-g|<1$.

At $k=1$, two values are needed to describe the state:

$$
\begin{pmatrix}u_{t+1}\\u_t\end{pmatrix}
=
\begin{pmatrix}1&-g\\1&0\end{pmatrix}
\begin{pmatrix}u_t\\u_{t-1}\end{pmatrix}.
$$ {#eq:state-matrix}

For general $k$, the state contains $k+1$ values. A mode that changes by a factor $z$ each period has form $z^t$. Substituting gives

$$
P_k(z)=z^{k+1}-z^k+g.
$$ {#eq:characteristic}

If all roots have modulus less than 1, every mode decays. A root outside the unit circle supplies a growing mode. Write a complex root as $z=r(\cos\theta+i\sin\theta)$. Multiplying by it scales length by $r$ and rotates by $\theta$. The real part of a mode then has the form $r^t\cos(t\theta)$, which separates the change in amplitude from the oscillation. For integer $k\ge0$, the exact stability condition is

$$
0<g<g^\star(k),\qquad
g^\star(k)=2\sin\!\left(\frac{\pi}{2(2k+1)}\right).
$$ {#eq:threshold}

At the boundary all solutions are bounded. When $k\ge1$, there is a simple conjugate pair on the unit circle. An initial state that excites this pair retains a periodic component of period $4k+2$. If it does not excite the pair, only decaying modes remain. When $k=0$, the boundary has just one root, $-1$, with period two. The proof is in [[REF:sec:proof]].

[[TABLE:stability_boundary]]

[[FIG:delay_response]]

The first delay reduces the bound from 2 to 1. The second reduces it further to about 0.618. Since $\sin x\approx x$ for small angles, for long delays the bound is approximately $\pi/(2k+1)$, and

$$
k g^\star(k)<\pi/2,\qquad
\lim_{k\to\infty} k g^\star(k)=\pi/2.
$$ {#eq:budget}

Reversing the feedback sign gives $u_{t+1}=u_t+g u_{t-k}$. Its characteristic polynomial is negative at $z=1$ and positive for sufficiently large real $z$, so it has a real root above 1. That is a growing mode for every positive gain in this scalar model. Applying that conclusion to a sectoral target rule requires establishing the sign and the scalar reduction for that rule.

### Why the administered price experiment is separate

The simulated price rule uses a household shortage rate, which is nonnegative. It sets

$$
\log p_{i,t+1}=\log p_{i,t}+\eta s_{i,\tau}
$$ {#eq:admin-price}

and applies fixed price bounds. An observed shortage raises a price, and zero shortage leaves it unchanged. This is a one-sided update. Its raw price series can only rise or stay flat. A series that rises at changing speeds can still move above and below a moving average. The standard deviation of that deviation measures amplitude around the chosen trend, even if the underlying price never falls.

B7 varies a single sector's price with feedback off, then compares the shortage readings. A finite difference is a useful description of that response. In a dynamic, multi-sector linearisation, cross-sector responses and additional states would enter a matrix. A sector average cannot bound its eigenvalues. We therefore report the scalar threshold separately from the simulated amplitude comparisons.

## Experimental design {#sec:design}

The protocol, amendments and deviations document the original design, changes after pilot runs, and later reinterpretations of the stability diagnostic. The present statistical analysis was revised after the audit. The historical plan and actual grid remain separately identifiable.

The grid has 7,364 treatment runs. B3 additionally uses matched no-shock controls. B2 contains 204 combinations and 4,080 runs. The reporting lags are 0, 1, 2, 3, 4 and 6, and only specified rules sweep gain. Most runs have 80 periods, with 30 discarded as burn-in. The standard capacity cut starts at period 40 and lasts eight periods. B3 uses the amended fifty per cent cut, replacing the original thirty per cent shock. B4 uses capacity multipliers 0.5, 0.3 and 0.1, corresponding to losses of 50, 70 and 90 per cent.

Matching on a seed in the same information regime holds the external random draws fixed. Realised demand may still differ because incomes, prices and carry-over respond to the rule. Pairing takes a difference within a seed. The statistical unit is the seed or the specified sector cell, rather than every observation on an autocorrelated time series. Holm adjustment controls the risk of false positives across a family of tests. For example, a one-sided p-value measures how unusual a test statistic at least this favourable would be under a specified null model. It does not give the probability that the null model is true. Fixed threshold checks are reported separately. [[@holm1979]] [[@wilcoxon1945]]

## Results {#sec:results}

### Baseline, noise and labour

The calm B1 calibration reproduces base output and near-zero shortage across the six rules. B1b adds both demand innovations and measurement noise without an external capacity cut. Capital can still fall if delivered investment is below depreciation. Output and household delivery are divided by their own base levels, so 0.94 means 94 per cent of the corresponding baseline.

[[TABLE:noise_only]]

A separate paired sensitivity check keeps all twenty material-balance seeds and their demand draws while removing measurement error. Mean output rises from 93.71 to 99.69 per cent of baseline. The mean shortage rate falls from 4.69 to 0.96 per cent. Demand still fluctuates in this comparison. The audit report gives the replication steps.

B8 varies the size of target revisions and the number of balancing rounds. Either step on its own leaves output close to the baseline. Together they lower it substantially. A larger target can require more labour than is available. A constrained plan can then cut the output of labour-intensive sectors, changing the mix as well as the total. The labour-supply comparisons in B6 support this explanation at the tested settings.

[[TABLE:balance_decomposition]]

### Delay, prices and target revisions

**H1.** At the highest price gain, amplitude comparisons match the same seed and elasticity and compare each positive reporting lag with zero. They describe how delay changes deviations from a trend. The one-sided update in [[REF:eq:admin-price]] prevents those comparisons from identifying the threshold in [[REF:eq:threshold]].

**H1′.** Higher material-balance revision gains accompany higher shortage in the tested grid. The gain 0.05 versus 1.6 comparison averages each seed across matched delays before testing. B8 shows why the size of the response is conditional on balancing and labour availability.

**H2.** At gain 0.30 and price elasticity 0.4, we fit the slope of mean utility points against lag within each seed and mechanism. We compare the central linear programme with every other mechanism using within-seed slope differences. Its sample decline is steepest, about 1.077 utility points per added period of delay, versus 1.045 for priority rationing. The difference is about 0.033 points with a Holm-adjusted p-value of 0.080. The sample ordering matches the prediction, while the difference between the leading two rules falls short of the adjusted criterion.

### Shock amplification and the incidence of shortage

**H3.** Average the post-burn-in household shortage rate across seeds for each of the twenty sector shocks to material balance. Compare each mean with 0.5 per cent. The adjudication table reports the range, mean, and number of cells below the line.

**H4.** If a sector produces 10 units at the baseline, cutting its capacity in half removes 5 units of potential output directly. If total production is 15 units below the matched no-shock path in the same period, the amplification ratio is 3. For sector $j$ and seed $s$, we calculate

$$
M_{j,s}=
\frac{\max_{40\le t\le60}\bigl(X^{\mathrm{control}}_{s,t}-X^{\mathrm{shock},j}_{s,t}\bigr)}
     {0.5\,x^{\mathrm{base}}_j}.
$$ {#eq:amplification}

This subtracts pre-existing differences from base output. The response includes the effects of demand, capital, inventory and allocation. We average over the fixed sectors within each seed before testing average amplification. A small sector's direct loss can give a large ratio. We therefore examine forward linkage against absolute output loss separately. Forward linkage is the share of the product's base output used as intermediate inputs. Mean paired amplification is approximately 7.56 for material balance, 8.21 for priority rationing and 7.56 for tâtonnement. All three amplification tests pass adjustment. The three forward-linkage correlations with absolute loss fall short of the adjusted criterion. The adjudication table also counts three significant secondary ratio correlations, whose denominator is especially sensitive to small sectors. Household incidence remains a separate question.

### Does a mixed share beat both endpoints?

**H5.** The dual-track share $\theta$ varies from zero to one. These are endpoints of the same dual-track rule. A grid of estimates can have an interior sample maximum even when true means are equal. For each interior candidate we pair with both endpoints by seed. We use two one-sided paired t-tests of the mean differences. The larger p-value tests whether the candidate beats both, and Holm adjustment covers the candidates across both lags. At zero reporting lag the highest interior sample mean, at 0.9, is roughly 0.011 utility points above the stronger endpoint at 1. The grid does not establish an interior advantage.

[[TABLE:hypothesis_adjudication]]

## Scope and reproduction {#sec:repro}

The mathematical threshold applies to the signed scalar equation. In the experiments, allocations, feedback directions and constraints govern which aspects of the equation provide a useful analogy. The magnitudes depend on fixed input recipes, one baseline network, simulated labour and demand, and assigned welfare weights. The model also lacks input substitution and external trade. Historical policy conclusions require separate validation with observed data.

The repository includes cell definitions, run summaries, sector panels and accounting samples. The per-period metric panels are generated locally. Run `scripts/run_all.py --dry-run` for the grid, `scripts/reproduce.sh` for the experimental pipeline, and `python -m verification.run_all` to check the theorem, accounting, sampled re-runs and statistics. The verification report lists missing period panels as skipped comparisons. The paper's hypothesis table is generated from the same log as the results tables.

## Appendix · Proof of the delay threshold {#sec:proof .appendix}

### Find the first possible crossing

Roots move continuously with $g$. A root crossing from inside to outside must have modulus 1 at the crossing. Write it as $z=e^{i\theta}$, with $0<\theta<2\pi$. The characteristic equation gives

$$
z^k(z-1)=2\sin(\theta/2)e^{i[(k+1/2)\theta+\pi/2]}=-g.
$$ {#eq:phase}

The modulus gives $g=2\sin(\theta/2)$. The angle is $\pi$ up to a whole turn, so

$$
\theta_j=\frac{\pi(1+4j)}{2k+1},\qquad j=0,1,\ldots,k.
$$ {#eq:angles}

The two extreme half-angles add to $\pi$. All other half-angles lie between them, so their sines are at least as large as those at the extremes. This gives the smallest possible crossing gain in [[REF:eq:threshold]].

### Determine which way the roots move

On the unit circle the derivative of the polynomial with respect to $z$ is $z^{k-1}[(k+1)z-k]$, which is nonzero. Implicit differentiation of the root equation and substitution of $z^k(z-1)=-g$ give

$$
\frac{d|z|^2}{dg}=
\frac{2[(k+1)g-\cos(k\theta)]}{|z^k-(k+1)g|^2}.
$$ {#eq:direction}

The denominator is positive. On $[0,\pi/2]$, the inequality $\sin x\ge 2x/\pi$ gives $(k+1)g^\star(k)\ge 2(k+1)/(2k+1)>1$. At every crossing $g\ge g^\star(k)$ and the cosine is at most 1. Hence all crossings are outward.

### Count the roots

At zero gain the roots consist of one 1 and $k$ zeros. For sufficiently small positive gain, the roots near zero stay inside the circle and the root near 1 starts moving inward as $1-g+O(g^2)$. All roots therefore lie inside up to the first crossing. After that crossing there is at least one outside, and further crossings can only take more roots outside. At the threshold, the unit roots are simple. A full turn of the surviving mode at angle $\pi/(2k+1)$ takes $4k+2$ steps. The mode persists only if the initial state excites it.

The verification code checks the roots and the boundary numerically for several delays, and checks time-domain trajectories separately.
