| id | institution | information used | allocation rule |
|:--|:--|:--|:--|
| `lp_central` | omniscient central linear programme | latest observation of demand and capacity | optimal plan of $\max w\cdot y$ subject to $(I-A)x\ge y$, $x\le\bar x$; pro-rata fulfilment |
| `material_balance` | resource–need balancing, $R$ rounds, priority tier first | plan targets and observed stocks | proportional within priority tier |
| `rationing_priority` | administrative quotas, tier 1 before tier 2 | observation | quota per claimant in priority order |
| `tatonnement` | Langean price board, $p\leftarrow p+\eta z$ | observation and own prices | willingness-to-pay order |
| `market` | within-period Walrasian tâtonnement | current prices within the period | market clearing |
| `hybrid_dual_track` | plan track at administered price, share $\theta$; market track, $1-\theta$ | mixed | two-tier |
