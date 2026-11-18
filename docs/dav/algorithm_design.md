# DAV Algorithm Design

## Candidate policy: AMCR

Working name: **Adaptive Marginal Cost Reduction (AMCR)**. This is a proposed heuristic name, not a standard established algorithm and not yet supported by an approximation guarantee.

### Decision loop

1. Maintain the current dependency DAG, versions, statuses, posterior error beliefs, and remaining budgets.
2. Enumerate eligible verification candidates.
3. For each candidate, estimate expected objective with and without verification.
4. Include verifier outcomes, repair success/failure, recomputation success/failure, overlapping affected descendants, and both budgets.
5. Select the feasible candidate with the largest positive expected objective reduction per verification cost.
6. Execute verification and update posterior beliefs.
7. On detected error, repair if recovery budget permits.
8. On repair success, create a new version, invalidate only descendants that consumed the old version, and recompute them topologically.
9. On repair/recompute failure or budget exhaustion, block dependent tasks.
10. Recompute candidate values from the new state.

## Pseudocode

```text
DAV(G, B_ver, B_rec):
    initialize node beliefs, statuses, versions, lineage
    remaining_ver = B_ver
    remaining_rec = B_rec

    while a feasible verification may improve expected objective:
        for each eligible node v:
            J0 = EstimateExpectedCost(state, action=None)
            Jv = EstimateExpectedCost(state, action=VERIFY(v))
            gain[v] = J0 - Jv
            score[v] = gain[v] / verification_cost[v]

        choose feasible node with largest positive score
        observe verifier result; spend verification budget
        update posterior beliefs using sensitivity/specificity

        if evidence indicates error:
            if repair cost exceeds remaining recovery budget:
                mark node BLOCKED; block descendants; continue

            execute probabilistic repair; spend recovery budget
            if repair fails:
                mark node BLOCKED; block descendants; continue

            create new source version
            affected = descendants whose lineage consumed old version
            mark affected nodes STALE

            for u in topological_order(affected):
                if any required parent is INVALID, STALE, or BLOCKED:
                    mark u BLOCKED; continue
                if recomputation cost exceeds remaining recovery budget:
                    mark u BLOCKED; block dependent tasks; continue

                execute recomputation; spend recovery budget
                if recomputation succeeds:
                    create new version; mark u VERIFIED
                else:
                    mark u BLOCKED; block dependent tasks

        update graph, beliefs, statuses, and budgets

    return final state, action trace, costs, residual risk
```

## Expected-cost estimator

The most important component is `EstimateExpectedCost`. It should estimate future cost under the current state and candidate action. The first implementation can use Monte Carlo rollouts; exact enumeration on tiny DAGs is the oracle.

Use common random numbers when comparing candidate actions where practical, to reduce Monte Carlo variance.

## Feasibility

Before each action, enforce hard remaining-budget checks. An action whose possible recovery cannot be completed may still be evaluated, but its failure branch must terminate safely with blocked descendants and corresponding residual task loss. Never assume repair is guaranteed.
