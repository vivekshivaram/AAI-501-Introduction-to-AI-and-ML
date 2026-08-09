# Experiment Results Summary
## Quick Reference Guide

**Analysis Date:** 2026-08-09  
**Total Experiments Completed:** 8  
**Total Simulations Run:** 54 individual configurations

---

## 📁 Generated Files

### Visualization Outputs
Located in: `data/outputs/`

1. **experiment_fleet_size.png** - Fleet size impact (5 to 50 vehicles)
2. **experiment_order_volume.png** - Order volume impact (1 to 20 orders/tick)
3. **experiment_routing_speed.png** - Routing speed impact (30 to 100 km/h)
4. **experiment_vehicle_capacity.png** - Vehicle capacity impact (0.5x to 3.0x)
5. **experiment_q_alpha.png** - Q-learning alpha impact (0.05 to 0.50)
6. **experiment_q_epsilon.png** - Q-learning epsilon impact (0.05 to 0.50)
7. **experiment_fleet_order_stress.png** - Fleet-order stress test (12 combinations)
8. **experiment_speed_capacity_tradeoff.png** - Speed-capacity trade-off (9 combinations)

### Data Files
9. **experiment_results.json** - Complete numerical results for all experiments

### Analysis Documents
10. **EXPERIMENT_INTERPRETATIONS.md** - Detailed interpretations for each subplot (THIS FILE)
11. **EXPERIMENT_INSIGHTS.md** - Explains metric biases and corrections
12. **EXPERIMENTS.md** - User guide for running experiments

---

## 🎯 Key Findings at a Glance

### Most Important Metrics

| Metric | Formula | What It Measures | Ideal Value |
|--------|---------|------------------|-------------|
| **Delivery Rate** | Delivered / (Delivered + Rejected) | Completion quality | >75% |
| **Dispatch Efficiency** | (Delivered + Rejected + In-Transit) / Total Sampled | Capacity utilization | >50% |
| **Throughput** | Delivered / MAX_SIMULATION_STEPS | System performance | >1.0/tick |
| **Avg Delivery Time** | Mean time from dispatch to delivery | Speed of service | <6 ticks |

### Critical Parameters (Ranked by Impact)

1. **Fleet Size** ⭐⭐⭐⭐⭐ - MOST CRITICAL
   - Directly controls capacity (dispatch efficiency 17% → 72%)
   - Scales throughput linearly (0.4 → 1.15 deliveries/tick)
   - Minimal impact on quality (delivery rate stays 70-80%)

2. **Order Volume** ⭐⭐⭐⭐
   - Saturation at 10+ orders/tick for 20-vehicle fleet
   - Throughput caps at ~1.4 deliveries/tick
   - Quality degrades beyond capacity (90% → 55%)

3. **Q-Learning Parameters** ⭐⭐
   - Optimal: alpha=0.15, epsilon=0.25
   - Affects pricing behavior (surge 1.09x → 1.31x)
   - Minimal impact on operational metrics

4. **Routing Speed** ⭐
   - Minor gains from 30 → 60 km/h (~12% throughput)
   - Plateaus beyond 60 km/h (diminishing returns)
   - Low investment priority

5. **Vehicle Capacity** (no stars)
   - No measurable impact in single-package scenario
   - Would matter for multi-package optimization
   - Not recommended for investment

---

## 🏆 Optimal Configurations

### Balanced Configuration (Recommended)
- **Fleet Size:** 20 vehicles
- **Order Volume:** 5 orders/tick
- **Routing Speed:** 60 km/h
- **Q-Alpha:** 0.15
- **Q-Epsilon:** 0.25
- **Results:** 81% delivery rate, 38% dispatch efficiency, 0.85 throughput

### High-Capacity Configuration
- **Fleet Size:** 50 vehicles
- **Order Volume:** 10 orders/tick
- **Results:** 79% delivery rate, 87% dispatch efficiency, 1.25 throughput

### System Breaking Point
- **Fleet Size:** 20 vehicles
- **Order Volume:** 20 orders/tick
- **Results:** 55% delivery rate, 16% dispatch efficiency (OVERLOADED)

---

## 📊 Performance Benchmarks

### Delivery Rate Benchmarks
- **Excellent:** >80% (Fleet=20, Orders=5)
- **Good:** 70-80% (Most balanced configurations)
- **Acceptable:** 60-70% (Mild overload)
- **Poor:** <60% (Severe overload, add fleet capacity)

### Dispatch Efficiency Benchmarks
- **Excellent:** >70% (Fleet=50, high capacity)
- **Good:** 50-70% (Fleet=30, moderate capacity)
- **Acceptable:** 30-50% (Fleet=20, baseline)
- **Poor:** <30% (Fleet<15, insufficient capacity)

### Throughput Benchmarks
- **Maximum Observed:** 1.45 deliveries/tick (Fleet=20, Orders=15)
- **High:** 1.0-1.4 deliveries/tick
- **Moderate:** 0.6-1.0 deliveries/tick
- **Low:** <0.6 deliveries/tick (underutilized)

---

## 💡 Strategic Recommendations

### When to Scale Fleet Size
**Symptoms of Insufficient Fleet:**
- Dispatch efficiency <30%
- Pending orders >100
- Fleet utilization >90%

**Action:** Add 10-20 vehicles to increase capacity

### When to Reduce Order Volume
**Symptoms of Overload:**
- Delivery rate <60%
- Dispatch efficiency <20%
- Throughput plateauing despite more orders

**Action:** Implement demand management (surge pricing, order throttling)

### When to Optimize Q-Learning
**Current parameters (alpha=0.15, epsilon=0.25) are optimal** ✅
- Only adjust if pricing behavior changes significantly
- Monitor average surge: should be 1.15-1.25x for balanced load

### When NOT to Invest
❌ **Don't invest in:**
- Faster vehicles beyond 60 km/h (diminishing returns)
- Larger vehicles for single-package scenarios (no impact)
- Speed-capacity trade-offs (no synergistic benefits)

✅ **Do invest in:**
- Additional vehicles (highest ROI)
- MILP optimization improvements (reduce dispatch latency)
- Demand forecasting (prevent overload)

---

## 🔬 Methodology Notes

### Simulation Parameters
- **Duration:** 20 ticks per simulation
- **Random Seed:** 42 (reproducible results)
- **Order Pool:** 3,839 valid orders (same pickup ≠ delivery)
- **CNN Rejection Rate:** ~5% (package damage inspection)

### Metric Corrections Applied
**Previous Bias:** Delivery rate included in-transit orders, favoring small fleets  
**Correction:** Delivery Rate = Delivered / (Delivered + Rejected)

**New Metrics Added:**
- **Dispatch Efficiency:** Measures capacity utilization
- **Throughput:** Measures actual delivery performance

### Stochastic Variation
**Sources of Randomness:**
1. Order sampling (different orders each run)
2. CNN package rejection (~5%)
3. MILP solver optimality gaps
4. Q-learning exploration (epsilon-greedy)

**Impact:** Delivery rate varies ±10-15% between identical configurations  
**Recommendation:** Run 3-5 trials and average results for production decisions

---

## 📈 Next Steps

### For Further Analysis
1. **Multi-trial experiments:** Run 5 trials per configuration, compute confidence intervals
2. **Cost-benefit analysis:** Calculate cost per delivery for each fleet size
3. **Longer simulations:** Extend to 50-100 ticks to eliminate in-transit bias
4. **Multi-package orders:** Test larger vehicle capacity benefits
5. **Dynamic pricing:** Compare Q-learning vs. rule-based surge pricing

### For Production Deployment
1. **Monitor KPIs:** Track delivery rate, dispatch efficiency, throughput daily
2. **Set Alerts:** Dispatch efficiency <30%, delivery rate <60%
3. **Autoscaling:** Add vehicles when queue length >100 orders
4. **A/B Testing:** Test Q-epsilon 0.18 vs. 0.25 in production
5. **Customer Impact:** Correlate surge multiplier with order acceptance rate

---

## 📞 Support

**Documentation:**
- [EXPERIMENT_INTERPRETATIONS.md](EXPERIMENT_INTERPRETATIONS.md) - Detailed subplot analysis
- [EXPERIMENT_INSIGHTS.md](EXPERIMENT_INSIGHTS.md) - Metric bias explanations
- [EXPERIMENTS.md](EXPERIMENTS.md) - How to run experiments

**Run Experiments:**
```bash
cd AAI-501-Introduction-to-AI-and-ML
python -m src.experiment_runner --all
```

**View Results:**
```bash
ls data/outputs/experiment_*.png
cat data/outputs/experiment_results.json
```

---

**Last Updated:** 2026-08-09  
**Experiment Framework Version:** 2.0 (with corrected metrics)
