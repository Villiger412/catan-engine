use crate::board::PLAYER_COUNT;

/// Results from a batch of simulations.
#[derive(Debug, Clone)]
pub struct SimulationStats {
    /// Number of simulations completed.
    pub n: u32,
    /// Win counts per player.
    pub wins: [u32; PLAYER_COUNT],
    /// Draw/timeout count.
    pub draws: u32,
    /// Sum of game lengths (for average).
    pub total_turns: u64,
}

impl SimulationStats {
    pub fn new() -> Self {
        Self {
            n: 0,
            wins: [0; PLAYER_COUNT],
            draws: 0,
            total_turns: 0,
        }
    }

    pub fn record_win(&mut self, winner: Option<u8>, turns: u16) {
        self.n += 1;
        self.total_turns += turns as u64;
        if let Some(w) = winner {
            self.wins[w as usize] += 1;
        } else {
            self.draws += 1;
        }
    }

    /// Merge another stats batch into this one.
    pub fn merge(&mut self, other: &SimulationStats) {
        self.n += other.n;
        self.draws += other.draws;
        self.total_turns += other.total_turns;
        for i in 0..PLAYER_COUNT {
            self.wins[i] += other.wins[i];
        }
    }

    /// Win probability estimate for each player.
    pub fn win_probabilities(&self) -> [f64; PLAYER_COUNT] {
        let n = self.n as f64;
        if n == 0.0 {
            return [0.0; PLAYER_COUNT];
        }
        let mut probs = [0.0; PLAYER_COUNT];
        for i in 0..PLAYER_COUNT {
            probs[i] = self.wins[i] as f64 / n;
        }
        probs
    }

    /// 95% confidence interval for each player's win probability.
    /// Returns (lower, upper) bounds.
    pub fn confidence_intervals_95(&self) -> [(f64, f64); PLAYER_COUNT] {
        let z = 1.96; // z_{0.025} for 95% CI
        let n = self.n as f64;
        if n == 0.0 {
            return [(0.0, 1.0); PLAYER_COUNT];
        }

        let probs = self.win_probabilities();
        let mut cis = [(0.0, 1.0); PLAYER_COUNT];
        for i in 0..PLAYER_COUNT {
            let p = probs[i];
            let se = (p * (1.0 - p) / n).sqrt();
            let margin = z * se;
            cis[i] = ((p - margin).max(0.0), (p + margin).min(1.0));
        }
        cis
    }

    /// Maximum half-width of the 95% CI across all players.
    pub fn max_margin(&self) -> f64 {
        let cis = self.confidence_intervals_95();
        cis.iter()
            .map(|(lo, hi)| (hi - lo) / 2.0)
            .fold(0.0f64, f64::max)
    }

    /// Average game length in turns.
    pub fn avg_turns(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.total_turns as f64 / self.n as f64
        }
    }

    /// Minimum simulations needed for target margin of error at 95% confidence.
    /// Uses worst-case variance (p=0.5).
    pub fn required_simulations(target_margin: f64) -> u32 {
        let z = 1.96;
        let max_variance = 0.25; // p(1-p) maximized at p=0.5
        let n = (z * z * max_variance) / (target_margin * target_margin);
        n.ceil() as u32
    }
}

/// Sison-Glaz simultaneous confidence intervals for multinomial proportions.
/// Provides tighter intervals than Bonferroni when sum(p_i) = 1.
pub fn sison_glaz_ci(wins: &[u32; PLAYER_COUNT], n: u32, alpha: f64) -> [(f64, f64); PLAYER_COUNT] {
    // Simplified implementation: use Bonferroni-adjusted Wald intervals
    // (proper Sison-Glaz requires iterative computation)
    let k = PLAYER_COUNT as f64;
    let z = normal_quantile(1.0 - alpha / (2.0 * k)); // Bonferroni correction
    let nf = n as f64;

    let mut cis = [(0.0, 1.0); PLAYER_COUNT];
    for i in 0..PLAYER_COUNT {
        let p = wins[i] as f64 / nf;
        let se = (p * (1.0 - p) / nf).sqrt();
        let margin = z * se;
        cis[i] = ((p - margin).max(0.0), (p + margin).min(1.0));
    }
    cis
}

/// Approximate normal quantile (inverse CDF) using rational approximation.
fn normal_quantile(p: f64) -> f64 {
    // Abramowitz and Stegun approximation 26.2.23
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let result = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 {
        -result
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_required_simulations() {
        let n = SimulationStats::required_simulations(0.02);
        assert!(n >= 2400 && n <= 2500, "Expected ~2401, got {}", n);
    }

    #[test]
    fn test_confidence_intervals() {
        let mut stats = SimulationStats::new();
        // 1000 games, player 0 wins 250
        for _ in 0..250 {
            stats.record_win(Some(0), 70);
        }
        for _ in 0..250 {
            stats.record_win(Some(1), 70);
        }
        for _ in 0..250 {
            stats.record_win(Some(2), 70);
        }
        for _ in 0..250 {
            stats.record_win(Some(3), 70);
        }

        let probs = stats.win_probabilities();
        assert!((probs[0] - 0.25).abs() < 0.001);

        let cis = stats.confidence_intervals_95();
        // With n=1000 and p=0.25, margin should be ~0.027
        let margin = (cis[0].1 - cis[0].0) / 2.0;
        assert!(margin > 0.02 && margin < 0.04, "Margin: {}", margin);
    }

    #[test]
    fn test_normal_quantile() {
        let z = normal_quantile(0.975);
        assert!((z - 1.96).abs() < 0.01, "z_0.975 = {}", z);
    }

    #[test]
    fn test_merge() {
        let mut a = SimulationStats::new();
        let mut b = SimulationStats::new();
        a.record_win(Some(0), 70);
        b.record_win(Some(1), 80);
        a.merge(&b);
        assert_eq!(a.n, 2);
        assert_eq!(a.wins[0], 1);
        assert_eq!(a.wins[1], 1);
    }
}
