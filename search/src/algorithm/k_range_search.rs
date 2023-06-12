#![cfg(feature = "std")]

use crate::algorithm::*;
use crate::config::*;
use crate::model::*;
use core::iter::FromIterator;
use heapless::Vec;
use std::cmp::min;
use uom::si::ratio::ratio;

use crate::math::calculate_w_vector;

pub struct KRangeSearch {
    model: Model,
    config: Config,
    ri_opt: f64,
    diff_thresh: f64,
    k_list: heapless::Vec<u16, 255>,
}

impl SearchAlgorithm for KRangeSearch {
    fn search(&mut self) -> Config {
        self.k_list = Vec::new();
        let mut opt_config: Config = Config::default();
        let mut ri_opt: f64 = f64::INFINITY;
        // Initialize partitions table
        // Get maximum k that meets PLR constraint
        let (k_max, p_max) = self.get_max_k_p();
        // Vector to store all (k,ri) pairs

        // Only continue if some redundancy is needed to meet PLR constraint
        if p_max > 0  && self.ri_opt > 0.0 {
            // p_opt(k) is monotonicaly increasing. Itearte over all k's in reverse to find all valid values
            let mut p = p_max;
            for k in (1..=k_max).rev() {
                // Each new p is either the same or minues 1 the previous p.
                let mut temp_c: Config = Config::new(k, &[p - 1]);
                while p > 0 && self.model.check_loss_rate(&temp_c) {
                    p -= 1;
                    temp_c.set_np(&[p - 1]);
                }
                if p > 255 {
                    let temp_c: Config = Config::new(k, &[p]);
                    if !self.model.check_loss_rate(&temp_c) {
                        break;
                    }
                }

                // If k and p are known, the maximum Nc is fixed by the delay constrant.
                // The configuration (k,p,nc) meets the D constraint
                let nc = self.get_nc(k, p);

                // Check if delay constraint is met, if not, continue
                let temp_c: Config = Config::new(k, &self.simple_schedule(p, nc).unwrap());
                if !self.model.check_delay(&temp_c) {
                    continue;
                }

                // Use graph algorithm to find the optimal schedule
                let (np, ri): (Vec<u16, 255>, f64) =
                    match self.find_schedule(k as usize, p as usize, nc as usize) {
                        Some(res) => res,
                        None => {
                            let mut np: Vec<u16, 255> = Vec::from_iter(
                                core::iter::repeat(1)
                                    .take(nc.min(p) as usize + 1)
                                    .collect::<Vec<_, 255>>(),
                            );
                            np[0] = 0;

                            let mut temp_c: Config = Config::new(k, &np);
                            let ri = temp_c.ri(&self.model);

                            (np, ri)
                        }
                    };

                if ri - self.ri_opt <= self.diff_thresh * self.ri_opt
                    && self.model.check_data_rate_ri(ri)
                {
                    self.k_list.push(k).unwrap();
                }

                if ri < ri_opt && self.model.check_data_rate_ri(ri) {
                    opt_config.set_config(k, p, nc, &np);
                    opt_config.set_ri(ri);
                    ri_opt = ri;
                }
            }
        } else if k_max > 0 {
            // There are valid configurations which meet PLR constraint with p=0
            opt_config.set_k(k_max);
            opt_config.set_np(&[p_max]);
            opt_config.ri(&self.model);

            // Find the minimum k that still meets the constraints
            for k in (1..=k_max).rev() {
                // Check if PLR constraint is met. If not, break
                let temp_c: Config = Config::new(k, &self.simple_schedule(0, 0).unwrap());
                if !self.model.check_loss_rate(&temp_c) {
                    break;
                }

                // If it meets the constraints, add the block length to the list of valid k's
                self.k_list.push(k).unwrap();
            }
        }
        #[cfg(feature = "verbose")]
        {
            println!("{:?}", opt_config);
        }

        self.config = opt_config;
        self.config.clone()
    }

    //====================================================================
    //============================= I/O ==================================
    //====================================================================

    #[cfg(feature = "std")]
    fn generate_output(&self, elapsed: &str) -> String {
        let k_min;
        let k_max;
        if self.k_list.len() > 0 {
            k_min = format!("{}", self.k_list.iter().min().unwrap());
            k_max = format!("{}", self.k_list.iter().max().unwrap());
        } else {
            k_min = format!("{}", 0);
            k_max = format!("{}", 0);
        }

        self.model.get_model_str()
            + &self.config.get_config_str()[..]
            + elapsed
            + ","
            + &k_min
            + ","
            + &k_max
            + "\n"
    }

    // math will persist (TODO Bare-Metal, Concurrency)
    fn update_model(&mut self, model: &Model) {
        self.model = model.clone();
    }

    #[cfg(feature = "std")]
    fn name(&self) -> String {
        "k_range".to_string()
    }

    #[cfg(feature = "std")]
    fn output_format(&self) -> String {
        String::from("PLR_T(prob),D_T(ms),R_C(bps),p_e(prob),T_s(ms),P_L(B),RTT(ms),D_RS(ms),D_PL(ms),k(pkts),n(pkts),N_c(cycles),N_p(vec_of_pkts),RI(rate),E_t(ns),k_min(pkts),k_max(pkts)\n")
    }
}

impl KRangeSearch {
    pub fn new() -> Self {
        Self {
            model: Model::default(),
            config: Config::default(),
            ri_opt: 0.0,
            diff_thresh: 0.0,
            k_list: heapless::Vec::new(),
        }
    }

    //====================================================================
    //=========================== Helpers ================================
    //====================================================================

    fn simple_schedule(&self, p: u16, nc: u16) -> Option<Vec<u16, 255>> {
        let mut np: Vec<u16, 255> = Vec::from_iter(
            core::iter::repeat(1)
                .take(nc.min(p) as usize + 1)
                .collect::<Vec<_, 255>>(),
        );

        if nc.min(p) > 0 {
            np[0] = 0;
            np[nc.min(p) as usize] = p - (nc.min(p) - 1);
        } else {
            np[0] = p;
        }

        Some(np)
    }

    pub fn set_ri_opt(&mut self, ri_opt: f64, diff_thresh: f64) {
        self.ri_opt = ri_opt;
        self.diff_thresh = diff_thresh;
    }

    pub fn find_schedule(&self, k: usize, p: usize, n_c: usize) -> Option<(Vec<u16, 255>, f64)> {
        if p < n_c {
            return None;
        }

        let p_e = self.model.get_channel_erasure_rate();
        let w = calculate_w_vector(k as u16, p as u16, p_e);

        let mut dp = vec![vec![f64::INFINITY; n_c + 1]; p + 1];
        let mut par = vec![vec![0; n_c + 1]; p + 1];

        let mut lower = 0;
        let mut upper = (p - n_c) as usize;

        for x in lower..=upper {
            dp[x][0] = x as f64;
            par[x][0] = x;
        }

        for y in 1..=n_c as usize {
            lower += 1;
            upper += 1;
            for x in lower..=upper {
                for prev in lower - 1..x {
                    let step = x - prev;
                    let current = dp[prev][y - 1] + step as f64 * w[prev];
                    if current < dp[x][y] {
                        dp[x][y] = current;
                        par[x][y] = step;
                    }
                }
            }
        }

        let mut n_p = Vec::new();
        let mut x = p;

        for y in (0..=n_c).rev() {
            let step = par[x][y];
            n_p.push(step as u16).unwrap();
            x -= step;
        }

        n_p.reverse();
        let ri = dp[p][n_c] / k as f64;
        Some((n_p, ri))
    }

    fn k_binary_search(&self, max: u16) -> u16 {
        let mut min = 0;
        let mut max = max;

        while max - min > 1 {
            let mid = min + (max - min) / 2;
            let temp_c: Config = Config::new(mid, &[255 - mid]);
            if self.model.check_loss_rate(&temp_c) {
                min = mid;
            } else {
                max = mid;
            }
        }

        let temp_c: Config = Config::new(max, &[255 - max]);
        if self.model.check_loss_rate(&temp_c) {
            return max;
        }
        min
    }

    fn p_binary_search(&self, k: u16) -> u16 {
        let mut min = 0;
        let mut max = 255 - k;

        while max - min > 1 {
            let mid = min + (max - min) / 2;
            let temp_c: Config = Config::new(k, &[mid]);
            if self.model.check_loss_rate(&temp_c) {
                max = mid;
            } else {
                min = mid;
            }
        }

        max
    }

    fn get_max_k_p(&self) -> (u16, u16) {
        let mut k_max = min(self.derive_k_max(), 255);
        k_max = min(k_max, self.k_binary_search(k_max));

        // IF the maximum blocklength is 0, the solution space is empty
        if k_max == 0 {
            return (0, 0);
        }

        let mut p_max = self.p_binary_search(k_max);

        let mut config: Config = Config::new(k_max, &[p_max]);
        while !self.model.check_delay(&config) && k_max > 0 {
            k_max -= 1;
            if p_max - 1 <= 255 {
                let temp_c: Config = Config::new(k_max, &[p_max - 1]);
                if self.model.check_loss_rate(&temp_c) {
                    p_max -= 1;
                }
            }

            config.set_k(k_max);
            config.set_np(&[p_max]);
        }

        (k_max, p_max)
    }

    // Get maximum absolute blocklength (p=Nc=0).
    fn derive_k_max(&self) -> u16 {
        ((self.model.get_target_delay()
            - (self.model.get_round_trip_time() + self.model.get_response_delay()) / 2.0)
            / self.model.get_source_packet_interval())
        .get::<ratio>()
        .floor() as u16
    }

    fn get_nc(&self, k: u16, p: u16) -> u16 {
        let nc = ((self.model.get_target_delay()
            - k as f64 * self.model.get_source_packet_interval()
            - p as f64 * self.model.get_average_packet_length()
                / self.model.get_channel_data_rate()
            - (self.model.get_round_trip_time() + self.model.get_response_delay()) / 2.0)
            / (self.model.get_round_trip_time()
                + self.model.get_response_delay()
                + self.model.get_packet_loss_detection_delay()))
        .get::<ratio>()
        .floor() as u16;

        nc.min(p)
    }
}
