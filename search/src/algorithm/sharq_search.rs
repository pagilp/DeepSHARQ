use crate::algorithm::*;
use crate::config::*;
use crate::model::*;
use core::cmp::min;
use core::iter::FromIterator;
use heapless::Vec;
use uom::si::ratio::ratio;

#[cfg(feature = "opt_schedule")]
use crate::math::MATH;

#[cfg(not(feature = "std"))]
use uom::num_traits::float::FloatCore;

pub struct SharqSearch {
    model: Model,
    config: Config,
}

impl SearchAlgorithm for SharqSearch {
    fn search(&mut self) -> Config {
        let mut opt_config: Config = Config::default();
        let mut ri_opt: f64 = f64::INFINITY;

        // Get maximum k that meets PLR constraint
        let (k_max, p_max) = self.get_max_k_p();

        // Only continue if some redundancy is needed to meet PLR constraint
        if p_max > 0 {
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

                #[cfg(feature = "opt_schedule")]
                {
                    // Use graph algorithm to find the optimal schedule
                    let (np, ri): (Vec<u16, 256>, f64) =
                        match self.find_schedule(k as usize, p as usize, nc as usize) {
                            Some(res) => res,
                            None => {
                                let mut np: Vec<u16, 256> = Vec::from_iter(
                                    core::iter::repeat(1)
                                        .take(nc.min(p) as usize + 1)
                                        .collect::<Vec<_, 256>>(),
                                );
                                np[0] = 0;

                                let mut temp_c: Config = Config::new(k, &np);
                                let ri = temp_c.ri(&self.model);

                                (np, ri)
                            }
                        };

                    if ri < ri_opt && self.model.check_data_rate_ri(ri) {
                        opt_config.set_config(k, p, nc, &np);
                        opt_config.set_ri(ri);
                        ri_opt = ri;
                    }
                }

                #[cfg(feature = "simple_schedule")]
                {
                    // Get simple schedule
                    let np: Vec<u16, 256> = self.simple_schedule(p, nc).unwrap();
                    let temp_c: Config = Config::new(k, &np);
                    let ri: f64 = self.model.get_ri_pp(&temp_c);

                    if ri < ri_opt && self.model.check_data_rate_ri(ri) {
                        opt_config.set_config(k, p, nc, &np);
                        opt_config.set_ri(ri);
                        ri_opt = ri;
                    }
                }
            }
        } else if k_max > 0 {
            opt_config.set_k(k_max);
            opt_config.set_np(&[p_max]);
            opt_config.ri(&self.model);
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
        self.model.get_model_str() + &self.config.get_config_str()[..] + elapsed + "\n"
    }

    // math will persist (TODO Bare-Metal, Concurrency)
    fn update_model(&mut self, model: &Model) {
        self.model = model.clone();
    }

    #[cfg(feature = "std")]
    fn name(&self) -> String {
        "sharq".to_string()
    }

    #[cfg(feature = "std")]
    fn output_format(&self) -> String {
        String::from("PLR_T(prob),D_T(ms),R_C(bps),p_e(prob),T_s(ms),P_L(B),RTT(ms),D_RS(ms),D_PL(ms),k(pkts),n(pkts),N_c(cycles),N_p(vec_of_pkts),RI(rate),E_t(ns)\n")
    }
}

impl SharqSearch {
    pub fn new() -> Self {
        Self {
            model: Model::default(),
            config: Config::default(),
        }
    }

    //====================================================================
    //=========================== Helpers ================================
    //====================================================================

    fn simple_schedule(&self, p: u16, nc: u16) -> Option<Vec<u16, 256>> {
        let mut np: Vec<u16, 256> = Vec::from_iter(
            core::iter::repeat(1)
                .take(nc.min(p) as usize + 1)
                .collect::<Vec<_, 256>>(),
        );

        if nc.min(p) > 0 {
            np[0] = 0;
            np[nc.min(p) as usize] = p - (nc.min(p) - 1);
        } else {
            np[0] = p;
        }

        Some(np)
    }

    #[cfg(feature = "opt_schedule")]
    pub fn find_schedule(&self, k: usize, p: usize, n_c: usize) -> Option<(Vec<u16, 256>, f64)> {
        use core::iter::repeat;
        use heapless::Vec;

        if p < n_c {
            return None;
        }

        let p_e = self.model.get_channel_erasure_rate();
        let w = MATH.calculate_w_vector(k as u16, p as u16, n_c as u16, p_e);

        let mut dp: Vec<Vec<f64, 256>, 256> =
            Vec::from_iter(repeat(Vec::from_iter(repeat(f64::INFINITY).take(n_c + 1))).take(p + 1));
        let mut par: Vec<Vec<_, 256>, 256> =
            Vec::from_iter(repeat(Vec::from_iter(repeat(0).take(n_c + 1))).take(p + 1));

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
