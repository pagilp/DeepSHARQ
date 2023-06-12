use crate::algorithm::SearchAlgorithm;
use crate::config::Config;
use crate::model::*;

use core::iter::FromIterator;
use heapless::Vec;

#[derive(Default)]
pub struct DeepHecSearch {
    model: Model,
    config: Config,
    k: u16,
    p: u16,
    nc: u16
}

impl SearchAlgorithm for DeepHecSearch {
    fn search(&mut self) -> Config {

        // k, p and Nc must have been given out of the search loop after querying the NN
        let np = self.generate_repair_schedule(self.k, self.k+self.p, self.nc);

        // Check if the found configuration is valid
        let mut opt_config: Config = Config::new(self.k, &np);
        if !self.model.is_valid(&mut opt_config) {
            opt_config = Config::default();
        } else {
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

    // math will persist (TODO Bare-Metal, Concurrency)
    fn update_model(&mut self, model: &Model) {
        self.model = model.clone();
    }

    #[cfg(feature = "std")]
    fn generate_output(&self, elapsed: &str) -> String {
        self.model.get_model_str() + &self.config.get_config_str()[..] + elapsed + "\n"
    }

    #[cfg(feature = "std")]
    fn name(&self) -> String {
        "deephec".to_string()
    }

    #[cfg(feature = "std")]
    fn output_format(&self) -> String {
        String::from("PLR_T(prob),D_T(ms),R_C(bps),p_e(prob),T_s(ms),P_L(B),RTT(ms),D_RS(ms),D_PL(ms),k(pkts),n(pkts),N_c(cycles),N_p(vec_of_pkts),RI(rate),E_t(ns)\n")
    }
}

impl DeepHecSearch {
    pub fn new() -> DeepHecSearch {
        DeepHecSearch::default()
    }

    pub fn set_params(&mut self, k: u16, p: u16, nc: u16) {
        self.k = k;
        self.p = p;
        self.nc = nc;
    }

    fn generate_repair_schedule(&self, k: u16, n: u16, nc: u16) -> Vec<u16, 255> {
        let p = n - k;
        let mut schedule = Vec::from_iter(
            core::iter::repeat(1)
                .take(nc.min(p) as usize + 1)
                .collect::<Vec<_, 255>>(),
        );
        if nc.min(p) > 0 {
            schedule[0] = 0;
            schedule[nc.min(p) as usize] = p - (nc.min(p) - 1);
        } else {
            schedule[0] = p;
        }

        for i in (1..=nc.min(p) as usize).rev() {
            let mut new_schedule = schedule.clone();
            while new_schedule[i] > 1 {
                new_schedule[i] -= 1;
                new_schedule[i - 1] += 1;

                let ri_old = self.model.get_ri(&Config::new(k, &schedule));
                let ri_new = self.model.get_ri(&Config::new(k, &new_schedule));

                if ri_new < ri_old {
                    schedule = new_schedule.clone();
                } else {
                    break;
                }
            }
        }

        schedule
    }
}
