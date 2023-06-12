use crate::algorithm::*;
use crate::config::*;
use crate::model::*;
use core::iter::FromIterator;
use heapless::Vec;
use uom::si::ratio::ratio;

#[derive(Default)]
pub struct FastSearch {
    model: Model,
    config: Config,
}

impl SearchAlgorithm for FastSearch {
    fn search(&mut self) -> Config {
        let mut opt_config: Config = Config::default();

        for nc in (0..=self.derive_nc_max()).rev() {
            for k in (1..=self.derive_k_max(nc).min(255)).rev() {
                // todo lower nc_max inputs lead to higher k leads to higher n leads to higher effective nc
                let (n,valid) = self.derive_n(k);
                if !valid {
                    continue;
                }

                // Obtain configuration with optimal repair schedule
                let mut new_config = self.generate_repair_schedule(k, n, nc);

                if new_config.ri(&self.model) < opt_config.ri(&self.model)
                    && self.model.is_valid(&mut new_config)
                {
                    // config is valid since it passes delay, data_rate and erasure_rate
                    opt_config = new_config;
                }

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
        self.model.get_model_str() + &self.config.get_config_str()[..] + elapsed + "\n"
    }

    // math will persist (TODO Bare-Metal, Concurrency)
    fn update_model(&mut self, model: &Model) {
        self.model = model.clone();
    }

    #[cfg(feature = "std")]
    fn name(&self) -> String {
        "fast".to_string()
    }

    #[cfg(feature = "std")]
    fn output_format(&self) -> String {
        String::from("PLR_T(prob),D_T(ms),R_C(bps),p_e(prob),T_s(ms),P_L(B),RTT(ms),D_RS(ms),D_PL(ms),k(pkts),n(pkts),N_c(cycles),N_p(vec_of_pkts),RI(rate),E_t(ns)\n")
    }
}

impl FastSearch {
    pub fn new() -> FastSearch {
        FastSearch::default()
    }

    //====================================================================
    //=========================== Helpers ================================
    //====================================================================

    fn generate_repair_schedule(&self, k: u16, n: u16, nc: u16) -> Config {
        let r = n - k;
        let mut np: Vec<u16,255> = Vec::from_iter(
            core::iter::repeat(1)
                .take(nc.min(r) as usize + 1)
                .collect::<Vec<_, 255>>(),
        );

        if nc.min(r) > 0 {
            np[0] = 0;
            np[nc.min(r) as usize] = r - (nc.min(r) - 1);
        } else {
            np[0] = r;
        }

        let mut config = Config::new(k, &np);

        let (mut ri,mut prob_c_vec) = self.model.get_ri(&config);

        for i in (1..=nc.min(r) as usize).rev() {
            let mut temp_config = config.clone();
            while temp_config.np(i as usize) > 1 {

                temp_config.one_parity_forward(i,i-1);

                let ri_new = self.model.update_ri(&temp_config,&mut prob_c_vec, i-1 as usize, i as usize);

                if ri_new < ri {
                    ri = ri_new;
                    config = temp_config.clone();
                } else {
                    break;
                }
            }
        }

        config
    }

    fn derive_nc_max(&self) -> u16 {
        let min_c = Config::new(1, &[0, 1]);
        ((self.model.get_target_delay() - self.model.get_fec_delay(&min_c))
            / self.model.get_arq_delay(&min_c))
        .get::<ratio>() as u16
    }

    fn derive_k_max(&self, nc: u16) -> u16 {
        let min_c = Config::new(1, &[0, 1]);

        ((self.model.get_target_delay()
            - (self.model.get_round_trip_time() + self.model.get_response_delay()) / 2.0
            - nc as f64 * self.model.get_arq_delay(&min_c))
            / self.model.get_source_packet_interval())
        .get::<ratio>() as u16
    }

    fn derive_n(&self, k: u16) -> (u16,bool) {
        for n in k..=255 {
            let temp_c = Config::new(k, &[n - k]);
            if self.model.check_loss_rate(&temp_c) {
                return (n,true);
            }
        }
        (255,false)
    }
}
