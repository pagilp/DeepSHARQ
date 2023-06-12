use crate::algorithm::*;
use crate::config::*;
use core::iter::FromIterator;
use heapless::Vec;
use uom::si::ratio::ratio;

#[cfg(feature = "opt_schedule")]
use crate::math::*;

#[cfg(not(feature = "std"))]
use uom::num_traits::float::FloatCore;

#[cfg(feature = "std")]
use std::time::Instant;

mod inference {
    use core::marker::PhantomData;

    use crate::model::Model;
    #[cfg(not(feature = "std"))]
    use uom::num_traits::float::FloatCore;

    #[cfg(not(feature = "std"))]
    const MODEL: &[u8; 27312] = &include_bytes!("../../models/deepsharq_model.tflite");
    #[cfg(not(feature = "std"))]
    const TENSOR_ARENA_SIZE: usize = 4 * 1024;

    pub struct Inference<'a> {
        #[cfg(feature = "std")]
        interpreter: tflitec::interpreter::Interpreter,
        #[cfg(not(feature = "std"))]
        arena: [u8; TENSOR_ARENA_SIZE],
        #[cfg(not(feature = "std"))]
        model: &'a tfmicro::Model,
        _d: PhantomData<&'a ()>,
    }

    impl<'a> Inference<'a> {
        #[cfg(feature = "std")]
        pub(super) fn new() -> Self {
            use tflitec::interpreter::{Interpreter, Options};
            use tflitec::tensor;

            // Create interpreter options
            let mut options = Options::default();
            options.thread_count = 1;
            // Create NN Model
            let model = tflitec::model::Model::new("models/deepsharq_model.tflite").unwrap();
            // Create interpreter
            let interpreter = Interpreter::new(&model, Some(options)).unwrap();
            // Resize input
            let input_shape = tensor::Shape::new(vec![1, 6]);
            interpreter.resize_input(0, input_shape).unwrap();
            // Allocate tensors if you just created Interpreter or resized its inputs
            interpreter.allocate_tensors().unwrap();

            Self {
                interpreter,
                _d: Default::default(),
            }
        }

        #[cfg(not(feature = "std"))]
        pub(super) fn new() -> Self {
            let model = tfmicro::Model::from_buffer(&MODEL[..]).expect("Model creation failed");

            // Create tensor arena
            let arena: [u8; TENSOR_ARENA_SIZE] = [0; TENSOR_ARENA_SIZE];
            Self {
                arena,
                model,
                _d: Default::default(),
            }
        }

        #[cfg(feature = "std")]
        pub(super) fn infer_k(&mut self, model: &Model) -> u16 {
            let input_tensor = self.interpreter.input(0).unwrap();
            let output_tensor = self.interpreter.output(0).unwrap();
            let input_vec = Self::model_to_vec(model);
            assert!(input_tensor.set_data(&input_vec[..]).is_ok());
            assert!(self.interpreter.invoke().is_ok());
            let output: &[f32] = output_tensor.data::<f32>();

            output
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.total_cmp(y.1))
                .unwrap()
                .0 as u16
        }

        #[cfg(not(feature = "std"))]
        pub(super) fn infer_k(&mut self, model: &Model) -> u16 {
            let op_resolver = tfmicro::AllOpResolver::new();
            let mut interpreter =
                tfmicro::MicroInterpreter::new(&self.model, op_resolver, &mut self.arena[..])
                    .unwrap();

            let input_vec = Self::model_to_vec(&model);
            interpreter.input(0, &input_vec[..]).unwrap();
            interpreter.invoke().unwrap();
            let output = interpreter.output(0).as_data::<f32>();

            output
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.total_cmp(y.1))
                .unwrap()
                .0 as u16
        }

        fn model_to_vec(model: &Model) -> [f32; 6] {
            use uom::si::information_rate::bit_per_second;
            use uom::si::time::millisecond;
            let z_norm_plr: f32 =
                (model.get_target_erasure_rate() as f32 - 0.001872892553140462) / 0.002525792;
            let z_norm_dt: f32 = (model.get_target_delay().get::<millisecond>() as f32
                - 279.1865054035207)
                / 266.1068757110481;
            let z_norm_rc: f32 = (model.get_channel_data_rate().get::<bit_per_second>() as f32
                - 1369340110.210422)
                / 2275899918.247129;
            let z_norm_pe: f32 = (model.get_channel_erasure_rate() as f32 - 0.02389486158205640)
                / 0.04230245758583454;
            let z_norm_ts: f32 = (model.get_source_packet_interval().get::<millisecond>() as f32
                - 14.03923188464618)
                / 21.70630890225326;
            let z_norm_rtt: f32 = (model.get_round_trip_time().get::<millisecond>() as f32
                - 21.15978850815602)
                / 24.39285640663291;
            [
                z_norm_plr, z_norm_dt, z_norm_rc, z_norm_pe, z_norm_ts, z_norm_rtt,
            ]
        }
    }
}

pub struct DeepSharqSearch<'a> {
    model: Model,
    config: Config,
    inference: inference::Inference<'a>,
    #[cfg(feature = "std")]
    elapsed_nn: u128,
    #[cfg(feature = "std")]
    elapsed_alg: u128,
}

impl<'a> SearchAlgorithm for DeepSharqSearch<'a> {
    fn search(&mut self) -> Config {
        #[cfg(feature = "std")]
        {
            let now_nn = Instant::now();
            let k = self.inference.infer_k(&self.model);
            self.elapsed_nn = now_nn.elapsed().as_nanos();

            let now_alg = Instant::now();
            let config = self.config_from_k(k);
            self.elapsed_alg = now_alg.elapsed().as_nanos();

            config
        }

        #[cfg(not(feature = "std"))]
        {
            let k = self.inference.infer_k(&self.model);
            self.config_from_k(k)
        }
    }

    #[cfg(feature = "std")]
    fn generate_output(&self, _elapsed: &str) -> String {
        let elapsed = (self.elapsed_nn + self.elapsed_alg).to_string();
        let elapsed_nn = self.elapsed_nn.to_string();
        let elapsed_alg = self.elapsed_alg.to_string();
        self.model.get_model_str() + &self.config.get_config_str()[..] + &elapsed + "," + &elapsed_nn + "," + &elapsed_alg + "\n"
    }

    // math will persist (TODO Bare-Metal, Concurrency)
    fn update_model(&mut self, model: &Model) {
        self.model = model.clone();
    }

    #[cfg(feature = "std")]
    fn name(&self) -> String {
        "deepsharq".to_string()
    }

    #[cfg(feature = "std")]
    fn output_format(&self) -> String {
        String::from("PLR_T(prob),D_T(ms),R_C(bps),p_e(prob),T_s(ms),P_L(B),RTT(ms),D_RS(ms),D_PL(ms),k(pkts),n(pkts),N_c(cycles),N_p(vec_of_pkts),RI(rate),E_t(ns),E_t_nn(ns),E_t_alg(ns)\n")
    }
}

impl<'a> DeepSharqSearch<'a> {
    pub fn new() -> DeepSharqSearch<'a> {
        DeepSharqSearch {
            model: Model::default(),
            config: Config::default(),
            inference: inference::Inference::new(),
            #[cfg(feature = "std")]
            elapsed_nn: 0,
            #[cfg(feature = "std")]
            elapsed_alg: 0,
        }
    }

    #[cfg(feature = "opt_schedule")]
    pub fn find_schedule(&self, k: usize, p: usize, n_c: usize) -> Option<(Vec<u16, 256>, f64)> {
        use core::iter::repeat;
        use core::cmp::max;

        if p < n_c {
            return None;
        }

        let p_e = self.model.get_channel_erasure_rate();
        let w = calculate_w_vector(k as u16, p as u16, n_c as u16, p_e);

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

    pub fn infer_k(&mut self) -> u16 {
        self.inference.infer_k(&self.model)
    }

    pub fn config_from_k(&mut self, k: u16) -> Config {
        let mut p: u16 = 0;
        if k != 0 {
            p = self.p_binary_search(k);
        }

        // If k and p are known, the maximum Nc is fixed by the delay constrant.
        // The configuration (k,p,nc) meets the D constraint
        let nc = self.get_nc(k, p);

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
            let temp_c: Config = Config::new(k, &np);

            if !self.model.check_data_rate_ri(ri)
                || !self.model.check_loss_rate(&temp_c)
                || !self.model.check_delay(&temp_c)
            {
                self.config = Config::default();
            } else {
                self.config.set_config(k, p, nc, &np);
                self.model.get_ri(&self.config);
                self.config.set_ri(ri);
            }
        }

        #[cfg(feature = "simple_schedule")]
        {
            // Get simple schedule
            let np: Vec<u16, 256> = self.simple_schedule(p, nc).unwrap();
            let temp_c: Config = Config::new(k, &np);
            let ri: f64;
            if k != 0 {
                ri = self.model.get_ri_pp(&temp_c);
            } else {
                ri = 0.0;
            }

            if !self.model.check_data_rate_ri(ri)
                || !self.model.check_loss_rate(&temp_c)
                || !self.model.check_delay(&temp_c)
            {
                self.config = Config::default();
            } else {
                self.config.set_config(k, p, nc, &np);
                self.model.get_ri(&self.config);
                self.config.set_ri(ri);
            }
        }

        #[cfg(feature = "verbose")]
        {
            println!("{:?}", self.config);
        }

        self.config.clone()
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

    pub fn check_valid_k(&self, k: u16) -> bool {
        //let p: u16 = self.p_binary_search(k);
        let mut p: u16 = 0;
        if k != 0 {
            p = self.p_binary_search(k);
        }

        // If k and p are known, the maximum Nc is fixed by the delay constrant.
        // The configuration (k,p,nc) meets the D constraint
        let nc = self.get_nc(k, p);

        // Get simple schedule
        let np: Vec<u16, 256> = self.simple_schedule(p, nc).unwrap();
        let temp_c: Config = Config::new(k, &np);
        let ri: f64;
        if k != 0 {
            ri = self.model.get_ri_pp(&temp_c);
        } else {
            ri = 0.0;
        }

        let result;
        if !self.model.check_data_rate_ri(ri)
            || !self.model.check_loss_rate(&temp_c)
            || !self.model.check_delay(&temp_c)
        {
            result = false;
        } else {
            result = true;
        }

        result
    }
}

pub fn p_binary_search(k: u16, model: &Model) -> u16 {
    let mut min = 0;
    let mut max = 255 - k;

    while max - min > 1 {
        let mid = min + (max - min) / 2;
        let temp_c: Config = Config::new(k, &[mid]);
        if model.check_loss_rate(&temp_c) {
            max = mid;
        } else {
            min = mid;
        }
    }

    max
}

pub fn simple_schedule(p: u16, nc: u16) -> Option<Vec<u16, 256>> {
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

pub fn get_nc(k: u16, p: u16, model: &Model) -> u16 {
    let nc = ((model.get_target_delay()
        - k as f64 * model.get_source_packet_interval()
        - p as f64 * model.get_average_packet_length()
            / model.get_channel_data_rate()
        - (model.get_round_trip_time() + model.get_response_delay()) / 2.0)
        / (model.get_round_trip_time()
            + model.get_response_delay()
            + model.get_packet_loss_detection_delay()))
    .get::<ratio>()
    .floor() as u16;

    nc.min(p)
}