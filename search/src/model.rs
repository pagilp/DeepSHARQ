pub mod data_rate;
pub mod delay;
pub mod loss;
pub mod redundancy;

use crate::config::Config;
use crate::model::{data_rate::*, redundancy::*, loss::*, delay::*};

use uom::si::f64::{Information, InformationRate, Time};
use uom::si::information::byte;
use uom::si::information_rate::bit_per_second;
use uom::si::time::millisecond;

use heapless::Vec;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Model {
    target_erasure_rate: f64,
    target_delay: Time,
    channel_data_rate: InformationRate,

    channel_erasure_rate: f64,
    source_packet_interval: Time,
    average_packet_length: Information,

    round_trip_time: Time,
    response_delay: Time,
    packet_loss_detection_delay: Time,
}

impl Model {
    // Delay
    pub fn get_arq_delay(&self, config: &Config) -> Time {
        get_arq_delay(self, config)
    }
    pub fn get_fec_delay(&self, config: &Config) -> Time {
        get_fec_delay(self, config)
    }
    pub fn get_delay(&self, config: &Config) -> Time {
        get_arq_delay(self, config) + self.get_fec_delay(config)
    }
    pub fn check_delay(&self, config: &Config) -> bool {
        #[cfg(feature = "log")]
        {
            println!("D={:?} D_T={:?}", self.get_delay(config), self.target_delay);
        }
        self.get_delay(config) <= self.target_delay
    }

    // Loss
    pub fn get_loss_rate(&self, config: &Config) -> f64 {
        get_loss_rate(self, config)
    }
    pub fn check_loss_rate(&self, config: &Config) -> bool {
        // The implementation below avoids wrong configurations due to approximation errors
        #[cfg(feature = "log")]
        {
            println!("PLR={} PLR_T={}", self.get_loss_rate(config), self.target_erasure_rate);
        }
        (self.get_loss_rate(config) - self.target_erasure_rate) <= 0.00000000001 
    }

    // DataRate
    pub fn get_ri(&self, config: &Config) -> (f64, Vec<f64,255>) {
        get_ri(self, config)
    }
    pub fn get_ri_pp(&self, config: &Config) -> f64 {
        get_ri_pp(&self, config)
    }
    pub fn update_ri(&self, config: &Config, prob_c_vec: &mut Vec<f64, 255>, i: usize, j: usize) -> f64 {
        update_ri(self, config, prob_c_vec, i, j)
    }
    pub fn get_effective_rate(&self, config: &mut Config) -> InformationRate {
        get_effective_rate(self, config)
    }

    pub fn get_data_rate(&self) -> InformationRate {
        get_data_rate(self)
    }

    pub fn check_data_rate(&self, mut config: &mut Config) -> bool {
        #[cfg(feature = "log")]
        {
            println!("R={:?} R_C={:?}", 
                    self.get_effective_rate(&mut config).get::<bit_per_second>(), 
                    self.channel_data_rate.get::<bit_per_second>());
        }
        self.get_effective_rate(&mut config) <= self.channel_data_rate
    }

    pub fn check_data_rate_ri(&self, ri: f64) -> bool {
        get_effective_rate_ri(self, ri) <= self.channel_data_rate
    }

    // misc
    pub fn is_valid(&self, mut config: &mut Config) -> bool {
        self.check_delay(config)
            && self.check_data_rate(&mut config)
            && self.check_loss_rate(config)
    }

    #[cfg(feature = "std")]
    pub fn print_model(&self) {
        println!();
        println!("PLR_T={} (rate), D_T={} (ms), R_C={:?} (bps), p_e={} (rate), T_s={:?} (ms), P_L={} (B), RTT={} (ms), D_RS={} (ms), D_PL={} (ms)",
                self.target_erasure_rate, self.target_delay.get::<millisecond>(),
                self.channel_data_rate.get::<bit_per_second>(),self.channel_erasure_rate,
                self.source_packet_interval.get::<millisecond>(),
                self.average_packet_length.get::<byte>(),
                self.round_trip_time.get::<millisecond>(),
                self.response_delay.get::<millisecond>(),
                self.packet_loss_detection_delay.get::<millisecond>());
        println!();
    }

    #[cfg(feature = "std")]
    pub fn get_model_str(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{},",
            self.target_erasure_rate,
            self.target_delay.get::<millisecond>(),
            self.channel_data_rate.get::<bit_per_second>(),
            self.channel_erasure_rate,
            self.source_packet_interval.get::<millisecond>(),
            self.average_packet_length.get::<byte>(),
            self.round_trip_time.get::<millisecond>(),
            self.response_delay.get::<millisecond>(),
            self.packet_loss_detection_delay.get::<millisecond>()
        )
    }
}

impl Model {
    #[allow(clippy::too_many_arguments)]
    pub fn from(
        target_erasure_rate: f64,
        target_delay: Time,
        channel_data_rate: InformationRate,
        channel_erasure_rate: f64,
        source_packet_interval: Time,
        average_packet_length: Information,
        round_trip_time: Time,
        response_delay: Time,
        packet_loss_detection_delay: Time,
    ) -> Self {
        Self {
            target_erasure_rate,
            target_delay,
            channel_data_rate,
            channel_erasure_rate,
            source_packet_interval,
            average_packet_length,
            round_trip_time,
            response_delay,
            packet_loss_detection_delay,
        }
    }

    // used mainly for testing
    #[allow(dead_code)]
    pub fn set_target_erasure_rate(&mut self, target_erasure_rate: f64) {
        self.target_erasure_rate = target_erasure_rate;
    }
    #[allow(dead_code)]
    pub fn set_target_delay(&mut self, target_delay: Time) {
        self.target_delay = target_delay;
    }
    #[allow(dead_code)]
    pub fn set_channel_data_rate(&mut self, channel_data_rate: InformationRate) {
        self.channel_data_rate = channel_data_rate;
    }
    #[allow(dead_code)]
    pub fn set_channel_erasure_rate(&mut self, channel_erasure_rate: f64) {
        self.channel_erasure_rate = channel_erasure_rate;
    }
    #[allow(dead_code)]
    pub fn set_source_packet_interval(&mut self, source_packet_interval: Time) {
        self.source_packet_interval = source_packet_interval;
    }
    #[allow(dead_code)]
    pub fn set_average_packet_length(&mut self, average_packet_length: Information) {
        self.average_packet_length = average_packet_length;
    }
    #[allow(dead_code)]
    pub fn set_round_trip_time(&mut self, round_trip_time: Time) {
        self.round_trip_time = round_trip_time;
    }
    #[allow(dead_code)]
    pub fn set_response_delay(&mut self, response_delay: Time) {
        self.response_delay = response_delay;
    }
    #[allow(dead_code)]
    pub fn set_packet_loss_detection_delay(&mut self, packet_loss_detection_delay: Time) {
        self.packet_loss_detection_delay = packet_loss_detection_delay;
    }

    // Getters
    #[allow(dead_code)]
    pub fn get_target_erasure_rate(&self) -> f64 {
        self.target_erasure_rate
    }
    #[allow(dead_code)]
    pub fn get_target_delay(&self) -> Time {
        self.target_delay
    }
    #[allow(dead_code)]
    pub fn get_channel_data_rate(&self) -> InformationRate {
        self.channel_data_rate
    }
    #[allow(dead_code)]
    pub fn get_channel_erasure_rate(&self) -> f64 {
        self.channel_erasure_rate
    }
    #[allow(dead_code)]
    pub fn get_source_packet_interval(&self) -> Time {
        self.source_packet_interval
    }
    #[allow(dead_code)]
    pub fn get_average_packet_length(&self) -> Information {
        self.average_packet_length
    }
    #[allow(dead_code)]
    pub fn get_round_trip_time(&self) -> Time {
        self.round_trip_time
    }
    #[allow(dead_code)]
    pub fn get_response_delay(&self) -> Time {
        self.response_delay
    }
    #[allow(dead_code)]
    pub fn get_packet_loss_detection_delay(&self) -> Time {
        self.packet_loss_detection_delay
    }
}

impl Default for Model {
    fn default() -> Self {
        Self {
            target_erasure_rate: 0.0,
            target_delay: Time::new::<millisecond>(0.0),
            channel_data_rate: InformationRate::new::<bit_per_second>(0.0),

            channel_erasure_rate: 0.0,
            source_packet_interval: Time::new::<millisecond>(0.0),
            average_packet_length: Information::new::<byte>(0.0),

            round_trip_time: Time::new::<millisecond>(0.0),
            response_delay: Time::new::<millisecond>(0.0),
            packet_loss_detection_delay: Time::new::<millisecond>(0.0),
        }
    }
}
