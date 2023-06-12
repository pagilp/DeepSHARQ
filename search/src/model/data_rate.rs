use crate::config::Config;
use crate::model::Model;

use uom::si::f64::InformationRate;
use uom::si::information::bit;
use uom::si::information_rate::bit_per_second;
use uom::si::time::second;

pub fn get_effective_rate(model: &Model, config: &mut Config) -> InformationRate {
    let data_rate = InformationRate::new::<bit_per_second>(
        model.average_packet_length.get::<bit>() / model.source_packet_interval.get::<second>(),
    );

    (1.0 + config.ri(model)) * data_rate
}

pub fn get_effective_rate_ri(model: &Model, ri: f64) -> InformationRate {
    let data_rate = InformationRate::new::<bit_per_second>(
         model.average_packet_length.get::<bit>() / model.source_packet_interval.get::<second>(),
    );

    (1.0 + ri) * data_rate
}

pub fn get_data_rate(model: &Model) -> InformationRate {
    InformationRate::new::<bit_per_second>(model.average_packet_length.get::<bit>()
                                            / model.source_packet_interval.get::<second>(),)
}