use crate::config::Config;
use crate::model::Model;
use uom::si::f64::Time;
use uom::si::time::millisecond;

pub fn get_fec_delay(model: &Model, config: &Config) -> Time {
    let tx_time: Time = model.average_packet_length / model.channel_data_rate;

    (model.round_trip_time + model.response_delay) / 2.0
        + config.k() as f64 * model.source_packet_interval + config.np(0) as f64 * tx_time
}
pub fn get_arq_delay(model: &Model, config: &Config) -> Time {
    let tx_time: Time = model.average_packet_length / model.channel_data_rate;
    let p: u16 = config.get_np().iter().sum();

    config.nc() as f64 * (model.round_trip_time + model.response_delay
        + model.packet_loss_detection_delay) + (p - config.np(0)) as f64 * tx_time
}

pub fn get_coding_delay(_config: &Config) -> Time {
        
    #[cfg(feature = "mds_pi")]
    {
        // Encoding delay model parameters
        let m_enc: f64 = 17833.624838095253;
        let b_enc: f64 = -39212.34044444499;

        // Decoding delay model parameters
        let m_dec: f64 = 18620.07521904762;
        let b_dec: f64 = -39674.81266666663;

        // Obtain coding delays
        let k = _config.k() as f64;
        let p = _config.n() as f64 - k;
        let d_enc = p * (m_enc * k + b_enc);
        let d_dec = p * (m_dec * k + b_dec);

        Time::new::<millisecond>((d_enc + d_dec) / 1000000.0)
    }

    #[cfg(not(feature = "mds_pi"))]
    {
        Time::new::<millisecond>(0.0)
    }
}

/*
#[cfg(test)]
mod tests {
    use super::*;

    use uom::si::f64::{Information, InformationRate};
    use uom::si::information::bit;
    use uom::si::information_rate::bit_per_second;
    use uom::si::time::second;

    #[test]
    fn get_fec_delay_test() {
        let delay = Delay::default();
        let mut model = Model::default();
        let mut config = Config::default();

        model.set_round_trip_time(Time::new::<second>(53.7958257736184));
        model.set_response_delay(Time::new::<second>(51.8744044646361));
        model.set_source_packet_interval(Time::new::<second>(58.703163814134754));
        config.set_np(&vec![2, 1]);
        config.set_k(489);
        assert!(
            28876.08854785929 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test1"
        );
        model.set_round_trip_time(Time::new::<second>(9.94399630162225));
        model.set_response_delay(Time::new::<second>(18.169104834180803));
        model.set_source_packet_interval(Time::new::<second>(55.94190502560097));
        config.set_np(&vec![
            1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 4, 1, 2, 3, 2, 2, 1, 2, 3, 1, 1, 5, 4, 3, 1, 1, 2, 1, 2,
            3, 1, 3, 3, 2, 1, 2, 3, 2, 2, 1, 3, 2, 2, 2, 1, 2, 1, 1, 3, 1, 1, 2, 1, 1, 2, 1, 3, 3,
            3, 1, 3, 2, 3, 2, 2, 2, 2, 2, 2, 3, 3, 2, 3, 2, 2, 2, 4, 2, 2, 2, 2, 1, 1, 3, 1, 3, 1,
            1, 3, 1, 2, 4, 3, 2, 1, 4, 3, 1, 3, 2, 2, 2, 2, 1, 3, 1, 3, 1, 2, 1, 2, 4, 3, 2, 1, 2,
            1, 2, 3, 3, 1, 4, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 4, 1, 2, 3, 2, 1, 1, 1, 2, 2, 1, 1, 2,
            3, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 3, 3, 1, 1, 1, 3, 2, 3, 1,
        ]);
        config.set_k(160);
        assert!(
            9020.703259689659 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test2"
        );
        model.set_round_trip_time(Time::new::<second>(79.86106622579837));
        model.set_response_delay(Time::new::<second>(63.34265310840927));
        model.set_source_packet_interval(Time::new::<second>(96.06385656869132));
        config.set_np(&vec![0, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1]);
        config.set_k(307);
        assert!(
            29563.20582625534 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test3"
        );
        model.set_round_trip_time(Time::new::<second>(60.1060725980949));
        model.set_response_delay(Time::new::<second>(39.498338433308376));
        model.set_source_packet_interval(Time::new::<second>(66.38227925053974));
        config.set_np(&vec![
            0, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
            1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 1, 1, 2, 3, 1, 1, 1, 2, 1, 2, 2, 1, 1,
            2, 1, 3, 2, 1, 1, 1, 1, 1, 2, 1,
        ]);
        config.set_k(367);
        assert!(
            24412.098690463787 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test4"
        );
        model.set_round_trip_time(Time::new::<second>(18.213718355328734));
        model.set_response_delay(Time::new::<second>(93.33356590696678));
        model.set_source_packet_interval(Time::new::<second>(14.157556261408732));
        config.set_np(&vec![
            0, 3, 4, 2, 2, 1, 1, 3, 3, 2, 5, 2, 3, 2, 1, 1, 2, 2, 2, 5, 3, 4, 2, 2, 2, 2, 4, 2, 2,
            3, 1, 2, 4, 2, 3, 3, 2, 4, 4, 6, 2, 4, 2, 2, 1, 2, 6, 4, 2, 3, 3, 2, 4, 4, 4, 5, 2, 2,
            2, 1, 5, 1, 1, 2, 3, 3, 1, 2, 2, 5, 4, 2, 3, 5, 2, 4, 3, 5, 1, 1,
        ]);
        config.set_k(9);
        assert!(
            183.19164848382636 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test5"
        );
        model.set_round_trip_time(Time::new::<second>(80.71308053919269));
        model.set_response_delay(Time::new::<second>(38.0157447344874));
        model.set_source_packet_interval(Time::new::<second>(48.84178867623841));
        config.set_np(&vec![
            0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
            1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ]);
        config.set_k(296);
        assert!(
            14516.533860803409 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test6"
        );
        model.set_round_trip_time(Time::new::<second>(69.32483158659805));
        model.set_response_delay(Time::new::<second>(63.29417326955825));
        model.set_source_packet_interval(Time::new::<second>(87.15084864136223));
        config.set_np(&vec![0, 1, 1]);
        config.set_k(467);
        assert!(
            40765.75581794424 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test7"
        );
        model.set_round_trip_time(Time::new::<second>(96.42257263360659));
        model.set_response_delay(Time::new::<second>(81.98340737839261));
        model.set_source_packet_interval(Time::new::<second>(67.97506504900642));
        config.set_np(&vec![
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1,
        ]);
        config.set_k(167);
        assert!(
            11441.03885319007 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test8"
        );
        model.set_round_trip_time(Time::new::<second>(34.962792452422896));
        model.set_response_delay(Time::new::<second>(42.462413723977456));
        model.set_source_packet_interval(Time::new::<second>(75.5793693204528));
        config.set_np(&vec![
            0, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1, 1, 3, 1, 3, 1, 1, 2, 2, 2, 2, 1, 1, 2, 1, 2, 3, 1, 2,
            2, 1, 2, 2, 1,
        ]);
        config.set_k(239);
        assert!(
            18102.181870676417 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test9"
        );
        model.set_round_trip_time(Time::new::<second>(70.49634243770852));
        model.set_response_delay(Time::new::<second>(22.519055025546773));
        model.set_source_packet_interval(Time::new::<second>(92.46505746764998));
        config.set_np(&vec![
            0, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1,
        ]);
        config.set_k(442);
        assert!(
            40916.06309943292 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test10"
        );
        model.set_round_trip_time(Time::new::<second>(78.55707181168306));
        model.set_response_delay(Time::new::<second>(43.84352688893356));
        model.set_source_packet_interval(Time::new::<second>(24.86023001706862));
        config.set_np(&vec![
            2, 4, 2, 3, 1, 1, 2, 5, 5, 6, 3, 6, 1, 4, 2, 1, 1, 5, 3, 4, 2, 3, 3, 6, 4, 3, 2, 1,
        ]);
        config.set_k(340);
        assert!(
            8563.398965187776 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test11"
        );
        model.set_round_trip_time(Time::new::<second>(62.216879892391866));
        model.set_response_delay(Time::new::<second>(63.02288306281273));
        model.set_source_packet_interval(Time::new::<second>(30.92687140365148));
        config.set_np(&vec![
            0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1,
            1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 3, 1,
            1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1,
        ]);
        config.set_k(326);
        assert!(
            10144.779959067984 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test12"
        );
        model.set_round_trip_time(Time::new::<second>(46.126570297114476));
        model.set_response_delay(Time::new::<second>(81.17296377646377));
        model.set_source_packet_interval(Time::new::<second>(30.899886760464057));
        config.set_np(&vec![
            7, 5, 7, 7, 2, 4, 6, 5, 3, 5, 5, 3, 5, 3, 1, 8, 4, 3, 4, 1, 3, 7, 7, 4, 6, 3, 4, 3, 1,
        ]);
        config.set_k(167);
        assert!(
            5440.230063357535 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test13"
        );
        model.set_round_trip_time(Time::new::<second>(38.18545785222145));
        model.set_response_delay(Time::new::<second>(77.59943701683923));
        model.set_source_packet_interval(Time::new::<second>(82.11437100176411));
        config.set_np(&vec![0, 1, 1, 1, 1, 1]);
        config.set_k(341);
        assert!(
            28058.892959036093 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test14"
        );
        model.set_round_trip_time(Time::new::<second>(16.661812859393756));
        model.set_response_delay(Time::new::<second>(9.85882508194209));
        model.set_source_packet_interval(Time::new::<second>(9.499412099982452));
        config.set_np(&vec![
            0, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 3, 2, 1, 1, 1, 2, 1, 2,
            1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 4, 2, 2, 1,
            2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 3, 1, 1, 2, 1, 1,
            1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2,
            1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1,
            1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 3, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1,
        ]);
        config.set_k(181);
        assert!(
            1732.6539090674917 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test15"
        );
        model.set_round_trip_time(Time::new::<second>(30.402707587049772));
        model.set_response_delay(Time::new::<second>(58.26678082253666));
        model.set_source_packet_interval(Time::new::<second>(42.55384048366363));
        config.set_np(&vec![
            21, 14, 27, 21, 12, 16, 23, 16, 15, 21, 16, 20, 13, 15, 22, 1,
        ]);
        config.set_k(189);
        assert!(
            8980.641245774154 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test16"
        );
        model.set_round_trip_time(Time::new::<second>(8.093116824038782));
        model.set_response_delay(Time::new::<second>(55.79946668372805));
        model.set_source_packet_interval(Time::new::<second>(81.42042696723837));
        config.set_np(&vec![
            0, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 3, 1, 3, 1, 1, 1, 1, 3, 1, 1, 2, 1, 1, 3, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1,
            5, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 3, 2, 1, 1,
        ]);
        config.set_k(361);
        assert!(
            29424.720426926935 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test17"
        );
        model.set_round_trip_time(Time::new::<second>(27.9331112251873));
        model.set_response_delay(Time::new::<second>(42.2631461521865));
        model.set_source_packet_interval(Time::new::<second>(19.570405570629966));
        config.set_np(&vec![2, 2, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1]);
        config.set_k(297);
        assert!(
            5886.649394307047 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test18"
        );
        model.set_round_trip_time(Time::new::<second>(73.99086490793422));
        model.set_response_delay(Time::new::<second>(44.268569069138906));
        model.set_source_packet_interval(Time::new::<second>(89.4359358664417));
        config.set_np(&vec![
            0, 1, 2, 1, 3, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 3, 2, 2, 1, 1, 4, 2, 1, 1, 1, 1, 3, 3,
            3, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 3, 1, 1, 1,
            1, 2, 1, 1, 1,
        ]);
        config.set_k(324);
        assert!(
            29036.372937715645 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test19"
        );
        model.set_round_trip_time(Time::new::<second>(3.450549126065927));
        model.set_response_delay(Time::new::<second>(98.09506782523643));
        model.set_source_packet_interval(Time::new::<second>(88.87561264750596));
        config.set_np(&vec![
            1, 2, 1, 1, 1, 3, 3, 3, 5, 4, 2, 5, 2, 2, 4, 4, 4, 1, 5, 3, 1, 1, 5, 3, 2, 1,
        ]);
        config.set_k(43);
        assert!(
            3961.2997649659137 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test20"
        );
        model.set_round_trip_time(Time::new::<second>(23.774367870843882));
        model.set_response_delay(Time::new::<second>(96.44577067634161));
        model.set_source_packet_interval(Time::new::<second>(79.7128171212151));
        config.set_np(&vec![
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 3, 1, 1, 2, 1,
            2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 3, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 3, 1, 2, 1, 1,
            1, 1, 2, 2, 1, 1, 1, 1, 1,
        ]);
        config.set_k(19);
        assert!(
            1574.6535945766796 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test21"
        );
        model.set_round_trip_time(Time::new::<second>(77.719105745675));
        model.set_response_delay(Time::new::<second>(54.02077823779127));
        model.set_source_packet_interval(Time::new::<second>(15.655687013203645));
        config.set_np(&vec![
            4, 5, 1, 3, 7, 3, 6, 5, 5, 3, 5, 3, 7, 1, 1, 3, 4, 3, 6, 7, 2, 1, 4, 2, 3, 2, 6, 3, 1,
            5, 4, 4, 2, 3, 4, 3, 8, 5, 3, 2, 2, 4, 1, 2, 4, 1, 2, 5, 3, 3, 7, 1, 3, 8, 2, 3, 4, 2,
            5, 3, 2, 3, 1,
        ]);
        config.set_k(226);
        assert!(
            3666.6779550285714 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test22"
        );
        model.set_round_trip_time(Time::new::<second>(68.37004522718807));
        model.set_response_delay(Time::new::<second>(52.58720234959269));
        model.set_source_packet_interval(Time::new::<second>(70.32883683708855));
        config.set_np(&vec![
            0, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1,
            2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1,
        ]);
        config.set_k(421);
        assert!(
            29668.918932202665 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test23"
        );
        model.set_round_trip_time(Time::new::<second>(74.28122865233642));
        model.set_response_delay(Time::new::<second>(56.52172891579876));
        model.set_source_packet_interval(Time::new::<second>(31.95039675654341));
        config.set_np(&vec![
            0, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 2, 2, 1, 2, 1,
            1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
            2, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1,
            1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1,
        ]);
        config.set_k(87);
        assert!(
            2845.085996603344 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test24"
        );
        model.set_round_trip_time(Time::new::<second>(49.45604940378715));
        model.set_response_delay(Time::new::<second>(88.21917816062549));
        model.set_source_packet_interval(Time::new::<second>(87.46887346116029));
        config.set_np(&vec![
            1, 2, 2, 1, 1, 3, 2, 1, 2, 3, 2, 2, 1, 3, 2, 1, 1, 3, 1, 2, 1, 1, 1, 2, 1, 2, 1,
        ]);
        config.set_k(436);
        assert!(
            38292.73531630925 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test25"
        );
        model.set_round_trip_time(Time::new::<second>(52.404346500307255));
        model.set_response_delay(Time::new::<second>(84.4852817746114));
        model.set_source_packet_interval(Time::new::<second>(4.9597506044266275));
        config.set_np(&vec![
            1, 2, 2, 1, 2, 1, 3, 2, 3, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 3, 2, 1, 2,
            3, 1, 2, 2, 1, 2, 3, 3, 4, 2, 2, 2, 1, 4, 1, 1, 2, 2, 6, 2, 1, 2, 3, 2, 2, 2, 2, 2, 1,
            2, 3, 1, 4, 5, 2, 1, 1, 1, 3, 2, 2, 1, 1, 2, 1, 1, 2, 1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 1,
            1, 4, 1, 4, 3, 1, 3, 1,
        ]);
        config.set_k(9);
        assert!(
            118.04232018172561 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test26"
        );
        model.set_round_trip_time(Time::new::<second>(96.09870630906313));
        model.set_response_delay(Time::new::<second>(14.22131853146369));
        model.set_source_packet_interval(Time::new::<second>(54.054760496796995));
        config.set_np(&vec![
            2, 3, 1, 1, 2, 1, 1, 2, 3, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 3, 2, 1, 2, 3,
            1, 2, 1, 2, 3, 2, 3, 2, 1, 1, 2, 3, 5, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 1, 1,
        ]);
        config.set_k(279);
        assert!(
            15244.54771202022 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test27"
        );
        model.set_round_trip_time(Time::new::<second>(9.178858177399551));
        model.set_response_delay(Time::new::<second>(41.89597649498841));
        model.set_source_packet_interval(Time::new::<second>(43.48349046384722));
        config.set_np(&vec![
            0, 1, 1, 2, 1, 3, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 2, 2, 3, 1, 1,
            1, 1, 1, 2, 1, 4, 3, 1, 1, 1, 2, 1, 1, 3, 3, 1, 1, 2, 1, 1, 2, 1, 2, 3, 1, 1, 1, 1, 1,
            2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 5, 1, 1, 1, 2, 2, 2, 1, 2, 1, 5, 1, 3, 1,
            2, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 4, 1, 1, 2, 1, 1, 1, 1, 1,
            2, 2, 2, 1, 3, 1, 1, 1,
        ]);
        config.set_k(104);
        assert!(
            4547.820425576305 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test28"
        );
        model.set_round_trip_time(Time::new::<second>(55.646211586960135));
        model.set_response_delay(Time::new::<second>(3.6277235866915003));
        model.set_source_packet_interval(Time::new::<second>(42.58710013866002));
        config.set_np(&vec![0, 1]);
        config.set_k(492);
        assert!(
            20982.490235807556 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test29"
        );
        model.set_round_trip_time(Time::new::<second>(60.112734675515554));
        model.set_response_delay(Time::new::<second>(57.722360813700426));
        model.set_source_packet_interval(Time::new::<second>(82.24404691264854));
        config.set_np(&vec![
            2, 1, 1, 1, 1, 2, 2, 1, 2, 2, 4, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 3, 2, 1, 2, 1, 2, 3,
            3, 1, 3, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1,
        ]);
        config.set_k(384);
        assert!(
            31805.119656026945 - (delay.get_fec_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test30"
        );
    }
    #[test]
    fn get_arq_delay_test() {
        let delay = Delay::default();
        let mut model = Model::default();
        let mut config = Config::default();

        // fuzzy
        model.set_round_trip_time(Time::new::<second>(97.79957150909365));
        model.set_average_packet_length(Information::new::<bit>(48.92323718445346));
        model.set_channel_data_rate(InformationRate::new::<bit_per_second>(11.434105926738924));
        model.set_response_delay(Time::new::<second>(82.55827439409133));
        model.set_packet_loss_detection_delay(Time::new::<second>(2.834306147209975));
        config.set_np(&vec![
            1, 1, 1, 1, 2, 3, 4, 2, 4, 2, 5, 4, 2, 2, 4, 4, 1, 1, 1, 5, 2, 4, 2, 2, 1, 2, 2, 1, 3,
            2, 2, 1, 1, 3, 2, 2, 1, 2, 1, 3, 1, 1, 1, 1, 2, 2, 3, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1,
        ]);
        assert!(
            10759.369726181214 - (delay.get_arq_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test1"
        );
        model.set_round_trip_time(Time::new::<second>(70.42815324336755));
        model.set_average_packet_length(Information::new::<bit>(54.042787288403424));
        model.set_channel_data_rate(InformationRate::new::<bit_per_second>(95.97487193277459));
        model.set_response_delay(Time::new::<second>(90.02860394735781));
        model.set_packet_loss_detection_delay(Time::new::<second>(80.2278699489226));
        config.set_np(&vec![1, 3, 1]);
        assert!(
            483.62162664302286 - (delay.get_arq_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test2"
        );
        model.set_round_trip_time(Time::new::<second>(45.20670558025578));
        model.set_average_packet_length(Information::new::<bit>(72.61543082857006));
        model.set_channel_data_rate(InformationRate::new::<bit_per_second>(93.53519766463506));
        model.set_response_delay(Time::new::<second>(86.36839450176272));
        model.set_packet_loss_detection_delay(Time::new::<second>(2.7033998460021413));
        config.set_np(&vec![4, 3, 4, 2, 3, 4, 1]);
        assert!(
            818.8688369053771 - (delay.get_arq_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test3"
        );
        model.set_round_trip_time(Time::new::<second>(31.78201552048019));
        model.set_average_packet_length(Information::new::<bit>(97.7736258531689));
        model.set_channel_data_rate(InformationRate::new::<bit_per_second>(96.44170974372939));
        model.set_response_delay(Time::new::<second>(29.06921529907869));
        model.set_packet_loss_detection_delay(Time::new::<second>(23.97609501626581));
        config.set_np(&vec![
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1,
        ]);
        assert!(
            3176.1220474475467 - (delay.get_arq_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test4"
        );
        model.set_round_trip_time(Time::new::<second>(90.9052505630394));
        model.set_average_packet_length(Information::new::<bit>(10.297769098605404));
        model.set_channel_data_rate(InformationRate::new::<bit_per_second>(96.91874787791728));
        model.set_response_delay(Time::new::<second>(69.10846050413153));
        model.set_packet_loss_detection_delay(Time::new::<second>(97.25099158764924));
        config.set_np(&vec![0, 1, 1, 1, 2, 1, 1]);
        assert!(
            1544.3319769170535 - (delay.get_arq_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test5"
        );
        model.set_round_trip_time(Time::new::<second>(69.72327357513227));
        model.set_average_packet_length(Information::new::<bit>(44.5662252292897));
        model.set_channel_data_rate(InformationRate::new::<bit_per_second>(53.330217970801165));
        model.set_response_delay(Time::new::<second>(12.874064827045206));
        model.set_packet_loss_detection_delay(Time::new::<second>(48.66306881826718));
        config.set_np(&vec![6, 8, 6, 6, 1]);
        assert!(
            542.5906051549957 - (delay.get_arq_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test6"
        );
        model.set_round_trip_time(Time::new::<second>(72.62149016567633));
        model.set_average_packet_length(Information::new::<bit>(29.970027610610327));
        model.set_channel_data_rate(InformationRate::new::<bit_per_second>(20.2285652948365));
        model.set_response_delay(Time::new::<second>(51.46547805194584));
        model.set_packet_loss_detection_delay(Time::new::<second>(16.178902623115743));
        config.set_np(&vec![
            1, 3, 4, 2, 3, 2, 4, 2, 2, 3, 1, 2, 1, 1, 3, 2, 1, 1, 5, 3, 2, 4, 3, 3, 4, 6, 3, 4, 1,
            4, 1, 3, 6, 2, 2, 3, 3, 1, 3, 4, 4, 4, 2, 1, 2, 1, 1, 3, 2, 4, 1, 6, 2, 2, 2, 1, 2, 3,
            2, 4, 1, 1, 2, 5, 3, 4, 2, 2, 3, 2, 4, 3, 1, 5, 4, 4, 5, 1, 3, 4, 1, 2, 7, 4, 2, 1, 8,
            2, 4, 1, 3, 4, 3, 2, 2, 7, 3, 2, 3, 5, 3, 2, 2, 2, 1, 1, 2, 4, 1, 1, 3, 2, 1,
        ]);
        assert!(
            16163.137835062871 - (delay.get_arq_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test7"
        );
        model.set_round_trip_time(Time::new::<second>(31.220234231200838));
        model.set_average_packet_length(Information::new::<bit>(33.00049079571142));
        model.set_channel_data_rate(InformationRate::new::<bit_per_second>(61.4166019799272));
        model.set_response_delay(Time::new::<second>(88.88833433936625));
        model.set_packet_loss_detection_delay(Time::new::<second>(85.64111370276584));
        config.set_np(&vec![
            0, 1, 1, 1, 2, 2, 1, 3, 1, 3, 2, 1, 1, 1, 2, 1, 1, 1, 5, 1, 1, 2, 1,
        ]);
        assert!(
            4545.29927966379 - (delay.get_arq_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test8"
        );
        model.set_round_trip_time(Time::new::<second>(23.435180550415502));
        model.set_average_packet_length(Information::new::<bit>(27.30334801680754));
        model.set_channel_data_rate(InformationRate::new::<bit_per_second>(77.71799596053084));
        model.set_response_delay(Time::new::<second>(57.46507230522306));
        model.set_packet_loss_detection_delay(Time::new::<second>(19.811923423169596));
        config.set_np(&vec![
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1,
            1,
        ]);
        assert!(
            2931.1925042864436 - (delay.get_arq_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test9"
        );
        model.set_round_trip_time(Time::new::<second>(33.62720905223877));
        model.set_average_packet_length(Information::new::<bit>(21.161946780663232));
        model.set_channel_data_rate(InformationRate::new::<bit_per_second>(91.45735127269116));
        model.set_response_delay(Time::new::<second>(67.75833321666764));
        model.set_packet_loss_detection_delay(Time::new::<second>(97.29047421815376));
        config.set_np(&vec![0, 2, 1, 2, 2, 2, 3, 2, 2, 2, 1, 1, 2, 1]);
        assert!(
            2588.110091350766 - (delay.get_arq_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test10"
        );
        model.set_round_trip_time(Time::new::<second>(34.44025180493628));
        model.set_average_packet_length(Information::new::<bit>(17.922697322567714));
        model.set_channel_data_rate(InformationRate::new::<bit_per_second>(38.888945827917155));
        model.set_response_delay(Time::new::<second>(9.252149237595164));
        model.set_packet_loss_detection_delay(Time::new::<second>(53.83642139732308));
        config.set_np(&vec![
            0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1,
            1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1,
            2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1,
            1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ]);
        assert!(
            18427.13148699648 - (delay.get_arq_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test11"
        );
        model.set_round_trip_time(Time::new::<second>(77.54844415782208));
        model.set_average_packet_length(Information::new::<bit>(24.82001635700589));
        model.set_channel_data_rate(InformationRate::new::<bit_per_second>(34.52680241034864));
        model.set_response_delay(Time::new::<second>(58.09958159600775));
        model.set_packet_loss_detection_delay(Time::new::<second>(65.20336649019424));
        config.set_np(&vec![
            1, 2, 1, 1, 2, 2, 1, 1, 2, 4, 1, 1, 2, 3, 1, 1, 2, 4, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1,
            1, 2, 1, 1, 1, 3, 2, 1, 2, 2, 1, 1, 2, 1, 1, 4, 2, 2, 3, 1, 2, 1, 2, 2, 1, 2, 1, 1, 1,
            3, 1, 2, 1, 1, 1, 1, 3, 2, 2, 1,
        ]);
        assert!(
            13738.407249430222 - (delay.get_arq_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test12"
        );
        model.set_round_trip_time(Time::new::<second>(7.9456620233185005));
        model.set_average_packet_length(Information::new::<bit>(6.4861560983335735));
        model.set_channel_data_rate(InformationRate::new::<bit_per_second>(9.046071301260362));
        model.set_response_delay(Time::new::<second>(64.10468032184748));
        model.set_packet_loss_detection_delay(Time::new::<second>(30.849850066543883));
        config.set_np(&vec![6, 8, 9, 8, 9, 5, 7, 3, 8, 10, 5, 5, 5, 6, 12, 1]);
        assert!(
            1615.9212592035522 - (delay.get_arq_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test13"
        );
        model.set_round_trip_time(Time::new::<second>(30.141050878161614));
        model.set_average_packet_length(Information::new::<bit>(85.85080319506919));
        model.set_channel_data_rate(InformationRate::new::<bit_per_second>(64.23732084617643));
        model.set_response_delay(Time::new::<second>(37.12310247957005));
        model.set_packet_loss_detection_delay(Time::new::<second>(92.28099833137287));
        config.set_np(&vec![
            0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 4, 1, 2, 2, 2, 1, 1, 1, 1, 2, 3, 1, 2, 1, 2, 1, 1, 2, 2,
            1, 3, 1, 2, 1, 2, 1, 1, 2, 1, 1, 4, 2, 1, 1, 1, 1, 3, 5, 2, 1, 1, 1, 4, 1, 1, 1, 1, 1,
            4, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 3, 1,
            1,
        ]);
        assert!(
            14062.187166360058 - (delay.get_arq_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test14"
        );
        model.set_round_trip_time(Time::new::<second>(73.75053829906444));
        model.set_average_packet_length(Information::new::<bit>(58.94980574737973));
        model.set_channel_data_rate(InformationRate::new::<bit_per_second>(28.400107245507623));
        model.set_response_delay(Time::new::<second>(18.874569930158835));
        model.set_packet_loss_detection_delay(Time::new::<second>(73.28794477096127));
        config.set_np(&vec![
            6, 5, 7, 7, 4, 6, 12, 7, 4, 8, 7, 3, 2, 5, 9, 3, 7, 11, 9, 4, 6, 4, 1,
        ]);
        assert!(
            3922.0024966804353 - (delay.get_arq_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test15"
        );
        model.set_round_trip_time(Time::new::<second>(89.27815171844883));
        model.set_average_packet_length(Information::new::<bit>(9.218043805971465));
        model.set_channel_data_rate(InformationRate::new::<bit_per_second>(15.14384140063676));
        model.set_response_delay(Time::new::<second>(12.441688803205519));
        model.set_packet_loss_detection_delay(Time::new::<second>(89.79183428817346));
        config.set_np(&vec![
            1, 1, 2, 3, 2, 2, 2, 1, 2, 1, 1, 3, 2, 2, 1, 1, 2, 1, 1, 2, 4, 2, 1, 2, 1, 1, 1, 2, 1,
            3, 4, 1, 2, 2, 3, 1, 4, 3, 1, 2, 1, 3, 2, 4, 3, 1, 1, 2, 1, 2, 4, 1, 2, 1, 1, 2, 4, 3,
            2, 2, 1,
        ]);
        assert!(
            11561.309593188962 - (delay.get_arq_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test16"
        );
        model.set_round_trip_time(Time::new::<second>(68.37769198576277));
        model.set_average_packet_length(Information::new::<bit>(64.20327042814972));
        model.set_channel_data_rate(InformationRate::new::<bit_per_second>(87.90297375517771));
        model.set_response_delay(Time::new::<second>(50.71852801982503));
        model.set_packet_loss_detection_delay(Time::new::<second>(99.0138260415687));
        config.set_np(&vec![
            1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1,
            1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1,
            1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1,
        ]);
        assert!(
            31310.98097996432 - (delay.get_arq_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test17"
        );
        model.set_round_trip_time(Time::new::<second>(80.4495358764779));
        model.set_average_packet_length(Information::new::<bit>(64.75318181305163));
        model.set_channel_data_rate(InformationRate::new::<bit_per_second>(89.48515542170689));
        model.set_response_delay(Time::new::<second>(52.511342841281994));
        model.set_packet_loss_detection_delay(Time::new::<second>(99.75246772305016));
        config.set_np(&vec![16, 15, 16, 16, 7, 15, 17, 12, 16, 11, 17, 1]);
        assert!(
            2663.324364814472 - (delay.get_arq_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test18"
        );
        model.set_round_trip_time(Time::new::<second>(45.040611557028875));
        model.set_average_packet_length(Information::new::<bit>(19.728860780817705));
        model.set_channel_data_rate(InformationRate::new::<bit_per_second>(41.57256639561137));
        model.set_response_delay(Time::new::<second>(85.69821198436674));
        model.set_packet_loss_detection_delay(Time::new::<second>(91.93490844486855));
        config.set_np(&vec![
            1, 3, 3, 3, 3, 2, 2, 2, 2, 1, 4, 4, 3, 2, 1, 2, 4, 3, 1, 2, 4, 2, 1, 3, 2, 2, 5, 3, 1,
            3, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 3, 1, 3, 3, 1, 2, 4, 3, 2, 3, 2, 2,
            2, 1, 1, 3, 4, 4, 1, 2, 2, 3, 2, 2, 2, 3, 2, 4, 2, 1, 3, 4, 2, 2, 1, 1, 3, 2, 4, 3, 2,
            2, 1, 1, 2, 3, 3, 2, 3, 3, 5, 1, 6, 2, 3, 1, 2, 1, 2, 2, 1, 1, 2, 3, 2, 4, 3, 2, 2, 3,
            2, 1, 3, 1, 2, 2, 5, 3, 1, 4, 3, 1, 1, 4, 4, 3, 2, 1, 2, 3, 2, 1, 1, 1, 3, 1, 2, 2, 3,
            1, 3, 1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 4, 3, 2, 2, 1, 1, 1, 6, 2, 2, 4, 2, 1, 2, 1, 3, 2,
            2, 3, 3, 1, 3, 2, 3, 3, 1,
        ]);
        assert!(
            40717.8686819979 - (delay.get_arq_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test19"
        );
        model.set_round_trip_time(Time::new::<second>(67.91608160546583));
        model.set_average_packet_length(Information::new::<bit>(73.28351921625826));
        model.set_channel_data_rate(InformationRate::new::<bit_per_second>(22.22654773468664));
        model.set_response_delay(Time::new::<second>(81.82927099823972));
        model.set_packet_loss_detection_delay(Time::new::<second>(65.1515987816407));
        config.set_np(&vec![
            2, 5, 3, 1, 3, 3, 2, 7, 2, 5, 4, 4, 2, 3, 3, 3, 3, 3, 3, 4, 4, 6, 6, 1,
        ]);
        assert!(
            5206.39920885347 - (delay.get_arq_delay(&model, &config).get::<second>() as f64)
                < 0.00000000000001,
            "Test20"
        );
    }
}

*/