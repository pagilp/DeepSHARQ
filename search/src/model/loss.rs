use crate::config::Config;
use crate::math::BINOM;
use crate::model::Model;

pub fn get_loss_rate(model: &Model, config: &Config) -> f64 {
    let p_e: f64 = model.channel_erasure_rate;
    let k: u16 = config.k();
    let p: u16 = config.n() - k;

    let mut ps_1: f64 = 0.0;
    let mut power_p_e: f64;
    if p <= k {
        power_p_e = p_e.powi(p.into()) * (1.0 - p_e).powi((k - p).into());
        
        for j in p + 1..=k {
            power_p_e *= p_e / (1.0 - p_e);
    
            ps_1 += j as f64 * BINOM[k as usize][j as usize] * power_p_e;
        }
    }

    let mut ps_p: f64 = 0.0;
    let mut ps_2: f64 = 0.0;

    for i in 0..=p.min(k) {
        ps_2 += BINOM[k as usize][i as usize] * ps_p * i as f64;
        ps_p = (p_e / (1.0 - p_e)) * (BINOM[p as usize][(p - i) as usize] + ps_p);
    }
    ps_2 *= p_e.powi(p.into()) * (1.0 - p_e).powi(k.into());

    (ps_1 + ps_2) / (k as f64)
}