use crate::config::Config;
use crate::model::Model;
use crate::math::*;
use core::cmp::max;
use heapless::Vec;


pub fn get_ri(model: &Model, config: &Config) -> (f64, Vec<f64, 255>) {
    let prob_c_vec: Vec<f64, 255> = prob_failure_cycle(model, config);

    let mut ri: f64 = 0.0;
    for c in 0..=config.nc() as usize {
        ri += prob_c_vec[c] * config.np(c) as f64;
    }
    ri /= config.k() as f64;

    (ri, prob_c_vec)
}

pub fn update_ri(model: &Model, config: &Config, prob_c_vec: &mut Vec<f64, 255>, i: usize, j: usize) -> f64 {
    update_prob_failure_cycle(model, config, prob_c_vec, i, j);

    let mut ri: f64 = 0.0;
    for c in 0..=config.nc() as usize {
        ri += prob_c_vec[c] * config.np(c) as f64;
    }
    ri /= config.k() as f64;

    ri
}


///
/// Updates the failure probability vectors only in those positions that are needed.
///
fn update_prob_failure_cycle(model: &Model, config: &Config,prob_c_vec: &mut Vec<f64, 255>, i: usize, j: usize) {
    let p = config.n() - config.k();

    for c in i..=j as usize {
        if c == 0 {
            continue;
        }

        let p_f: f64 = match config.k() < p {
            true => p_f(model, config, c),
            false => p_f_inv(model, config, c),
        };

        prob_c_vec[c] = p_f;
    }
}

///
/// Obtains a vector of length Nc+1 with the probability of each cycle to fail
///
fn prob_failure_cycle(model: &Model, config: &Config) -> Vec<f64, 255> {
    let mut prob_vec: Vec<f64,255> = Vec::new();
    prob_vec.push(1.0).unwrap();

    let p: u16 = config.n() - config.k();
    if config.k() < p {
        for c in 1..=config.nc() as usize {
            let p_f: f64 = p_f(model, config, c);
            prob_vec.push(p_f).unwrap();
        }
    } else {
        for c in 1..=config.nc() as usize {
            let p_f: f64 = p_f_inv(model, config, c);
            prob_vec.push(p_f).unwrap();
        }
    }

    prob_vec
}

///
/// Obtains the probability of a cycle to fail
///
/// This function iterates  between max(0,n[c-1]-p) and k-1
/// to obtain the probability of the cycle to fail
///
fn p_f(model: &Model, config: &Config, c: usize) -> f64 {
    let p = config.n() - config.k();

    let p_c: u16 = config.get_np()[0..c].iter().sum();
    let n_c: u16 = config.k() + p_c;

    let min: u16 = max(0,n_c as i16 - p as i16) as u16;
    let max: u16 = config.k() - 1;

    let mut p_f: f64 = 0.0;

    for i in min..=max {
        p_f += p_m_ri(i, n_c, model.channel_erasure_rate);
    }
    p_f
}

///
/// Obtains the probability of a cycle to fail
///
/// The only difference with fun p_f() is that it obtains p_f = 1 - prob.
/// This function is though for cases in which k > p and hence therefore
/// the prob is faster calculated from the inverse.
///
fn p_f_inv(model: &Model, config: &Config, c: usize) -> f64 {
    let p = config.n() - config.k();

    let p_c: u16 = config.get_np()[0..c].iter().sum();
    let n_c: u16 = config.k() + p_c;

    let mut p_f: f64 = 0.0;
    for i in config.k()..=n_c {
        p_f += p_m_ri(i, n_c, model.channel_erasure_rate);
    }
    for i in 0..(max(0,n_c-p)) {
        p_f += p_m_ri(i, n_c, model.channel_erasure_rate);
    }
    1.0 - p_f
}


///
/// Obtains the RI in O(p + log(k))
///
/// Given k and p, it initializes the pp table only with those coefficients
/// that will be used, and obtains the RI for a given Np
///
pub fn get_ri_pp(model: &Model, config: &Config) -> f64 {
    // Get configuration parameters
    let k: u16 = config.k();
    let p: u16 = config.n() - k;
    let nc: u16 = config.nc();
    let np = config.get_np();

    if nc == 0 {
        return p as f64 / k as f64;
    }

    // Get channele rasure probability
    let p_e = model.get_channel_erasure_rate();
    let w = calculate_w_vector(k as u16, p as u16, p_e);

    let mut res: f64 = np[0].into();
    for c in 1..=nc as usize {
        let p_c: u16 = np[0..=c-1].iter().sum();
        res += w[p_c as usize] * np[c] as f64;
    }
    res / k as f64
}
