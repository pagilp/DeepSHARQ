use core::cmp::max;
use heapless::Vec;

pub static BINOM: [[f64; 257]; 257] = {
    let mut array = [[0.0; 257]; 257];
    // need to use while because for is not allowed in const evaluation
    let mut n = 0;
    while n < 257 {
        array[n][0] = 1.0;
        let mut k = 1;
        while k <= n {
            array[n][k] = array[n - 1][k] + array[n - 1][k - 1];
            k += 1;
        }
        n += 1;
    }
    array
};

// p_m == iid_block_error_distribution
pub fn p_m(j: u16, n: u16, p_e: f64) -> f64 {
    BINOM[(n) as usize][(j) as usize] * p_e.powi(j.into()) * (1.0 - p_e).powi((n - j).into())
}

pub fn p_m_ri(j: u16, n: u16, p_e: f64) -> f64 {
    BINOM[(n) as usize][(j) as usize] * p_e.powi((n - j).into()) * (1.0 - p_e).powi((j).into())
}

pub fn p_m_times_h_d(n: u16, k: u16, i: u16, j: u16, p_e: f64) -> f64 {
    p_e.powi(j.into())
        * (1.0 - p_e).powi((n - j).into())
        * BINOM[(k) as usize][(i) as usize]
        * BINOM[(n - k) as usize][(j - i) as usize]
}

pub fn calculate_w_vector(k: u16, p: u16, p_e: f64) -> Vec<f64, 256> {
    let diff: u16 = max(0, k as i16 - p as i16) as u16;

    let mut w_s: Vec<f64,256> = heapless::Vec::new();

    // Initialize w_s vector with w[0]
    let mut w: f64 = 0.0;
    let mut a: f64 = (1.0 - p_e).powi(diff.into()) * p_e.powi((k-diff).into());
    for j in diff..k {
        w += a * BINOM[(k) as usize][(j) as usize];
        a *= (1.0 - p_e) / p_e;
    }
    w_s.push(w).unwrap();

    let mut a_first = if k > p {
        (1.0 - p_e).powi((k-p).into()) * p_e.powi(p.into())
    } else {
        p_e.powi((k).into())
    };
    let mut a_last = (1.0 - p_e).powi((k-1).into()) * p_e;

    if p==0 {
        return w_s;
    }

    // For p > 0
    for i in 0..(p-1) {
        let upper;

        if (k as i16 - p  as i16 + i as i16) >= 0 {
            upper = BINOM[(k+i) as usize][(k-p+i) as usize] * a_first;
            a_first *= 1.0 - p_e;
        } else {
            upper = a_first;
            a_first *= p_e;
        }
        let lower = BINOM[(k+i) as usize][(k-1) as usize] * a_last;
        a_last *= p_e;

        w = p_e * (w_s[i as usize] - upper) + (1.0 - p_e) * (w_s[i as usize] - lower);

        if (k as i16 - p  as i16 + i  as i16) < 0 {
            w += a_first;
        }

        w_s.push(w).unwrap();
    }

    w_s
}

pub fn binomial_coefficient(n: u16, mut k: u16) -> f64 {
    // credit to https://www.geeksforgeeks.org/space-and-time-efficient-binomial-coefficient/
    if k > n - k {
        k = n - k;
    }
    let mut res: f64 = 1.0;
    for i in 0..k {
        res *= (n - i) as f64;
        res /= i as f64 + 1.0;
    }
    res
}

pub fn ascending_restricted_integer_compositions(
    p: u16,
    s: u16,
    a: u16,
    b: u16,
) -> Vec<Vec<u16, 32>, 32> {
    let mut result: Vec<Vec<u16, 32>, 32> = Vec::new();
    if s == 0 {
        return result;
    }
    if s == 1 {
        // size is 1
        if a <= p && p <= b {
            result.push(Vec::from_slice(&[p]).unwrap()).unwrap();
        }
        return result;
    }
    // else size > 1 -> recursion
    // we iterare over all possible first position's values -> a to max(b,r-s+1)
    for i in a..=b.min(p - (s - 1)) {
        let others = ascending_restricted_integer_compositions(p - i, s - 1, i, b);

        // for all others pre-pend i
        let pre = core::iter::once(i as u16);
        let ext: Vec<Vec<u16, 32>, 32> = others
            .iter()
            .map(|other| {
                pre.clone()
                    .chain(other.iter().cloned())
                    .collect::<Vec<_, 32>>()
            })
            .collect::<Vec<_, 32>>();
        result.extend(ext);
    }

    result
}