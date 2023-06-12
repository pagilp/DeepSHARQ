use crate::model::Model;
use heapless::Vec;

#[derive(Debug, Clone)]
pub struct Config {
    k: u16,
    np: Vec<u16, 255>,
    cum_np: Vec<u16,255>,
    n: u16,
    nc: u16,
    ri: f64,
}

// efficient datastructure for storing a configuration
// lazy ri get
// O(1) get of n,k,nc
// setters include: k, set_np,
//  and other np manipulations: shift_np, inc_np, dec_np
// can produce clones of Vec<i32> via Config::clone(vec)

impl Config {
    //==========================================================
    //======================= GETTER ===========================
    //==========================================================

    // get k
    pub fn k(&self) -> u16 {
        self.k
    }
    // get n
    pub fn n(&self) -> u16 {
        self.n
    }
    // get ri, lazy
    pub fn ri(&mut self, model: &Model) -> f64 {
        if self.ri < 0.0 {
            (self.ri,_) = model.get_ri(self);
        }
        self.ri
    }
    // get nc == nc
    pub fn nc(&self) -> u16 {
        self.nc
    }
    // accesses element in np Array
    pub fn np(&self, i: usize) -> u16 {
        assert!(
            i < self.np.len(),
            "FATAL ERROR: index out of range \n want to access index {} in np {:?}",
            i,
            self.np
        );
        self.np[i as usize]
    }
    // get copy of current repair schedule
    #[allow(dead_code)]
    pub fn get_np(&self) -> Vec<u16, 255> {
        self.np.clone()
    }

    pub fn one_parity_forward(&mut self, i: usize, j: usize) {
        // Move one parity packet from position i to position j
        self.np[i] -= 1;
        self.np[j] += 1;

        // Update the comulative schedule as well
        self.cum_np[j] += 1;
    }

    pub fn cum_np(&self, i: usize) -> u16 {
        assert!(
            i < self.cum_np.len(),
            "FATAL ERROR: index out of range \n want to access index {} in np {:?}",
            i,
            self.cum_np
        );
        self.cum_np[i]
    }

    //==========================================================
    //======================= SETTER ===========================
    //==========================================================

    // set k
    #[allow(dead_code)]
    pub fn set_k(&mut self, k: u16) {
        self.k = k;
        self.ri = -1.0; // reset ri
    }
    // set np
    #[allow(dead_code)]
    pub fn set_np(&mut self, np: &[u16]) {
        let mut cum_np: Vec<u16, 255> = Vec::new();
        cum_np.push(self.k+np[0]).unwrap();
        for c in 1..=np.len()-1 as usize {
            cum_np.push(cum_np[c-1] + np[c]).unwrap();
        }

        self.np = Vec::from_slice(np).unwrap();
        // update datastructure
        self.n = self.np.iter().sum::<u16>() + self.k;
        self.nc = self.np.len() as u16 - 1;
        self.cum_np = cum_np;
        self.ri = -1.0; // reset ri
    }

    // set full configuraiton
    #[allow(dead_code)]
    pub fn set_config(&mut self, k: u16, p: u16, nc: u16, np: &[u16]) {
        self.k = k;
        self.n = k+p;
        self.nc = nc;
        self.np = Vec::from_slice(np).unwrap();
    }

    pub fn set_ri(&mut self, ri: f64) {
        self.ri = ri;
    }

    //==========================================================
    //======================== MISC ============================
    //==========================================================
    pub fn new(k: u16, np: &[u16]) -> Self {
        let mut cum_np: Vec<u16, 255> = Vec::new();
        cum_np.push(k+np[0]).unwrap();
        for c in 1..=np.len()-1 as usize {
            cum_np.push(cum_np[c-1] + np[c]).unwrap();
        }

        Self {
            k,
            n: k + np.iter().sum::<u16>(),
            nc: (np.len() - 1) as u16,
            np: Vec::from_slice(np).unwrap(),
            cum_np:  cum_np,
            ri: -1.0,
        }
    }

    #[cfg(feature = "std")]
    pub fn get_config_str(&self) -> String {
        let np_str: String = format!("{:?}", self.np);
        let np_str: String = str::replace(&np_str, ", ", ";");
        let np_str: String = str::replace(&np_str, "[", "<");
        let np_str: String = str::replace(&np_str, "]", ">");
        format!(
            "{},{},{},{},{},",
            self.k, self.n, self.nc, np_str, self.ri
        )
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            k: 0,
            n: 0,
            nc: 0,
            np: Vec::from_slice(&[0]).unwrap(),
            cum_np: Vec::from_slice(&[0]).unwrap(),
            ri: f64::INFINITY, // TODO maybe ri via setters that implicitly set it?
        }
    }
}
