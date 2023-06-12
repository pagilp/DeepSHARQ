pub mod deepsharq_search;
pub mod sharq_search;
pub mod fast_search;
pub mod k_range_search;
pub mod deephec_search;

use crate::config::Config;
use crate::model::Model;

pub trait SearchAlgorithm {
    fn search(&mut self) -> Config;

    fn update_model(&mut self, model: &Model);

    #[cfg(feature = "std")]
    fn name(&self) -> String;

    #[cfg(feature = "std")]
    fn generate_output(&self, elapsed: &str) -> String;

    #[cfg(feature = "std")]
    fn output_format(&self) -> String;
}
