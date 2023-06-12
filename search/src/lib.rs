#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(all(feature = "std", feature = "no_std"))]
compile_error!("feature \"std\" and \"no_std\" cannot be enabled at the same time");

#[cfg(not(any(feature = "std", feature = "no_std")))]
compile_error!("either feature \"std\" or \"no_std\" must be enabled (currently none is enabled)");

#[cfg(all(feature = "simple_schedule", feature = "opt_schedule"))]
compile_error!(
    "feature \"simple_schedule\" and \"opt_schedule\" cannot be enabled at the same time"
);

#[cfg(not(any(feature = "simple_schedule", feature = "opt_schedule")))]
compile_error!("either feature \"simple_schedule\" or \"opt_schedule\" must be enabled (currently none is enabled)");

#[cfg(feature = "mem")]
#[macro_use]
extern crate lazy_static;

pub mod algorithm;
pub mod config;
#[cfg(feature = "std")]
pub mod csv_handler;
pub mod math;
pub mod model;