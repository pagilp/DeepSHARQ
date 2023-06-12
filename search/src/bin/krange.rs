use full_search::algorithm::k_range_search::KRangeSearch;
use full_search::algorithm::sharq_search::SharqSearch;
use full_search::algorithm::SearchAlgorithm;
use full_search::csv_handler::CSVHanlder;

use chrono::Utc;
use std::env;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn output_file(output_dir: &str, algorithm: &dyn SearchAlgorithm) -> (File, PathBuf) {
    let output_header = algorithm.output_format();
    let time_string = Utc::now().to_rfc3339().replace(":", "_");
    let path_str = format!(
        "{}{}_out{}{}.csv",
        output_dir,
        algorithm.name(),
        "-",
        time_string
    );
    let path = Path::new(&path_str);
    let mut file = match File::create(&path) {
        Err(why) => panic!("couldn't create {}: {}", path.display(), why),
        Ok(file) => file,
    };
    if let Err(why) = file.write_all(output_header.as_bytes()) {
        panic!("couldn't write to {}: {}", path.display(), why)
    }
    (file, PathBuf::from(path))
}

fn main() {
    // parse args
    let args: std::vec::Vec<String> = env::args().collect();

    // create output path if it does not exist
    let mut output_dir = &String::from("out/");
    if args.len() == 2 {
        output_dir = &args[1];
    }
    match std::fs::create_dir_all(output_dir) {
        Err(why) => panic!("couldn't create dir: {}", why),
        Ok(()) => (),
    };

    // Prepare inputs
    let input_file = "input.csv";

    let csv_handler = CSVHanlder::from(input_file);

    // Algorithms
    let mut algorithm = KRangeSearch::new();
    let mut algorithm_opt = SharqSearch::new();

    // Prepare output file
    let (mut file, path) = output_file(output_dir, &algorithm);

    // K range delta
    let delta = 0.1;
    //let delta = f64::INFINITY;

    for model in csv_handler {
        // Obtain optimal configure
        algorithm_opt.update_model(&model);
        let mut opt_config = algorithm_opt.search();

        // Obtain k range 
        algorithm.update_model(&model);
        algorithm.set_ri_opt(opt_config.ri(&model),delta);
        let now = Instant::now();
        algorithm.search();
        let elapsed = now.elapsed();
        // generate output from experiment and write to file
        if let Err(why) = file.write_all(
            algorithm
                .generate_output(&elapsed.as_nanos().to_string())
                .as_bytes(),
        ) {
            panic!("couldn't write to {}: {}", path.display(), why)
        }
    }
}