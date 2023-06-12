use full_search::algorithm::deephec_search::DeepHecSearch;
use full_search::algorithm::SearchAlgorithm;
use full_search::csv_handler::CSVHanlder;
use full_search::model::Model;

use uom::si::time::millisecond;
use uom::si::information_rate::bit_per_second;

use tflitec::interpreter::{Interpreter, Options};
use tflitec::tensor;
use tflitec;

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

    // Create interpreter options
    let mut options = Options::default();
    options.thread_count = 1;
    // Create NN Model
    let model = tflitec::model::Model::new("models/deephec_model.tflite").unwrap();
    // Create interpreter
    let interpreter = Interpreter::new(&model, Some(options)).unwrap();
    // Resize input
    let input_shape = tensor::Shape::new(vec![1, 6]);
    interpreter.resize_input(0, input_shape).unwrap();
    // Allocate tensors if you just created Interpreter or resized its inputs
    interpreter.allocate_tensors().unwrap();

    let input_tensor = interpreter.input(0).unwrap();
    let output_tensor = interpreter.output(0).unwrap();

    // Search algorithm object
    let mut algorithm = DeepHecSearch::new();

    // Prepare output file
    let (mut file, path) = output_file(output_dir, &algorithm);

    for model in csv_handler {
        algorithm.update_model(&model);

        // Run NN inference
        let now = Instant::now();
        let input_vec: Vec<f32> = model_to_vec(&model);
        assert!(input_tensor.set_data(&input_vec[..]).is_ok());
        assert!(interpreter.invoke().is_ok());
        let output: &[f32] = output_tensor.data::<f32>();

        let k: u16 = output[0..255].iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0 as u16;
        let n: u16 = output[256..512].iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0 as u16;
        let p: u16;
        if k < n {
            p = n - k;
        } else {
            p = 0
        }
        let nc: u16 = output[513..768].iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0 as u16;
        // Run algorithmic inference
        algorithm.set_params(k,p,nc);
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

fn model_to_vec(model: &Model) -> Vec<f32> {
    let z_norm_plr: f32 = ( model.get_target_erasure_rate() as f32 - 0.001872892553140462) / 0.002525792;
    let z_norm_dt: f32 = (model.get_target_delay().get::<millisecond>() as f32 - 279.1865054035207) / 266.1068757110481;
    let z_norm_rc: f32 = (model.get_channel_data_rate().get::<bit_per_second>() as f32 - 1369340110.210422) / 2275899918.247129;
    let z_norm_pe: f32 = (model.get_channel_erasure_rate() as f32 - 0.02389486158205640) / 0.04230245758583454;
    let z_norm_ts: f32 = (model.get_source_packet_interval().get::<millisecond>() as f32 - 14.03923188464618) / 21.70630890225326;
    let z_norm_rtt: f32 = (model.get_round_trip_time().get::<millisecond>() as f32 - 21.15978850815602) / 24.39285640663291;
    [
        z_norm_plr,
        z_norm_dt,
        z_norm_rc,
        z_norm_pe,
        z_norm_ts,
        z_norm_rtt,
    ].to_vec()
}
