use uom::si::f64::{Information, InformationRate, Time};
use uom::si::information::byte;
use uom::si::information_rate::bit_per_second;
use uom::si::time::millisecond;

use csv::Reader;
use std::fs::File;

use crate::model::Model;

pub struct CSVHanlder {
    reader: Reader<File>,
}

impl CSVHanlder {
    pub fn from(file_name: &str) -> Self {
        Self {
            reader: Reader::from_path(file_name).unwrap(),
        }
    }
}

impl Iterator for CSVHanlder {
    type Item = Model;

    fn next(&mut self) -> Option<Model> {
        let mut iter = self.reader.records();

        let record = match iter.next() {
            Some(x) => match x {
                Ok(line) => line,
                Err(error) => panic!("Problem reading the csv file: {:?}", error),
            },
            None => return None,
        };

        let model = Model::from(
            record[0].parse::<f64>().unwrap(),
            Time::new::<millisecond>(record[1].parse::<f64>().unwrap()),
            InformationRate::new::<bit_per_second>(record[2].parse::<f64>().unwrap()),
            record[3].parse::<f64>().unwrap(),
            Time::new::<millisecond>(record[4].parse::<f64>().unwrap()),
            Information::new::<byte>(record[5].parse::<f64>().unwrap()),
            Time::new::<millisecond>(record[6].parse::<f64>().unwrap()),
            Time::new::<millisecond>(record[7].parse::<f64>().unwrap()),
            Time::new::<millisecond>(record[8].parse::<f64>().unwrap()),
        );

        Some(model)
    }
}
