# Search Algorithms

## Algorithm execution

An algorithm can be executed with the command:

```bash
cargo run --release --bin alg_name
```

with `alg_name=[deepsharq|deephec|sharq|fast|krange]` the name of any of the binary files in the `src/bin` folder.

## Input File

The code expects an `input.csv` file in the root folder with the following format:

```bash
PLR_T(prob),D_T(ms),R_C(bps),p_e(prob),T_s(ms),P_L(B),RTT(ms),D_RS(ms),D_PL(ms)
0.00008,100,3000000000,0.07,30,1500,8,1,135
```


## `#![no_std]` Support

* Run with `cargo check --no-default-features`.