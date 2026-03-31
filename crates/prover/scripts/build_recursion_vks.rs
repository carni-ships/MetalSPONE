use std::path::PathBuf;

use clap::Parser;
use sp1_core_machine::utils::setup_logger;
use sp1_prover::{
    components::CpuProverComponents, shapes::build_vk_map_to_file, REDUCE_BATCH_SIZE,
};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(short, long)]
    build_dir: PathBuf,
    #[clap(short, long, default_value_t = false)]
    dummy: bool,
    #[clap(short, long, default_value_t = REDUCE_BATCH_SIZE)]
    reduce_batch_size: usize,
    #[clap(long, default_value_t = 4)]
    num_compiler_workers: usize,
    #[clap(long, default_value_t = 2)]
    num_setup_workers: usize,
    #[clap(long)]
    start: Option<usize>,
    #[clap(long)]
    end: Option<usize>,
}

fn main() {
    setup_logger();
    let args = Args::parse();

    build_vk_map_to_file::<CpuProverComponents>(
        args.build_dir,
        args.reduce_batch_size,
        args.dummy,
        args.num_compiler_workers,
        args.num_setup_workers,
        args.start,
        args.end,
    )
    .unwrap();
}
